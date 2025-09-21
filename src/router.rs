use crate::auth::AppState;
use crate::model_manager::Selection;
use crate::config::ApiType;
use crate::models::{ErrorResponse, ErrorDetail, ModelsResponse, ModelInfo};
use crate::converters::{
    openai::{OpenAIRequest},
    anthropic::{AnthropicRequest},
    gemini::GeminiRequest,
    request_wrapper::RequestWrapper,
    response_handler::{handle_non_streaming_response, handle_streaming_response},
};
use axum::{
    extract::State,
    http::{StatusCode},
    response::{IntoResponse},
    Json,
};
use axum::extract::Path;
use serde_json::json;
use tracing::{debug, info, warn};

#[axum_macros::debug_handler]
pub async fn openai_chat(
    State(config): State<AppState>,
    Json(openai_request): Json<OpenAIRequest>,
) -> impl IntoResponse {
    route_chat(ApiType::OpenAI, config, RequestWrapper::OpenAI(openai_request)).await
}

#[axum_macros::debug_handler]
pub async fn anthropic_chat(
    State(config): State<AppState>,
    Json(anthropic_request): Json<AnthropicRequest>,
) -> impl IntoResponse {
    route_chat(ApiType::Anthropic, config, RequestWrapper::Anthropic(anthropic_request)).await
}

// Gemini API entrypoint compatible with:
// - POST /models/{model}:generateContent
// - POST /models/{model}:streamGenerateContent?alt=sse
#[axum_macros::debug_handler]
pub async fn gemini_chat(
    State(config): State<AppState>,
    Path(path_tail): Path<String>,
    Json(mut body): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Parse model from tail like "models/{model}:generateContent" or "models/{model}:streamGenerateContent"
    // Our route is defined as /models/*tail, so tail includes "{model}:..."
    let (model, is_stream) = match path_tail.rsplit_once(":") {
        Some((model_part, action)) => {
            let model = model_part.to_string();
            let is_stream = action == "streamGenerateContent";
            (model, is_stream)
        }
        None => {
            let error = json!({
                "error": {"message": "invalid Gemini path", "type": "invalid_request"}
            });
            return (StatusCode::BAD_REQUEST, Json(error)).into_response();
        }
    };

    // Inject routing fields expected by our types
    body["model"] = json!(model);
    body["stream"] = json!(is_stream);

    let gemini_request: GeminiRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            let error = json!({"error": {"message": format!("invalid request: {}", e), "type": "invalid_request"}});
            return (StatusCode::BAD_REQUEST, Json(error)).into_response();
        }
    };

    route_chat(ApiType::Gemini, config, RequestWrapper::Gemini(gemini_request)).await.into_response()
}


pub async fn route_chat(
    api_type: ApiType,
    config: AppState,
    request_wrapper: RequestWrapper,
) -> axum::response::Response {
    
    // Parse the request into the appropriate structure based on API type
    let model = request_wrapper.get_model();
    
    let stream = request_wrapper.is_stream().unwrap_or(false);
    
    debug!("raw request: {}", serde_json::to_string(&request_wrapper).expect("Failed to serialize request"));

    // Narrow read-lock scope to selection only
    let selection: Selection = {
        let model_manager = config.model_manager.read().await;
        let request_json = serde_json::to_value(&request_wrapper).unwrap_or_else(|_| json!({}));
        match model_manager.resolve(model, &request_json) {
            Some(sel) => {
                debug!("Resolved model selection for: {} -> {:?}", model, sel);
                sel
            }
            None => {
                info!("Model '{}' not found in configuration", model);
                let error_response = ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Model '{}' not found", model),
                        r#type: "invalid_request_error".to_string(),
                        code: Some("model_not_found".to_string()),
                    },
                };
                return (StatusCode::NOT_FOUND, Json(error_response)).into_response();
            }
        }
    };

    // Track the start of the request
    {
        let model_manager = config.model_manager.read().await;
        model_manager.start(&selection);
    }

    let response = config
        .llm_client
        .forward_request(&request_wrapper, &selection.config);
    let response = match response.await {
        Ok(resp) => resp,
        Err(e) => {
            warn!("Failed to send streaming request: {}", e);
            // Track the failed request
            {
                let model_manager = config.model_manager.read().await;
                model_manager.end(&selection, false);
            }
            let error_response = ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to send request: {}", e),
                    r#type: "api_error".to_string(),
                    code: Some("request_failed".to_string()),
                },
            };
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response();
        }
    };
    if !response.status().is_success() {
        use axum::http::header::CONTENT_TYPE;
        let status = response.status();
        let content_type = response.headers().get(CONTENT_TYPE).cloned();
        let body_bytes = response.bytes().await.unwrap_or_default();
        warn!("Upstream request failed with status {}", status);
        // Track the failed request
        {
            let model_manager = config.model_manager.read().await;
            model_manager.end(&selection, false);
        }

        let mut resp = (status, body_bytes).into_response();
        if let Some(ct) = content_type { resp.headers_mut().insert(CONTENT_TYPE, ct); }
        return resp;
    }
    // Handle streaming and non-streaming responses
    if stream {
        info!("Processing streaming request");
        let result = handle_streaming_response(
            response.bytes_stream(),
            model.to_string(),
            selection.config.llm_params.api_type.clone(),
            api_type.clone(),
        ).await;
        // Track the successful completion of streaming request
        {
            let model_manager = config.model_manager.read().await;
            model_manager.end(&selection, true);
        }
        result
    } else {
        info!("Processing non-streaming request");
        let result = handle_non_streaming_response(
            response,
            model.to_string(),
            selection.config.llm_params.api_type.clone(),
            api_type.clone(),
        ).await;
        // Track the successful completion of non-streaming request
        {
            let model_manager = config.model_manager.read().await;
            model_manager.end(&selection, true);
        }
        result
    }
}


#[axum_macros::debug_handler]
pub async fn list_models(
    State(config): State<AppState>,
) -> impl IntoResponse {
    debug!("Received models list request");
    
    let model_groups = {
        let model_manager = config.model_manager.read().await;
        let cfg = model_manager.get_config();
        cfg.router_settings.model_groups.clone()
    };
    
    let mut models = Vec::new();
    
    // Add all model group aliases
    for model_group in &model_groups {
        models.push(ModelInfo {
            id: model_group.name.clone(),
            object: "model".to_string()
        });
    }
    
    let response = ModelsResponse {
        object: "list".to_string(),
        data: models,
    };
    
    debug!("Returning {} models", response.data.len());
    Json(response).into_response()
}
