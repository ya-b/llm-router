use crate::auth::AppState;
use crate::config::ModelConfig;
use crate::models::{ErrorResponse, ErrorDetail, ModelsResponse, ModelInfo};
use crate::converter::ApiConverter;
use anyhow::Result;
use axum::{
    extract::State,
    http::{StatusCode, Uri},
    response::{sse::{Event, Sse}, IntoResponse},
    Json,
};
use futures::StreamExt;
use std::time::Duration;
use bytes::Bytes;
use futures::{stream, Stream};
use serde_json::{json, Value};
use std::convert::Infallible;
use tracing::{debug, info, warn};


fn get_api_type_from_uri(uri: &Uri) -> String {
    if uri.path() == "/v1/messages" {
        "anthropic".to_string()
    } else {
        "openai".to_string()
    }
}

fn build_target_url(model_config: &ModelConfig) -> String {
    let api_base = &model_config.llm_params.api_base;
    let path = if model_config.llm_params.api_type == "anthropic" { "v1/messages" } else { "chat/completions" };
    
    // Handle the case where api_base might end with a '/'
    if api_base.ends_with('/') {
        format!("{}{}", api_base, path)
    } else {
        format!("{}/{}", api_base, path)
    }
}

fn request(api_type: &String, request_json: &Value, model_config: &ModelConfig) -> impl Future<Output = Result<reqwest::Response, reqwest::Error>> {
    // Forward request to the target API
    let client = reqwest::Client::new();
    let target_url = build_target_url(model_config);
    info!("Forwarding request to: {}", target_url);
    debug!("raw request: {:?}", serde_json::to_string(&request_json));

    let mut target_request = client
        .post(&target_url)
        .header("Authorization", format!("Bearer {}", model_config.llm_params.api_key))
        .header("Content-Type", "application/json");
    if model_config.llm_params.api_type == "anthropic" {
        target_request = target_request.header("x-api-key", model_config.llm_params.api_key.to_string());
    }

    // Use the original request JSON as the target body
    let mut target_body: Value;
    if api_type.eq(&model_config.llm_params.api_type) {
        target_body = request_json.clone();
    } else if model_config.llm_params.api_type == "anthropic" {
        target_body = ApiConverter::openai_to_anthropic_request(&request_json);
    } else {
        target_body = ApiConverter::anthropic_to_openai_request(&request_json);
    };

    // Update the model in the request body
    if let Some(obj) = target_body.as_object_mut() {
        let original_model = obj.get("model").cloned().unwrap_or(json!(null));
        obj.insert("model".to_string(), json!(model_config.llm_params.model));
        debug!("Model mapping: {:?} -> {:?}", original_model, model_config.llm_params.model);
    }
    debug!("request {:?}: {:?}", &target_url, serde_json::to_string(&target_body));
    target_request.json(&target_body).send()
}


#[axum_macros::debug_handler]
pub async fn chat_completion(
    uri: Uri,
    State(config): State<AppState>,
    Json(request_json): Json<Value>,
) -> impl IntoResponse {
    let api_type = get_api_type_from_uri(&uri);
    // Extract only the model and stream fields, making the request parsing extensible
    let model = request_json.get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("");
    
    let stream = request_json.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    
    debug!("Received chat completion request for model: {}", model);
    debug!("Request details: model={}, stream={}", model, stream);

    // Get model configuration using model manager
    let model_config = {
        let model_manager = config.model_manager.read().await;
        match model_manager.get_model_config(model) {
            Some(config) => {
                info!("Found model configuration for: {}", model);
                Some(config.clone())
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
    }.unwrap(); // Safe to unwrap because we return on None
    let response = request(&api_type, &request_json, &model_config);
    let response = match response.await {
        Ok(resp) => resp,
        Err(e) => {
            warn!("Failed to send streaming request: {}", e);
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
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        warn!("Non-streaming request failed with status {}: {}", status, error_text);
        let error_response = ErrorResponse {
            error: ErrorDetail {
                message: format!("API error: {}", error_text),
                r#type: "api_error".to_string(),
                code: Some("api_error".to_string()),
            },
        };
        return (StatusCode::BAD_REQUEST, Json(error_response)).into_response()
    }
    // Handle streaming and non-streaming responses
    if stream {
        info!("Processing streaming request");
        handle_streaming_response(response.bytes_stream(), model.to_string(), model_config.llm_params.api_type.clone(), api_type.to_string()).await
    } else {
        info!("Processing non-streaming request");
        handle_non_streaming_response(response, model.to_string(), model_config.llm_params.api_type.clone(), api_type.to_string()).await
    }
}


async fn handle_non_streaming_response(
    response: reqwest::Response,
    model: String,
    source_api_type: String,
    target_api_type: String
) -> axum::response::Response {

    let mut response_json: Value = match response.json().await {
        Ok(resp) => resp,
        Err(e) => {
            warn!("Failed to parse response: {}", e);
            let error_response = ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to parse response: {}", e),
                    r#type: "api_error".to_string(),
                    code: Some("parse_error".to_string()),
                },
            };
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response();
        }
    };
    debug!("raw response: {:?}", serde_json::to_string(&response_json));
    // Replace the model field in the response with the requested model
    if let Some(obj) = response_json.as_object_mut() {
        obj.insert("model".to_string(), json!(model));
    }

    if source_api_type != target_api_type && target_api_type == "anthropic" {
        response_json = ApiConverter::openai_to_anthropic_response(&response_json);
    } else if source_api_type != target_api_type && target_api_type == "openai" {
        response_json = ApiConverter::anthropic_to_openai_response(&response_json);
    }

    debug!("Response received with model updated to: {}", model);
    Json(response_json).into_response()
}

async fn handle_streaming_response(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: String,
    source_api_type: String,
    target_api_type: String
) -> axum::response::Response {
    let mut previous_event = "".to_string();
    let mut previous_delta_type = "".to_string();
    let mut msg_index = 0;
    let event_stream = stream.map(move |result| match result {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.split('\n').collect();
            let event_stream: Vec<Result<Event, Infallible>> = lines.into_iter()
                .flat_map(|line| {
                    debug!("raw streaming response: {:?}", line);
                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if data == "[DONE]" && target_api_type == "openai" {
                            vec![Ok(Event::default().data("[DONE]"))]
                        } else {
                            match serde_json::from_str::<serde_json::Value>(data) {
                                Ok(mut json) => {
                                    let mut results = vec![];
                                    // Replace the model field if it exists
                                    if let Some(obj) = json.as_object_mut() {
                                        obj.insert("model".to_string(), json!(model));
                                    }
                                    if source_api_type != target_api_type && target_api_type == "openai" {
                                        json = ApiConverter::anthropic_to_openai_stream_chunk(&json);
                                        if let Ok(modified_data) = serde_json::to_string(&json) {
                                            results.push(Ok(Event::default().data(modified_data)));
                                        }
                                    } else if source_api_type != target_api_type && target_api_type == "anthropic" {
                                        results = ApiConverter::openai_to_anthropic_stream_chunks(&json, &model, &mut previous_event, &mut previous_delta_type, &mut msg_index)
                                                    .into_iter()
                                                    .map(|(a, b)| Ok(Event::default().event(a).data(b)))
                                                    .collect();
                                    } else {
                                        if let Ok(modified_data) = serde_json::to_string(&json) {
                                            if target_api_type == "openai" {
                                                results.push(Ok(Event::default().data(modified_data)));
                                            } else if target_api_type == "anthropic" {
                                                results.push(Ok(Event::default().event(json.get("type").and_then(|t| t.as_str()).unwrap_or("")).data(modified_data)));
                                            }
                                        }
                                    }
                                    results
                                }
                                Err(_) => vec![],
                            }
                        }
                    } else {
                        vec![]
                    }
                })
                .collect();
            
            stream::iter(event_stream)
        }
        Err(_) => stream::iter(vec![]),
    })
    .flatten();
    
    // Return SSE with keep-alive
    Sse::new(event_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(1)),
    ).into_response()
}

#[axum_macros::debug_handler]
pub async fn list_models(
    State(config): State<AppState>,
) -> impl IntoResponse {
    info!("Received models list request");
    
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
    
    info!("Returning {} models", response.data.len());
    Json(response).into_response()
}
