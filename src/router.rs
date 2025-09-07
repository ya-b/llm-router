use crate::auth::AppState;
use crate::models::{ErrorResponse, ErrorDetail};
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::{Event, Sse}, IntoResponse},
    Json,
};
use futures::StreamExt;
use std::time::Duration;
use bytes::Bytes;
use futures::{stream, Stream};
use reqwest::Client;
use serde_json::{json, Value};
use std::convert::Infallible;
use tracing::{debug, error, info, warn};

#[axum_macros::debug_handler]
pub async fn chat_completion(
    State(config): State<AppState>,
    Json(request_json): Json<Value>,
) -> impl IntoResponse {
    // Extract only the model and stream fields, making the request parsing extensible
    let model = request_json.get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("");
    
    let stream = request_json.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    
    info!("Received chat completion request for model: {}", model);
    debug!("Request details: model={}, stream={}", model, stream);

    // Get model configuration
    let cfg = config.config.read().await;
    let model_config = match cfg.get_model_config(model) {
        Some(config) => {
            info!("Found model configuration for: {}", model);
            config
        }
        None => {
            warn!("Model '{}' not found in configuration", model);
            let error_response = ErrorResponse {
                error: ErrorDetail {
                    message: format!("Model '{}' not found", model),
                    r#type: "invalid_request_error".to_string(),
                    code: Some("model_not_found".to_string()),
                },
            };
            return (StatusCode::NOT_FOUND, Json(error_response)).into_response();
        }
    };

    // Forward request to the target API
    let client = Client::new();
    let target_url = format!("{}/chat/completions", model_config.llm_params.api_base);
    info!("Forwarding request to: {}", target_url);

    let target_request = client
        .post(&target_url)
        .header("Authorization", format!("Bearer {}", model_config.llm_params.api_key))
        .header("Content-Type", "application/json");

    // Use the original request JSON as the target body
    let mut target_body = request_json.clone();

    // Update the model in the request body
    if let Some(obj) = target_body.as_object_mut() {
        let original_model = obj.get("model").cloned().unwrap_or(json!(null));
        obj.insert("model".to_string(), json!(model_config.llm_params.model));
        debug!("Model mapping: {:?} -> {:?}", original_model, model_config.llm_params.model);
    }

    // Handle streaming and non-streaming responses
    if stream {
        info!("Processing streaming request");
        let response = match target_request.json(&target_body).send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to send streaming request: {}", e);
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

        if response.status().is_success() {
            info!("Streaming request successful");
            handle_streaming_response(response.bytes_stream(), model.to_string()).await.into_response()
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            error!("Streaming request failed with status {}: {}", status, error_text);
            let error_response = ErrorResponse {
                error: ErrorDetail {
                    message: format!("API error: {}", error_text),
                    r#type: "api_error".to_string(),
                    code: Some("api_error".to_string()),
                },
            };
            (StatusCode::BAD_REQUEST, Json(error_response)).into_response()
        }
    } else {
        info!("Processing non-streaming request");
        let response = match target_request.json(&target_body).send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to send non-streaming request: {}", e);
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

        if response.status().is_success() {
            info!("Non-streaming request successful");
            let mut response_json: Value = match response.json().await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Failed to parse response: {}", e);
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

            // Replace the model field in the response with the requested model
            if let Some(obj) = response_json.as_object_mut() {
                obj.insert("model".to_string(), json!(model));
            }

            debug!("Response received with model updated to: {}", model);
            Json(response_json).into_response()
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            error!("Non-streaming request failed with status {}: {}", status, error_text);
            let error_response = ErrorResponse {
                error: ErrorDetail {
                    message: format!("API error: {}", error_text),
                    r#type: "api_error".to_string(),
                    code: Some("api_error".to_string()),
                },
            };
            (StatusCode::BAD_REQUEST, Json(error_response)).into_response()
        }
    }
}

async fn handle_streaming_response(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send + 'static> {
    let event_stream = stream.map(move |result| match result {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.split('\n').collect();
            
            let event_stream: Vec<Result<Event, Infallible>> = lines.into_iter()
                .filter_map(|line| {
                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if data == "[DONE]" {
                            Some(Ok(Event::default().data("[DONE]")))
                        } else {
                            match serde_json::from_str::<serde_json::Value>(data) {
                                Ok(mut json) => {
                                    // Replace the model field if it exists
                                    if let Some(obj) = json.as_object_mut() {
                                        obj.insert("model".to_string(), json!(model));
                                    }
                                    
                                    // Serialize the modified JSON back to string
                                    if let Ok(modified_data) = serde_json::to_string(&json) {
                                        // Create a new SSE event with the modified data
                                        Some(Ok(Event::default().data(modified_data)))
                                    } else {
                                        // Fallback to original data if serialization fails
                                        Some(Ok(Event::default().data(data.to_string())))
                                    }
                                }
                                Err(_) => None,
                            }
                        }
                    } else {
                        None
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
    )
}

pub async fn health_check() -> &'static str {
    debug!("Health check endpoint called");
    "OK"
}