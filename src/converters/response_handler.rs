use crate::config::ApiType;
use crate::converters::anthropic::AnthropicResponse;
use crate::converters::openai::OpenAIResponse;
use crate::converters::response_wrapper::ResponseWrapper;
use crate::converters::stream::convert_sse_data_line;
use crate::models::{ErrorDetail, ErrorResponse};
use axum::{
    http::StatusCode,
    response::{sse::Event, sse::Sse, IntoResponse},
    Json,
};
use bytes::Bytes;
use futures::{stream, Stream, StreamExt};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::Duration;
use tracing::{debug, warn};

pub async fn handle_non_streaming_response(
    response: reqwest::Response,
    model: String,
    source_api_type: ApiType,
    target_api_type: ApiType,
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

    let response_wrapper = if source_api_type == ApiType::OpenAI {
        match serde_json::from_value::<OpenAIResponse>(response_json) {
            Ok(resp) => {
                if target_api_type == ApiType::OpenAI {
                    ResponseWrapper::OpenAI(resp)
                } else {
                    ResponseWrapper::Anthropic(resp.into())
                }
            }
            Err(e) => {
                warn!("Failed to deserialize OpenAI response: {}", e);
                let error_response = ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Failed to deserialize response: {}", e),
                        r#type: "api_error".to_string(),
                        code: Some("deserialize_error".to_string()),
                    },
                };
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response();
            }
        }
    } else {
        match serde_json::from_value::<AnthropicResponse>(response_json) {
            Ok(resp) => {
                if target_api_type == ApiType::OpenAI {
                    ResponseWrapper::OpenAI(resp.into())
                } else {
                    ResponseWrapper::Anthropic(resp)
                }
            }
            Err(e) => {
                warn!("Failed to deserialize Anthropic response: {}", e);
                let error_response = ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Failed to deserialize response: {}", e),
                        r#type: "api_error".to_string(),
                        code: Some("deserialize_error".to_string()),
                    },
                };
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response();
            }
        }
    };

    debug!(
        "Response received with model updated to: {}\n{:?}",
        model,
        serde_json::to_string(&response_wrapper)
    );
    Json(response_wrapper).into_response()
}

pub async fn handle_streaming_response(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: String,
    source_api_type: ApiType,
    target_api_type: ApiType,
) -> axum::response::Response {
    // Track contextual state needed for conversion
    let mut previous_event = String::new();
    let mut previous_delta_type = String::new();
    let mut msg_index = 0;

    // Byte buffer to accumulate partial UTF-8 lines across chunks
    let mut pending_bytes: Vec<u8> = Vec::new();

    // Move these once into the closure to avoid per-line clones in the hot path
    let src_api = source_api_type;
    let tgt_api = target_api_type;

    let event_stream = stream
        .map(move |result| match result {
            Ok(bytes) => {
                // Accumulate bytes; handle partial lines safely without lossy conversion
                pending_bytes.extend_from_slice(&bytes);

                let mut out: Vec<Result<Event, Infallible>> = Vec::new();

                // Find and process complete lines terminated by '\n'
                loop {
                    if let Some(pos) = pending_bytes.iter().position(|&b| b == b'\n') {
                        // Consider bytes up to (but not including) the '\n'
                        let line_slice = &pending_bytes[..pos];

                        // Attempt UTF-8 conversion; if it fails, keep bytes for next chunk
                        match std::str::from_utf8(line_slice) {
                            Ok(mut line_str) => {
                                // Trim optional CR at end of line
                                if let Some(stripped) = line_str.strip_suffix('\r') {
                                    line_str = stripped;
                                }

                                debug!("raw streaming response: {:?}", line_str);

                                if line_str.starts_with("data: ") {
                                    let data = &line_str[6..];
                                    if data == "[DONE]" && tgt_api == ApiType::OpenAI {
                                        out.push(Ok(Event::default().data("[DONE]")));
                                    } else {
                                        let converted = convert_sse_data_line(
                                            &src_api,
                                            &tgt_api,
                                            data,
                                            &model,
                                            &mut previous_event,
                                            &mut previous_delta_type,
                                            &mut msg_index,
                                        );
                                        for (event_opt, payload) in converted.into_iter() {
                                            let mut ev = Event::default().data(payload);
                                            if let Some(name) = event_opt {
                                                ev = ev.event(name);
                                            }
                                            out.push(Ok(ev));
                                        }
                                    }
                                }

                                // Remove processed line including the '\n'
                                pending_bytes.drain(..=pos);
                                // Continue to look for the next line in the remaining buffer
                                continue;
                            }
                            Err(_) => {
                                // Incomplete/malformed UTF-8 at boundary; wait for more bytes
                                break;
                            }
                        }
                    } else {
                        // No full line yet; try to parse pending as a full line (common in tests)
                        if !pending_bytes.is_empty() {
                            if let Ok(mut line_str) = std::str::from_utf8(&pending_bytes) {
                                if let Some(stripped) = line_str.strip_suffix('\r') {
                                    line_str = stripped;
                                }
                                if line_str.starts_with("data: ") {
                                    let data = &line_str[6..];
                                    if data == "[DONE]" && tgt_api == ApiType::OpenAI {
                                        out.push(Ok(Event::default().data("[DONE]")));
                                        pending_bytes.clear();
                                    } else {
                                        let converted = convert_sse_data_line(
                                            &src_api,
                                            &tgt_api,
                                            data,
                                            &model,
                                            &mut previous_event,
                                            &mut previous_delta_type,
                                            &mut msg_index,
                                        );
                                        if !converted.is_empty() {
                                            for (event_opt, payload) in converted.into_iter() {
                                                let mut ev = Event::default().data(payload);
                                                if let Some(name) = event_opt {
                                                    ev = ev.event(name);
                                                }
                                                out.push(Ok(ev));
                                            }
                                            // Clear pending only when successfully parsed
                                            pending_bytes.clear();
                                        }
                                    }
                                } else if line_str.starts_with("event: ")
                                    || line_str.trim().is_empty()
                                {
                                    // Treat standalone event/empty line as complete to avoid blocking following data line
                                    pending_bytes.clear();
                                }
                            }
                        }
                        // Await more bytes to complete the line
                        break;
                    }
                }

                stream::iter(out)
            }
            Err(e) => {
                // Log upstream errors and emit an error event to help clients
                warn!("Upstream streaming error: {}", e);
                let payload = serde_json::to_string(&json!({
                    "error": {
                        "message": format!("upstream streaming error: {}", e),
                        "type": "upstream_error"
                    }
                }))
                .unwrap_or_else(|_| {
                    "{\"error\":{\"message\":\"upstream streaming error\"}}".to_string()
                });
                let ev = Event::default().event("error").data(payload);
                stream::iter(vec![Ok(ev)])
            }
        })
        .flatten();

    // Return SSE with keep-alive
    Sse::new(event_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new().interval(Duration::from_secs(1)),
        )
        .into_response()
}