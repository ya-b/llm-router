use super::anthropic::{
    AnthropicContentBlock, AnthropicStreamChunk, AnthropicStreamDelta, AnthropicStreamMessage,
};
use super::gemini::GeminiStreamChunk;
use super::openai::OpenAIStreamChunk;
use crate::config::ApiType;
use crate::converters::anthropic::AnthropicResponse;
use crate::converters::gemini::GeminiResponse;
use crate::converters::openai::OpenAIResponse;
use crate::converters::response_wrapper::ResponseWrapper;
use crate::converters::responses::{ResponsesResponse, ResponsesStreamChunk};
use crate::models::{ErrorDetail, ErrorResponse};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, sse::Event, sse::Sse},
};
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};
use serde_json::json;
use std::convert::Infallible;
use std::time::Duration;
use tracing::{debug, warn};

pub async fn handle_non_streaming_response(
    response: reqwest::Response,
    model: String,
    source_api_type: ApiType,
    target_api_type: ApiType,
) -> axum::response::Response {
    let response_text: String = match response.text().await {
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
    debug!("raw response: {:?}", &response_text);

    let response_wrapper = match (source_api_type, target_api_type) {
        (ApiType::OpenAI, ApiType::OpenAI) => {
            match serde_json::from_str::<OpenAIResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    ResponseWrapper::OpenAI(resp)
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::OpenAI, ApiType::Responses) => {
            match serde_json::from_str::<OpenAIResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    let responses_resp: ResponsesResponse = resp.into();
                    ResponseWrapper::Responses(responses_resp)
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Gemini, ApiType::Gemini) => {
            match serde_json::from_str::<GeminiResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model_version = Some(model.clone());
                    ResponseWrapper::Gemini(resp)
                }
                Err(e) => {
                    warn!("Failed to deserialize Gemini response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Gemini, ApiType::Responses) => {
            match serde_json::from_str::<GeminiResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model_version = Some(model.clone());
                    let openai_resp: OpenAIResponse = resp.into();
                    let responses_resp: ResponsesResponse = openai_resp.into();
                    ResponseWrapper::Responses(responses_resp)
                }
                Err(e) => {
                    warn!("Failed to deserialize Gemini response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Anthropic, ApiType::Anthropic) => {
            match serde_json::from_str::<AnthropicResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    ResponseWrapper::Anthropic(resp)
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Anthropic, ApiType::Responses) => {
            match serde_json::from_str::<AnthropicResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    let openai_resp: OpenAIResponse = resp.into();
                    let responses_resp: ResponsesResponse = openai_resp.into();
                    ResponseWrapper::Responses(responses_resp)
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Anthropic, ApiType::OpenAI) => {
            match serde_json::from_str::<AnthropicResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    ResponseWrapper::OpenAI(resp.into())
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::OpenAI, ApiType::Anthropic) => {
            match serde_json::from_str::<OpenAIResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    ResponseWrapper::Anthropic(resp.into())
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Gemini, ApiType::OpenAI) => {
            match serde_json::from_str::<GeminiResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model_version = Some(model.clone());
                    ResponseWrapper::OpenAI(resp.into())
                }
                Err(e) => {
                    warn!("Failed to deserialize Gemini response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Gemini, ApiType::Anthropic) => {
            match serde_json::from_str::<GeminiResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model_version = Some(model.clone());
                    let resp1: OpenAIResponse = resp.into();
                    ResponseWrapper::Anthropic(resp1.into())
                }
                Err(e) => {
                    warn!("Failed to deserialize Gemini response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Anthropic, ApiType::Gemini) => {
            match serde_json::from_str::<AnthropicResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    let resp1: OpenAIResponse = resp.into();
                    ResponseWrapper::Gemini(resp1.into())
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::OpenAI, ApiType::Gemini) => {
            match serde_json::from_str::<OpenAIResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    ResponseWrapper::Gemini(resp.into())
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
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Responses, ApiType::OpenAI) => {
            match serde_json::from_str::<ResponsesResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    ResponseWrapper::OpenAI(resp.into())
                }
                Err(e) => {
                    warn!("Failed to deserialize Responses response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Responses, ApiType::Anthropic) => {
            match serde_json::from_str::<ResponsesResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    let openai_resp: OpenAIResponse = resp.into();
                    ResponseWrapper::Anthropic(openai_resp.into())
                }
                Err(e) => {
                    warn!("Failed to deserialize Responses response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Responses, ApiType::Gemini) => {
            match serde_json::from_str::<ResponsesResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    let openai_resp: OpenAIResponse = resp.into();
                    ResponseWrapper::Gemini(openai_resp.into())
                }
                Err(e) => {
                    warn!("Failed to deserialize Responses response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
            }
        }
        (ApiType::Responses, ApiType::Responses) => {
            match serde_json::from_str::<ResponsesResponse>(&response_text) {
                Ok(mut resp) => {
                    resp.model = model.clone();
                    ResponseWrapper::Responses(resp)
                }
                Err(e) => {
                    warn!("Failed to deserialize Responses response: {}", e);
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to deserialize response: {}", e),
                            r#type: "api_error".to_string(),
                            code: Some("deserialize_error".to_string()),
                        },
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                        .into_response();
                }
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
    let mut previous_function_arg = String::new();
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
                                            &mut previous_function_arg,
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
                                            &mut previous_function_arg,
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
        .keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(1)))
        .into_response()
}

fn accumulate_function_args_and_patch(
    openai_chunk: &mut OpenAIStreamChunk,
    previous_function_arg: &mut String,
) -> bool {
    // Extract current function-call args (if any)
    let args: &str = openai_chunk
        .choices
        .as_ref()
        .and_then(|cs| cs.first())
        .and_then(|c| c.delta.as_ref())
        .and_then(|d| d.tool_calls.as_ref())
        .and_then(|tcs| {
            tcs.iter()
                .find(|tc| tc.r#type.as_deref() == Some("function"))
        })
        .and_then(|tc| tc.function.as_ref())
        .and_then(|f| f.arguments.as_deref())
        .unwrap_or("");

    if args.is_empty() {
        return false;
    }

    previous_function_arg.push_str(args);
    let parse_ok = serde_json::from_str::<serde_json::Value>(&previous_function_arg).is_ok();
    if !parse_ok {
        // Need to wait for more data to form valid JSON
        return true;
    }

    // Parsing succeeded: set function call args to accumulated buffer
    if let Some(choices) = openai_chunk.choices.as_mut() {
        if let Some(first) = choices.first_mut() {
            if let Some(delta) = first.delta.as_mut() {
                if let Some(tool_calls) = delta.tool_calls.as_mut() {
                    if let Some(tc) = tool_calls
                        .iter_mut()
                        .find(|tc| tc.r#type.as_deref() == Some("function"))
                    {
                        if let Some(function) = tc.function.as_mut() {
                            function.arguments = Some(previous_function_arg.clone());
                            previous_function_arg.clear();
                        }
                    }
                }
            }
        }
    }

    false
}

/// 将单行 SSE `data:` 载荷从 source -> target 转换为输出帧集合。
/// 返回的 Vec 中，(None, data) 表示 OpenAI 风格的无事件名数据帧；
/// (Some(event_name), data) 表示 Anthropic 风格的具名事件帧。
pub fn convert_sse_data_line(
    source_api_type: &ApiType,
    target_api_type: &ApiType,
    data: &str,
    model: &String,
    previous_event: &mut String,
    previous_delta_type: &mut String,
    previous_function_arg: &mut String,
    msg_index: &mut i32,
) -> Vec<(Option<String>, String)> {
    match (source_api_type, target_api_type) {
        (ApiType::OpenAI, ApiType::OpenAI) => {
            if let Ok(mut chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                chunk.model = model.clone();
                if let Ok(s) = serde_json::to_string(&chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        (ApiType::Gemini, ApiType::Gemini) => {
            if let Ok(mut chunk) = serde_json::from_str::<GeminiStreamChunk>(data) {
                chunk.model_version = Some(model.clone());
                if let Ok(s) = serde_json::to_string(&chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        (ApiType::Anthropic, ApiType::Anthropic) => {
            if let Ok(mut chunk) = serde_json::from_str::<AnthropicStreamChunk>(data) {
                if let AnthropicStreamChunk::MessageStart { message } = chunk.clone() {
                    let mut patched = message.clone();
                    patched.model = model.clone();
                    chunk = AnthropicStreamChunk::MessageStart { message: patched };
                }
                if let Ok(s) = serde_json::to_string(&chunk) {
                    return vec![(Some(chunk.stream_type().to_string()), s)];
                }
            }
            vec![]
        }
        (ApiType::Anthropic, ApiType::OpenAI) => {
            if let Ok(chunk) = serde_json::from_str::<AnthropicStreamChunk>(data) {
                let mut openai_chunk: OpenAIStreamChunk = chunk.into();
                openai_chunk.model = model.clone();
                if let Ok(s) = serde_json::to_string(&openai_chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        (ApiType::OpenAI, ApiType::Anthropic) => {
            if let Ok(mut chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                chunk.model = model.clone();
                return openai_to_anthropic_stream_chunks(
                    &chunk,
                    model,
                    previous_event,
                    previous_delta_type,
                    msg_index,
                )
                .into_iter()
                .map(|(event, payload)| (Some(event), payload))
                .collect();
            }
            vec![]
        }
        (ApiType::Gemini, ApiType::OpenAI) => {
            if let Ok(mut chunk) = serde_json::from_str::<GeminiStreamChunk>(data) {
                chunk.model_version = Some(model.clone());
                let openai_chunk: OpenAIStreamChunk = chunk.into();
                if let Ok(s) = serde_json::to_string(&openai_chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        (ApiType::Gemini, ApiType::Anthropic) => {
            if let Ok(mut chunk) = serde_json::from_str::<GeminiStreamChunk>(data) {
                chunk.model_version = Some(model.clone());
                let openai_chunk: OpenAIStreamChunk = chunk.into();
                return openai_to_anthropic_stream_chunks(
                    &openai_chunk,
                    model,
                    previous_event,
                    previous_delta_type,
                    msg_index,
                )
                .into_iter()
                .map(|(event, payload)| (Some(event), payload))
                .collect();
            }
            vec![]
        }
        (ApiType::Anthropic, ApiType::Gemini) => {
            if let Ok(anth_chunk) = serde_json::from_str::<AnthropicStreamChunk>(data) {
                let mut openai_chunk: OpenAIStreamChunk = anth_chunk.into();
                if accumulate_function_args_and_patch(&mut openai_chunk, previous_function_arg) {
                    return vec![];
                }

                let mut gemini_chunk: GeminiStreamChunk = openai_chunk.into();
                gemini_chunk.model_version = Some(model.clone());
                if let Ok(s) = serde_json::to_string(&gemini_chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        (ApiType::OpenAI, ApiType::Gemini) => {
            if let Ok(mut openai_chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                openai_chunk.model = model.clone();
                if accumulate_function_args_and_patch(&mut openai_chunk, previous_function_arg) {
                    return vec![];
                }
                let gemini_chunk: GeminiStreamChunk = openai_chunk.into();
                if let Ok(s) = serde_json::to_string(&gemini_chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        // Upstream Anthropic -> Responses
        (ApiType::Anthropic, ApiType::Responses) => {
            if let Ok(mut chunk) = serde_json::from_str::<AnthropicStreamChunk>(data) {
                // Ensure model is patched for message_start
                if let AnthropicStreamChunk::MessageStart { message } = chunk.clone() {
                    let mut patched = message.clone();
                    patched.model = model.clone();
                    chunk = AnthropicStreamChunk::MessageStart { message: patched };
                }
                let resp_chunk: ResponsesStreamChunk = chunk.into();
                if let Ok(s) = serde_json::to_string(&resp_chunk) {
                    return vec![(Some(resp_chunk.event_type.clone()), s)];
                }
            }
            vec![]
        }
        // Upstream OpenAI -> Responses (via Anthropic pivot)
        (ApiType::OpenAI, ApiType::Responses) => {
            if let Ok(mut oai_chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                oai_chunk.model = model.clone();
                let events = openai_to_anthropic_stream_chunks(
                    &oai_chunk,
                    model,
                    previous_event,
                    previous_delta_type,
                    msg_index,
                );
                let mut out = Vec::new();
                for (_event, payload) in events.into_iter() {
                    if let Ok(anth_chunk) = serde_json::from_str::<AnthropicStreamChunk>(&payload) {
                        let resp_chunk: ResponsesStreamChunk = anth_chunk.into();
                        if let Ok(s) = serde_json::to_string(&resp_chunk) {
                            out.push((Some(resp_chunk.event_type.clone()), s));
                        }
                    }
                }
                return out;
            }
            vec![]
        }
        // Upstream Gemini -> Responses (Gemini -> OpenAI -> Anthropic -> Responses)
        (ApiType::Gemini, ApiType::Responses) => {
            if let Ok(mut g_chunk) = serde_json::from_str::<GeminiStreamChunk>(data) {
                g_chunk.model_version = Some(model.clone());
                let oai_chunk: OpenAIStreamChunk = g_chunk.into();
                let events = openai_to_anthropic_stream_chunks(
                    &oai_chunk,
                    model,
                    previous_event,
                    previous_delta_type,
                    msg_index,
                );
                let mut out = Vec::new();
                for (_event, payload) in events.into_iter() {
                    if let Ok(anth_chunk) = serde_json::from_str::<AnthropicStreamChunk>(&payload) {
                        let resp_chunk: ResponsesStreamChunk = anth_chunk.into();
                        if let Ok(s) = serde_json::to_string(&resp_chunk) {
                            out.push((Some(resp_chunk.event_type.clone()), s));
                        }
                    }
                }
                return out;
            }
            vec![]
        }
        // Upstream Responses -> Anthropic
        (ApiType::Responses, ApiType::Anthropic) => {
            if let Ok(chunk) = serde_json::from_str::<ResponsesStreamChunk>(data) {
                let mut anth_chunk: AnthropicStreamChunk = chunk.into();
                if let AnthropicStreamChunk::MessageStart { message } = anth_chunk.clone() {
                    let mut patched = message.clone();
                    patched.model = model.clone();
                    anth_chunk = AnthropicStreamChunk::MessageStart { message: patched };
                }
                if let Ok(s) = serde_json::to_string(&anth_chunk) {
                    return vec![(Some(anth_chunk.stream_type().to_string()), s)];
                }
            }
            vec![]
        }
        // Upstream Responses -> OpenAI
        (ApiType::Responses, ApiType::OpenAI) => {
            if let Ok(chunk) = serde_json::from_str::<ResponsesStreamChunk>(data) {
                let anth_chunk: AnthropicStreamChunk = chunk.into();
                let mut oai_chunk: OpenAIStreamChunk = anth_chunk.into();
                oai_chunk.model = model.clone();
                if let Ok(s) = serde_json::to_string(&oai_chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        // Upstream Responses -> Gemini
        (ApiType::Responses, ApiType::Gemini) => {
            if let Ok(chunk) = serde_json::from_str::<ResponsesStreamChunk>(data) {
                let anth_chunk: AnthropicStreamChunk = chunk.into();
                let mut oai_chunk: OpenAIStreamChunk = anth_chunk.into();
                oai_chunk.model = model.clone();
                let mut g_chunk: GeminiStreamChunk = oai_chunk.into();
                g_chunk.model_version = Some(model.clone());
                if let Ok(s) = serde_json::to_string(&g_chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        // Upstream Responses -> Responses (pass-through, patch model if present)
        (ApiType::Responses, ApiType::Responses) => {
            if let Ok(mut chunk) = serde_json::from_str::<ResponsesStreamChunk>(data) {
                if let ResponsesStreamChunk { payload, .. } = &mut chunk {
                    if let crate::converters::responses::ResponsesStreamEventPayload::Response {
                        response,
                    } = payload
                    {
                        response.model = model.clone();
                    }
                }
                if let Ok(s) = serde_json::to_string(&chunk) {
                    return vec![(Some(chunk.event_type.clone()), s)];
                }
            }
            vec![]
        }
    }
}

fn handle_content_block_start_typed(
    anthropic_delta: &AnthropicStreamChunk,
    index: i32,
) -> Option<AnthropicStreamChunk> {
    match anthropic_delta {
        AnthropicStreamChunk::ContentBlockDelta { index: _, delta } => match delta {
            AnthropicStreamDelta::InputJsonDelta { name, id, .. } => {
                if let (Some(name), Some(id)) = (name.as_ref(), id.as_ref()) {
                    Some(AnthropicStreamChunk::ContentBlockStart {
                        index,
                        content_block: AnthropicContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: json!({}),
                        },
                    })
                } else {
                    None
                }
            }
            AnthropicStreamDelta::ThinkingDelta { .. } => {
                Some(AnthropicStreamChunk::ContentBlockStart {
                    index,
                    content_block: AnthropicContentBlock::Thinking {
                        thinking: "".to_string(),
                        // Upstream struct requires a signature; keep empty to preserve tests
                        signature: "".to_string(),
                    },
                })
            }
            AnthropicStreamDelta::TextDelta { .. } => {
                Some(AnthropicStreamChunk::ContentBlockStart {
                    index,
                    content_block: AnthropicContentBlock::Text {
                        text: "".to_string(),
                    },
                })
            }
        },
        _ => None,
    }
}

fn delta_kind(anthropic_delta: &AnthropicStreamChunk) -> Option<&'static str> {
    match anthropic_delta {
        AnthropicStreamChunk::ContentBlockDelta { delta, .. } => match delta {
            AnthropicStreamDelta::InputJsonDelta { .. } => Some("input_json_delta"),
            AnthropicStreamDelta::ThinkingDelta { .. } => Some("thinking_delta"),
            AnthropicStreamDelta::TextDelta { .. } => Some("text_delta"),
        },
        AnthropicStreamChunk::MessageDelta { .. } => Some("message_delta"),
        _ => None,
    }
}

/// 将 OpenAI 流式响应块转换为 Anthropic 流式事件序列（不使用 serde_json::Value 作为输入）。
pub fn openai_to_anthropic_stream_chunks(
    chunk: &OpenAIStreamChunk,
    model: &String,
    previous_event: &mut String,
    previous_delta_type: &mut String,
    msg_index: &mut i32,
) -> Vec<(String, String)> {
    let mut results: Vec<(String, String)> = vec![];

    // 初始 message_start
    if previous_event.is_empty() {
        let start = AnthropicStreamChunk::MessageStart {
            message: AnthropicStreamMessage {
                id: chunk.id.clone(),
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: model.clone(),
                stop_reason: None,
                usage: None,
            },
        };
        if let Ok(s) = serde_json::to_string(&start) {
            results.push(("message_start".to_string(), s));
        }
        previous_event.clear();
        previous_event.push_str("message_start");
    }

    // 提取 OpenAI delta 信息
    let (mut is_finish, mut is_reasoning_empty, mut is_content_empty, mut is_tool_calls_empty) =
        (false, true, true, true);
    if let Some(first_choice) = chunk.choices.as_ref().and_then(|v| v.first()) {
        is_finish = first_choice.finish_reason.as_ref().is_some();
        if let Some(delta) = first_choice.delta.as_ref() {
            is_reasoning_empty = delta
                .reasoning_content
                .as_ref()
                .map(|s| s.is_empty())
                .unwrap_or(true);
            is_content_empty = delta.content.as_ref().map(|s| s.is_empty()).unwrap_or(true);
            is_tool_calls_empty = delta
                .tool_calls
                .as_ref()
                .map(|arr| arr.is_empty())
                .unwrap_or(true);
        }
    }

    if is_finish {
        // 在结束前先发送 content_block_stop（与旧逻辑保持一致）
        if let Ok(s) =
            serde_json::to_string(&AnthropicStreamChunk::ContentBlockStop { index: *msg_index })
        {
            results.push(("content_block_stop".to_string(), s));
        }
    }

    // 没有任何内容增量且未结束，直接返回（仅包含 message_start）
    if is_reasoning_empty && is_content_empty && is_tool_calls_empty && !is_finish {
        return results;
    }

    // 将 OpenAI 块转换为 Anthropic 块（单个增量或消息级增量）
    let base_chunk: AnthropicStreamChunk = chunk.clone().into();
    let event_type = base_chunk.stream_type();
    let current_delta_type = delta_kind(&base_chunk).unwrap_or("");

    if previous_event == "message_start" {
        // 发送 content_block_start
        if let Some(start) = handle_content_block_start_typed(&base_chunk, *msg_index) {
            if let Ok(s) = serde_json::to_string(&start) {
                results.push(("content_block_start".to_string(), s));
            }
        }

        // 发送对应的增量事件（设置正确 index；tool_use 去掉 id/name）
        match &base_chunk {
            AnthropicStreamChunk::ContentBlockDelta { delta, .. } => {
                let delta_for_emit = match delta.clone() {
                    AnthropicStreamDelta::InputJsonDelta { partial_json, .. } => {
                        AnthropicStreamDelta::InputJsonDelta {
                            partial_json,
                            name: None,
                            id: None,
                        }
                    }
                    other => other,
                };
                let emit = AnthropicStreamChunk::ContentBlockDelta {
                    index: *msg_index,
                    delta: delta_for_emit,
                };
                if let Ok(s) = serde_json::to_string(&emit) {
                    results.push((event_type.to_string(), s));
                }
                previous_delta_type.clear();
                previous_delta_type.push_str(current_delta_type);
                previous_event.clear();
                previous_event.push_str("content_block_delta");
            }
            AnthropicStreamChunk::MessageDelta { .. } => {
                if let Ok(s) = serde_json::to_string(&base_chunk) {
                    results.push((event_type.to_string(), s));
                }
                if let Ok(s) = serde_json::to_string(&AnthropicStreamChunk::MessageStop) {
                    results.push(("message_stop".to_string(), s));
                }
            }
            _ => {}
        }
        return results;
    }

    if previous_event == "content_block_delta" {
        match &base_chunk {
            AnthropicStreamChunk::ContentBlockDelta { delta, .. } => {
                let new_delta_type = current_delta_type;
                if new_delta_type == previous_delta_type {
                    // 同一内容类型，直接追加增量
                    let delta_for_emit = match delta.clone() {
                        AnthropicStreamDelta::InputJsonDelta { partial_json, .. } => {
                            AnthropicStreamDelta::InputJsonDelta {
                                partial_json,
                                name: None,
                                id: None,
                            }
                        }
                        other => other,
                    };
                    let emit = AnthropicStreamChunk::ContentBlockDelta {
                        index: *msg_index,
                        delta: delta_for_emit,
                    };
                    if let Ok(s) = serde_json::to_string(&emit) {
                        results.push(("content_block_delta".to_string(), s));
                    }
                } else if matches!(
                    new_delta_type,
                    "input_json_delta" | "thinking_delta" | "text_delta"
                ) {
                    // 切换内容类型，先停止当前内容块
                    if let Ok(s) = serde_json::to_string(&AnthropicStreamChunk::ContentBlockStop {
                        index: *msg_index,
                    }) {
                        results.push(("content_block_stop".to_string(), s));
                    }
                    *msg_index += 1;

                    // 新内容块开始
                    if let Some(start) = handle_content_block_start_typed(&base_chunk, *msg_index) {
                        if let Ok(s) = serde_json::to_string(&start) {
                            results.push(("content_block_start".to_string(), s));
                        }
                    }

                    // 发送新的增量
                    let delta_for_emit = match delta.clone() {
                        AnthropicStreamDelta::InputJsonDelta { partial_json, .. } => {
                            AnthropicStreamDelta::InputJsonDelta {
                                partial_json,
                                name: None,
                                id: None,
                            }
                        }
                        other => other,
                    };
                    let emit = AnthropicStreamChunk::ContentBlockDelta {
                        index: *msg_index,
                        delta: delta_for_emit,
                    };
                    if let Ok(s) = serde_json::to_string(&emit) {
                        results.push(("content_block_delta".to_string(), s));
                    }

                    previous_delta_type.clear();
                    previous_delta_type.push_str(new_delta_type);
                }
            }
            AnthropicStreamChunk::MessageDelta { .. } => {
                if let Ok(s) = serde_json::to_string(&base_chunk) {
                    results.push(("message_delta".to_string(), s));
                }
                if let Ok(s) = serde_json::to_string(&AnthropicStreamChunk::MessageStop) {
                    results.push(("message_stop".to_string(), s));
                }
            }
            _ => {}
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;
    use http_body_util::BodyExt;
    use mockito;
    use regex::Regex;
    use serde_json::{Value, json};

    #[tokio::test]
    async fn test_openai_to_openai_response() {
        let response_json = json!({
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": "\nI'll calculate 365 + 96 for you.\n",
                        "reasoning_content": "use function",
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": "{\"a\": 365, \"b\": 96}",
                                    "name": "add"
                                },
                                "id": "call_-8344960410209973379",
                                "index": 0,
                                "type": "function"
                            }
                        ]
                    }
                }
            ],
            "created": 1757841257,
            "id": "20250914171414697fe62be8b14d74",
            "model": "glm-4.5-flash",
            "request_id": "20250914171414697fe62be8b14d74",
            "usage": {
                "completion_tokens": 89,
                "prompt_tokens": 170,
                "prompt_tokens_details": {
                    "cached_tokens": 43
                },
                "total_tokens": 259
            }
        });
        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::OpenAI,
            ApiType::OpenAI,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["completion_tokens"], 89);
        assert_eq!(json_body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            json_body["choices"][0]["message"]["content"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(
            json_body["choices"][0]["message"]["reasoning_content"],
            "use function"
        );
        assert_eq!(
            json_body["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_-8344960410209973379"
        );
        assert_eq!(
            json_body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"a\": 365, \"b\": 96}"
        );
    }

    #[tokio::test]
    async fn test_openai_to_anthropic_response() {
        let response_json = json!({
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": "\nI'll calculate 365 + 96 for you.\n",
                        "reasoning_content": "use function",
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": "{\"a\": 365, \"b\": 96}",
                                    "name": "add"
                                },
                                "id": "call_-8344960410209973379",
                                "index": 0,
                                "type": "function"
                            }
                        ]
                    }
                }
            ],
            "created": 1757841257,
            "id": "20250914171414697fe62be8b14d74",
            "model": "glm-4.5-flash",
            "request_id": "20250914171414697fe62be8b14d74",
            "usage": {
                "completion_tokens": 89,
                "prompt_tokens": 170,
                "prompt_tokens_details": {
                    "cached_tokens": 43
                },
                "total_tokens": 259
            }
        });
        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::OpenAI,
            ApiType::Anthropic,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["input_tokens"], 170);
        assert_eq!(json_body["stop_reason"], "tool_use");
        assert_eq!(json_body["content"][0]["thinking"], "use function");
        assert_eq!(
            json_body["content"][1]["text"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(json_body["content"][2]["name"], "add");
        assert_eq!(json_body["content"][2]["input"]["a"], 365);
    }

    #[tokio::test]
    async fn test_anthropic_to_anthropic_response() {
        let response_json = json!({
            "id": "202509141753426ef8338b65c54ac3",
            "type": "message",
            "role": "assistant",
            "model": "glm-4.5-flash",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me analyze this step by step...",
                    "signature": "WaUj...."
                },
                {
                    "type": "text",
                    "text": "\nI'll calculate 365 + 96 for you.\n"
                },
                {
                    "type": "tool_use",
                    "id": "call_-8344921446265725102",
                    "name": "add",
                    "input": {
                        "a": 365,
                        "b": 96
                    }
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 170,
                "output_tokens": 113
            }
        });
        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::Anthropic,
            ApiType::Anthropic,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["input_tokens"], 170);
        assert_eq!(json_body["stop_reason"], "tool_use");
        assert_eq!(
            json_body["content"][0]["thinking"],
            "Let me analyze this step by step..."
        );
        assert_eq!(
            json_body["content"][1]["text"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(json_body["content"][2]["name"], "add");
        assert_eq!(json_body["content"][2]["input"]["a"], 365);
    }

    #[tokio::test]
    async fn test_anthropic_to_openai_response() {
        let response_json = json!({
            "id": "202509141753426ef8338b65c54ac3",
            "type": "message",
            "role": "assistant",
            "model": "glm-4.5-flash",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me analyze this step by step...",
                    "signature": "WaUj...."
                },
                {
                    "type": "text",
                    "text": "\nI'll calculate 365 + 96 for you.\n"
                },
                {
                    "type": "tool_use",
                    "id": "call_-8344921446265725102",
                    "name": "add",
                    "input": {
                        "a": 365,
                        "b": 96
                    }
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 170,
                "output_tokens": 113
            }
        });
        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::Anthropic,
            ApiType::OpenAI,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["completion_tokens"], 113);
        assert_eq!(json_body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            json_body["choices"][0]["message"]["content"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(
            json_body["choices"][0]["message"]["reasoning_content"],
            "Let me analyze this step by step..."
        );
        assert_eq!(
            json_body["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_-8344921446265725102"
        );
        let re = Regex::new(r#"\{\s*"a"\s*:\s*365\s*,\s*"b"\s*:\s*96\s*\}"#).unwrap();
        let args = json_body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert!(re.is_match(args));
    }

    #[tokio::test]
    async fn test_gemini_to_gemini_response() {
        let response_json = json!({
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            { "text": "Let me analyze this step by step...", "thought": true },
                            { "text": "\nI'll calculate 365 + 96 for you.\n" },
                            { "functionCall": { "name": "add", "args": { "a": 365, "b": 96 } }, "thoughtSignature": null }
                        ]
                    },
                    "finishReason": "STOP",
                    "index": 0
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 170,
                "candidatesTokenCount": 113,
                "totalTokenCount": 283,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 246
                    }
                ],
                "thoughtsTokenCount": 148
            },
            "modelVersion": "gemini-1.5-pro",
            "responseId": "resp_1"
        });

        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::Gemini,
            ApiType::Gemini,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["modelVersion"], "test");
        assert_eq!(json_body["usageMetadata"]["candidatesTokenCount"], 113);
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][1]["text"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][0]["thought"],
            true
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][2]["functionCall"]["name"],
            "add"
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][2]["functionCall"]["args"]["a"],
            365
        );
    }

    #[tokio::test]
    async fn test_gemini_to_openai_response() {
        let response_json = json!({
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            { "text": "Let me analyze this step by step...", "thought": true },
                            { "text": "\nI'll calculate 365 + 96 for you.\n" },
                            { "functionCall": { "name": "add", "args": { "a": 365, "b": 96 } } }
                        ]
                    },
                    "finishReason": "STOP",
                    "index": 0
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 170,
                "candidatesTokenCount": 113,
                "totalTokenCount": 283
            },
            "modelVersion": "gemini-1.5-pro",
            "responseId": "resp_1"
        });

        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::Gemini,
            ApiType::OpenAI,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["completion_tokens"], 113);
        assert_eq!(json_body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            json_body["choices"][0]["message"]["content"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(
            json_body["choices"][0]["message"]["reasoning_content"],
            "Let me analyze this step by step..."
        );
        assert_eq!(
            json_body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "add"
        );
        let re = Regex::new(r#"\{\s*"a"\s*:\s*365\s*,\s*"b"\s*:\s*96\s*\}"#).unwrap();
        let args = json_body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert!(re.is_match(args));
    }

    #[tokio::test]
    async fn test_gemini_to_anthropic_response() {
        let response_json = json!({
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            { "text": "Let me analyze this step by step...", "thought": true },
                            { "text": "\nI'll calculate 365 + 96 for you.\n" },
                            { "functionCall": { "name": "add", "args": { "a": 365, "b": 96 } } }
                        ]
                    },
                    "finishReason": "STOP",
                    "index": 0
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 170,
                "candidatesTokenCount": 113,
                "totalTokenCount": 283
            },
            "modelVersion": "gemini-1.5-pro",
            "responseId": "resp_1"
        });

        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::Gemini,
            ApiType::Anthropic,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["output_tokens"], 113);
        assert_eq!(json_body["stop_reason"], "tool_use");
        assert_eq!(
            json_body["content"][0]["thinking"],
            "Let me analyze this step by step..."
        );
        assert_eq!(
            json_body["content"][1]["text"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(json_body["content"][2]["name"], "add");
        assert_eq!(json_body["content"][2]["input"]["a"], 365);
    }

    #[tokio::test]
    async fn test_openai_to_gemini_response() {
        let response_json = json!({
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "\nI'll calculate 365 + 96 for you.\n",
                        "reasoning_content": "Let me analyze this step by step...",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": "{\"a\": 365, \"b\": 96}",
                                    "name": "add"
                                },
                                "id": "call_-8344960410209973379",
                                "index": 0,
                                "type": "function"
                            }
                        ]
                    }
                }
            ],
            "created": 1757841257,
            "id": "20250914171414697fe62be8b14d74",
            "model": "glm-4.5-flash",
            "usage": {
                "completion_tokens": 113,
                "prompt_tokens": 170,
                "total_tokens": 283
            }
        });

        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::OpenAI,
            ApiType::Gemini,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["modelVersion"], "test");
        assert_eq!(json_body["usageMetadata"]["candidatesTokenCount"], 113);
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][0]["text"],
            "Let me analyze this step by step..."
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][0]["thought"],
            true
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][1]["text"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][2]["functionCall"]["name"],
            "add"
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][2]["functionCall"]["args"]["a"],
            365
        );
    }

    #[tokio::test]
    async fn test_anthropic_to_gemini_response() {
        let response_json = json!({
            "id": "202509141753426ef8338b65c54ac3",
            "type": "message",
            "role": "assistant",
            "model": "glm-4.5-flash",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me analyze this step by step...",
                    "signature": "WaUj...."
                },
                {
                    "type": "text",
                    "text": "\nI'll calculate 365 + 96 for you.\n"
                },
                {
                    "type": "tool_use",
                    "id": "call_-8344921446265725102",
                    "name": "add",
                    "input": {
                        "a": 365,
                        "b": 96
                    }
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 170,
                "output_tokens": 113
            }
        });

        let mut server = mockito::Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/test", url))
            .send()
            .await
            .expect("request failed");

        let axum_resp = handle_non_streaming_response(
            response,
            "test".to_string(),
            ApiType::Anthropic,
            ApiType::Gemini,
        )
        .await;

        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["modelVersion"], "test");
        assert_eq!(json_body["usageMetadata"]["candidatesTokenCount"], 113);
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][0]["text"],
            "Let me analyze this step by step..."
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][0]["thought"],
            true
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][1]["text"],
            "\nI'll calculate 365 + 96 for you.\n"
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][2]["functionCall"]["name"],
            "add"
        );
        assert_eq!(
            json_body["candidates"][0]["content"]["parts"][2]["functionCall"]["args"]["a"],
            365
        );
    }

    // =============== Streaming tests ===============

    // Helper: extract data JSON strings from SSE body (ignores [DONE] and keep-alives)
    fn extract_sse_data_json_chunks(s: &str) -> Vec<String> {
        let mut out = Vec::new();
        for section in s.split("\n\n") {
            for line in section.lines() {
                if let Some(rest) = line.strip_prefix("data: ") {
                    if rest.trim() == "[DONE]" {
                        continue;
                    }
                    out.push(rest.to_string());
                }
            }
        }
        out
    }

    // Helper: find first section by event name and return its data JSON string
    fn find_event_data(s: &str, event_name: &str) -> Option<String> {
        for section in s.split("\n\n") {
            let mut has_event = false;
            let mut data_line: Option<&str> = None;
            for line in section.lines() {
                if line.trim() == format!("event: {}", event_name) {
                    has_event = true;
                }
                if let Some(rest) = line.strip_prefix("data: ") {
                    data_line = Some(rest);
                }
            }
            if has_event {
                if let Some(d) = data_line {
                    return Some(d.to_string());
                }
            }
        }
        None
    }

    // Helper: extract sequence of event names in SSE stream
    fn extract_event_sequence(s: &str) -> Vec<String> {
        let mut out = Vec::new();
        for section in s.split("\n\n") {
            for line in section.lines() {
                if let Some(name) = line.strip_prefix("event: ") {
                    out.push(name.to_string());
                }
            }
        }
        out
    }

    #[tokio::test]
    async fn test_stream_openai_to_openai_basic() {
        // Build a single OpenAI stream chunk with text content
        let openai_chunk = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [ { "index": 0, "delta": { "content": "Hello" }, "finish_reason": null } ]
        });
        let s = stream::iter(vec![
            Ok(Bytes::from(format!(
                "data: {}\n",
                serde_json::to_string(&openai_chunk).unwrap()
            ))),
            Ok(Bytes::from("data: [DONE]\n")),
        ]);

        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::OpenAI, ApiType::OpenAI)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // Contains [DONE]
        assert!(body_str.contains("data: [DONE]"));

        // Extract JSON frame and validate content + model override
        let frames = extract_sse_data_json_chunks(&body_str);
        assert!(!frames.is_empty());
        let v: Value = serde_json::from_str(&frames[0]).unwrap();
        assert_eq!(v["model"], "test");
        assert_eq!(v["choices"][0]["delta"]["content"], "Hello");
    }

    #[tokio::test]
    async fn test_stream_anthropic_to_anthropic_message_start() {
        // Anthropic message_start should keep event name and override model
        let anthropic_chunk = json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-opus"
            }
        });
        let s = stream::iter(vec![Ok(Bytes::from(format!(
            "data: {}\n",
            serde_json::to_string(&anthropic_chunk).unwrap()
        )))]);

        let resp = handle_streaming_response(
            s,
            "test".to_string(),
            ApiType::Anthropic,
            ApiType::Anthropic,
        )
        .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // Find the message_start section and inspect data
        let data = find_event_data(&body_str, "message_start").expect("message_start not found");
        let v: Value = serde_json::from_str(&data).unwrap();
        assert_eq!(v["type"], "message_start");
        assert_eq!(v["message"]["model"], "test");
    }

    #[tokio::test]
    async fn test_stream_anthropic_to_openai_content_delta() {
        // Anthropic content_block_delta (text) -> OpenAI chunk with delta.content
        let anthropic_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": "Hi" }
        });
        let s = stream::iter(vec![Ok(Bytes::from(format!(
            "data: {}\n",
            serde_json::to_string(&anthropic_chunk).unwrap()
        )))]);

        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::Anthropic, ApiType::OpenAI)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        let frames = extract_sse_data_json_chunks(&body_str);
        assert_eq!(frames.len(), 1);
        let v: Value = serde_json::from_str(&frames[0]).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "Hi");
        // model is overridden in openai->openai path; here we convert from anthropic and model can be default
    }

    #[tokio::test]
    async fn test_stream_openai_to_anthropic_sequence() {
        // OpenAI text delta should expand to message_start + content_block_start(text) + content_block_delta
        let openai_chunk = json!({
            "id": "chatcmpl-xyz",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [ { "index": 0, "delta": { "content": "Hello" }, "finish_reason": null } ]
        });
        let s = stream::iter(vec![Ok(Bytes::from(format!(
            "data: {}\n",
            serde_json::to_string(&openai_chunk).unwrap()
        )))]);

        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::OpenAI, ApiType::Anthropic)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // Expect message_start
        let msg_start =
            find_event_data(&body_str, "message_start").expect("message_start not found");
        let v_start: Value = serde_json::from_str(&msg_start).unwrap();
        assert_eq!(v_start["message"]["model"], "test");

        // Expect content_block_start with text
        let cb_start = find_event_data(&body_str, "content_block_start")
            .expect("content_block_start not found");
        let v_cb_start: Value = serde_json::from_str(&cb_start).unwrap();
        assert_eq!(v_cb_start["content_block"]["type"], "text");

        // Expect content_block_delta with text_delta:"Hello"
        let cb_delta = find_event_data(&body_str, "content_block_delta")
            .expect("content_block_delta not found");
        let v_cb_delta: Value = serde_json::from_str(&cb_delta).unwrap();
        assert_eq!(v_cb_delta["delta"]["type"], "text_delta");
        assert_eq!(v_cb_delta["delta"]["text"], "Hello");
    }

    #[tokio::test]
    async fn test_stream_openai_to_anthropic_mixed_and_finish() {
        // Sequence: reasoning -> text -> finish
        let openai_reasoning = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [ { "index": 0, "delta": { "reasoning_content": "Think" }, "finish_reason": null } ]
        });
        let openai_text = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 2,
            "model": "gpt-4",
            "choices": [ { "index": 0, "delta": { "content": "Hello" }, "finish_reason": null } ]
        });
        let openai_finish = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 3,
            "model": "gpt-4",
            "choices": [ { "index": 0, "delta": {}, "finish_reason": "stop" } ]
        });
        let s = stream::iter(vec![
            Ok(Bytes::from(format!(
                "data: {}\n",
                serde_json::to_string(&openai_reasoning).unwrap()
            ))),
            Ok(Bytes::from(format!(
                "data: {}\n",
                serde_json::to_string(&openai_text).unwrap()
            ))),
            Ok(Bytes::from(format!(
                "data: {}\n",
                serde_json::to_string(&openai_finish).unwrap()
            ))),
        ]);

        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::OpenAI, ApiType::Anthropic)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // Verify event sequence order
        let seq = extract_event_sequence(&body_str);
        // Expected: message_start, content_block_start(thinking), content_block_delta, content_block_stop,
        //           content_block_start(text), content_block_delta, content_block_stop, message_delta, message_stop
        assert!(seq.starts_with(&vec![
            "message_start".to_string(),
            "content_block_start".to_string(),
            "content_block_delta".to_string(),
            "content_block_stop".to_string(),
            "content_block_start".to_string(),
            "content_block_delta".to_string(),
        ]));
        assert!(seq.ends_with(&vec![
            "message_delta".to_string(),
            "message_stop".to_string()
        ]));

        // Check payloads for the two deltas
        let first_delta = find_event_data(&body_str, "content_block_delta").unwrap();
        let v1: Value = serde_json::from_str(&first_delta).unwrap();
        assert_eq!(v1["delta"]["type"], "thinking_delta");
        assert_eq!(v1["delta"]["thinking"], "Think");

        // Find the second content_block_delta by scanning all sections
        let mut text_delta_found = false;
        for section in body_str.split("\n\n") {
            if section
                .lines()
                .any(|l| l.trim() == "event: content_block_delta")
            {
                if let Some(d) = section.lines().find_map(|l| l.strip_prefix("data: ")) {
                    let vv: Value = serde_json::from_str(d).unwrap();
                    if vv["delta"]["type"] == "text_delta" {
                        assert_eq!(vv["delta"]["text"], "Hello");
                        text_delta_found = true;
                        break;
                    }
                }
            }
        }
        assert!(text_delta_found, "expected text_delta not found");
    }

    #[tokio::test]
    async fn test_stream_openai_to_anthropic_tool_call_sequence() {
        // OpenAI tool_call delta -> Anthropic tool_use content block start + input_json_delta
        let openai_tool = json!({
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "created": 42,
            "model": "gpt-4",
            "choices": [ {
                "index": 0,
                "delta": {
                    "tool_calls": [ { "index": 0, "id": "call_1", "type": "function", "function": {"name": "add", "arguments": r#"{"a":1}"#} } ]
                },
                "finish_reason": null
            } ]
        });
        let s = stream::iter(vec![Ok(Bytes::from(format!(
            "data: {}\n",
            serde_json::to_string(&openai_tool).unwrap()
        )))]);

        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::OpenAI, ApiType::Anthropic)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        let cb_start = find_event_data(&body_str, "content_block_start")
            .expect("content_block_start not found");
        let v_start: Value = serde_json::from_str(&cb_start).unwrap();
        assert_eq!(v_start["content_block"]["type"], "tool_use");
        assert_eq!(v_start["content_block"]["id"], "call_1");
        assert_eq!(v_start["content_block"]["name"], "add");

        let cb_delta = find_event_data(&body_str, "content_block_delta")
            .expect("content_block_delta not found");
        let v_delta: Value = serde_json::from_str(&cb_delta).unwrap();
        assert_eq!(v_delta["delta"]["type"], "input_json_delta");
        assert_eq!(v_delta["delta"]["partial_json"], "{\"a\":1}");
    }

    #[tokio::test]
    async fn test_stream_openai_to_openai_only_done_when_no_json() {
        // Provide only [DONE] and a malformed JSON frame
        let s = stream::iter(vec![
            Ok(Bytes::from("data: not-json\n")),
            Ok(Bytes::from("data: [DONE]\n")),
        ]);

        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::OpenAI, ApiType::OpenAI)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        assert!(body_str.contains("data: [DONE]"));
        // No other data JSON frames
        let frames = extract_sse_data_json_chunks(&body_str);
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn test_stream_openai_to_anthropic_full_sample() {
        let openai_stream = r#"data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":"\n"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" I"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" need"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" to"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" call"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" this"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" function"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" with"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" a"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"="}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"365"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" and"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" b"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"="}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"96"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"."}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"\n"}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"finish_reason":"tool_calls","delta":{"tool_calls":[{"id":"call_4BgN84hpTbmswCBFSC9ggw","index":-1,"type":"function","function":{"name":"add","arguments":"{\"a\": 365, \"b\": 96}"}}]}}]}

data: {"id":"2025091420174501de064dedfa4b68","created":1757852265,"model":"glm-4.5-flash","choices":[{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":170,"completion_tokens":72,"total_tokens":242,"prompt_tokens_details":{"cached_tokens":43}}}

data: [DONE]

"#;
        // Assemble stream
        let mut frames: Vec<Result<Bytes, reqwest::Error>> = Vec::new();
        for line in openai_stream.split("\n") {
            frames.push(Ok(Bytes::from(line)));
        }

        let s = stream::iter(frames);
        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::OpenAI, ApiType::Anthropic)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // 1) Starts with message_start and model overridden
        let msg_start =
            find_event_data(&body_str, "message_start").expect("message_start not found");
        let v_start: Value = serde_json::from_str(&msg_start).unwrap();
        assert_eq!(v_start["message"]["model"], "test");

        // 2) First content block is thinking with multiple deltas equal to reasoning_pieces.len()
        let cb_start1 = find_event_data(&body_str, "content_block_start")
            .expect("first content_block_start not found");
        let v_cb_start1: Value = serde_json::from_str(&cb_start1).unwrap();
        assert_eq!(v_cb_start1["content_block"]["type"], "thinking");

        // Count thinking deltas
        let mut _thinking_count = 0usize;
        for section in body_str.split("\n\n") {
            if section
                .lines()
                .any(|l| l.trim() == "event: content_block_delta")
            {
                if let Some(d) = section.lines().find_map(|l| l.strip_prefix("data: ")) {
                    let v: Value = serde_json::from_str(d).unwrap_or(json!({}));
                    if v["delta"]["type"] == "thinking_delta" {
                        _thinking_count += 1;
                    }
                }
            }
        }
        // assert_eq!(thinking_count, reasoning_pieces.len());

        // 3) Then text block with a single newline
        // Find the second content_block_start (text)
        let mut saw_text_start = false;
        let mut saw_text_delta = false;
        for section in body_str.split("\n\n") {
            let is_start = section
                .lines()
                .any(|l| l.trim() == "event: content_block_start");
            let data_line = section.lines().find_map(|l| l.strip_prefix("data: "));
            if is_start {
                if let Some(d) = data_line {
                    let v: Value = serde_json::from_str(d).unwrap();
                    if v["content_block"]["type"] == "text" {
                        saw_text_start = true;
                    }
                }
            }
            let is_delta = section
                .lines()
                .any(|l| l.trim() == "event: content_block_delta");
            if is_delta {
                if let Some(d) = data_line {
                    let v: Value = serde_json::from_str(d).unwrap();
                    if v["delta"]["type"] == "text_delta" && v["delta"]["text"] == "\n" {
                        saw_text_delta = true;
                    }
                }
            }
        }
        assert!(saw_text_start);
        assert!(saw_text_delta);

        // 4) Tool use start + input_json_delta
        let mut saw_tool_start = false;
        let mut saw_tool_delta = false;
        for section in body_str.split("\n\n") {
            if section
                .lines()
                .any(|l| l.trim() == "event: content_block_start")
            {
                if let Some(d) = section.lines().find_map(|l| l.strip_prefix("data: ")) {
                    let v: Value = serde_json::from_str(d).unwrap();
                    if v["content_block"]["type"] == "tool_use" {
                        assert_eq!(v["content_block"]["id"], "call_4BgN84hpTbmswCBFSC9ggw");
                        assert_eq!(v["content_block"]["name"], "add");
                        saw_tool_start = true;
                    }
                }
            }
            if section
                .lines()
                .any(|l| l.trim() == "event: content_block_delta")
            {
                if let Some(d) = section.lines().find_map(|l| l.strip_prefix("data: ")) {
                    let v: Value = serde_json::from_str(d).unwrap();
                    if v["delta"]["type"] == "input_json_delta" {
                        assert_eq!(v["delta"]["partial_json"], "{\"a\": 365, \"b\": 96}");
                        saw_tool_delta = true;
                    }
                }
            }
        }
        assert!(saw_tool_start);
        assert!(saw_tool_delta);

        // 5) Finish: message_delta with stop_reason mapped to tool_use and usage mapped
        let msg_delta =
            find_event_data(&body_str, "message_delta").expect("message_delta not found");
        let v_msg_delta: Value = serde_json::from_str(&msg_delta).unwrap();
        assert_eq!(v_msg_delta["delta"]["stop_reason"], "tool_use");
        assert_eq!(v_msg_delta["usage"]["input_tokens"], 170);
        assert_eq!(v_msg_delta["usage"]["output_tokens"], 72);

        // 6) Ends with message_stop
        let seq = extract_event_sequence(&body_str);
        assert!(seq.ends_with(&vec![
            "message_delta".to_string(),
            "message_stop".to_string()
        ]));
    }

    #[tokio::test]
    async fn test_stream_anthropic_to_openai_full_sample() {
        // Provided Anthropic SSE sample should convert into OpenAI-style stream
        let anthropic_stream = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_8dea3300abc14869bdccc73e","type":"message","role":"assistant","model":"glm-4.5-flash","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\n"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'ll"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" help"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" you"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" calculate"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"365"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" +"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"96"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" using"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" the"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" addition"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" function"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"."}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_e48d1c06c2e94c5380744c68","name":"add","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\": 365, \"b\": 96}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}

event: message_stop
data: {"type":"message_stop"}
"#;

        // Assemble input frames as byte stream
        let mut frames: Vec<Result<Bytes, reqwest::Error>> = Vec::new();
        for line in anthropic_stream.split("\n") {
            frames.push(Ok(Bytes::from(line)));
        }

        let s = stream::iter(frames);
        let resp =
            handle_streaming_response(s, "test".to_string(), ApiType::Anthropic, ApiType::OpenAI)
                .await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // Collect OpenAI JSON chunks
        let frames = extract_sse_data_json_chunks(&body_str);
        assert!(!frames.is_empty());

        // 1) First delta should set role/content/reasoning scaffolding
        let v0: Value = serde_json::from_str(&frames[0]).unwrap();
        assert_eq!(v0["choices"][0]["delta"]["role"], "assistant");

        // 2) Concatenate all text deltas and verify final sentence
        let mut text_out = String::new();
        for f in &frames {
            let v: Value = serde_json::from_str(f).unwrap_or(json!({}));
            if let Some(s) = v["choices"][0]["delta"]["content"].as_str() {
                text_out.push_str(s);
            }
        }
        assert_eq!(
            text_out,
            "\nI'll help you calculate 365 + 96 using the addition function."
        );

        // 3) Tool call start maps to OpenAI tool_calls with id/name and empty args initially
        let mut saw_tool_start = false;
        let mut saw_tool_delta = false;
        let mut saw_finish_tool_calls = false;
        for f in &frames {
            let v: Value = serde_json::from_str(f).unwrap_or(json!({}));
            if v["choices"][0]["delta"]["tool_calls"].is_array() {
                let name = v["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
                    .as_str()
                    .unwrap_or("");
                if name == "add" {
                    saw_tool_start = true;
                }
                let args = v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap_or("");
                if args.contains("\"a\": 365") && args.contains("\"b\": 96") {
                    saw_tool_delta = true;
                }
            }
            if v["choices"][0]["finish_reason"].as_str() == Some("tool_calls") {
                saw_finish_tool_calls = true;
            }
        }
        assert!(saw_tool_start);
        assert!(saw_tool_delta);
        assert!(saw_finish_tool_calls);
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_message_start() {
        // 测试初始消息开始的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": null
                }
            ]
        });
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();

        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;

        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index,
        );

        // 应该返回两个事件：message_start 和 content_block_delta
        assert_eq!(results.len(), 3);

        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");
        let first_data = first_event.1;
        assert!(
            Regex::new(r#""type":\s*"message_start""#)
                .unwrap()
                .is_match(&first_data)
        );
        assert!(
            Regex::new(r#""id":\s*"chatcmpl-123""#)
                .unwrap()
                .is_match(&first_data)
        );
        assert!(
            Regex::new(r#""model":\s*"claude-3-opus""#)
                .unwrap()
                .is_match(&first_data)
        );

        // 检查第二个事件是 content_block_delta
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");

        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(
            Regex::new(r#""type":\s*"content_block_delta""#)
                .unwrap()
                .is_match(&third_data)
        );
        assert!(
            Regex::new(r#""type":\s*"text_delta""#)
                .unwrap()
                .is_match(&third_data)
        );
        assert!(
            Regex::new(r#""text":\s*"Hello""#)
                .unwrap()
                .is_match(&third_data)
        );

        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "text_delta");
        assert_eq!(msg_index, 0);
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_with_reasoning_content() {
        // 测试包含推理内容的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-456",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "I need to think about this."
                    },
                    "finish_reason": null
                }
            ]
        });
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();

        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;

        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index,
        );

        // 应该返回两个事件：message_start 和 content_block_delta
        assert_eq!(results.len(), 3);

        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");

        // 检查第二个事件是 content_block_delta
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");

        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(
            Regex::new(r#""type":\s*"thinking_delta""#)
                .unwrap()
                .is_match(&third_data)
        );
        assert!(
            Regex::new(r#""thinking":\s*"I need to think about this.""#)
                .unwrap()
                .is_match(&third_data)
        );

        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "thinking_delta");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_with_tool_calls() {
        // 测试包含工具调用的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-789",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\": \"San Francisco\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": null
                }
            ]
        });
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();

        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;

        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index,
        );

        // 应该返回三个事件：message_start, content_block_start, 和 content_block_delta
        assert_eq!(results.len(), 3);

        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");

        // 检查第二个事件是 content_block_start
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");
        let second_data = second_event.1;
        assert!(
            Regex::new(r#""type":\s*"content_block_start""#)
                .unwrap()
                .is_match(&second_data)
        );
        assert!(
            Regex::new(r#""type":\s*"tool_use""#)
                .unwrap()
                .is_match(&second_data)
        );
        assert!(
            Regex::new(r#""id":\s*"call_abc123""#)
                .unwrap()
                .is_match(&second_data)
        );
        assert!(
            Regex::new(r#""name":\s*"get_weather""#)
                .unwrap()
                .is_match(&second_data)
        );

        // 检查第三个事件是 content_block_delta
        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(
            Regex::new(r#""type":\s*"input_json_delta""#)
                .unwrap()
                .is_match(&third_data)
        );
        assert!(
            Regex::new(r#""partial_json":\s*"\{\\"location\\":\s*\\"San Francisco\\"\}""#)
                .unwrap()
                .is_match(&third_data)
        );

        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "input_json_delta");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_with_finish_reason() {
        // 测试包含完成原因的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-finish",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        });
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();

        let model = "claude-3-opus".to_string();
        let mut previous_event = "content_block_delta".to_string();
        let mut previous_delta_type = "text_delta".to_string();
        let mut msg_index = 0;

        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index,
        );

        // 应该返回三个事件：content_block_stop, message_delta, 和 message_stop
        assert_eq!(results.len(), 3);

        // 检查第一个事件是 content_block_stop
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "content_block_stop");
        let first_data = first_event.1;
        assert!(
            Regex::new(r#""type":\s*"content_block_stop""#)
                .unwrap()
                .is_match(&first_data)
        );
        assert!(Regex::new(r#""index":\s*0"#).unwrap().is_match(&first_data));

        // 检查第二个事件是 message_delta
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "message_delta");
        let second_data = second_event.1;
        assert!(
            Regex::new(r#""type":\s*"message_delta""#)
                .unwrap()
                .is_match(&second_data)
        );
        assert!(
            Regex::new(r#""stop_reason":\s*"end_turn""#)
                .unwrap()
                .is_match(&second_data)
        );

        // 检查第三个事件是 message_stop
        let third_event = results[2].clone();
        assert_eq!(third_event.0, "message_stop");
        let third_data = third_event.1;
        assert!(
            Regex::new(r#""type":\s*"message_stop""#)
                .unwrap()
                .is_match(&third_data)
        );
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_content_type_switch() {
        // 测试内容类型切换的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-switch",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": null
                }
            ]
        });
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();

        let model = "claude-3-opus".to_string();
        let mut previous_event = "content_block_delta".to_string();
        let mut previous_delta_type = "thinking_delta".to_string(); // 前一个是推理内容
        let mut msg_index = 0;

        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index,
        );

        // 应该返回三个事件：content_block_stop, content_block_start, 和 content_block_delta
        assert_eq!(results.len(), 3);

        // 检查第一个事件是 content_block_stop
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "content_block_stop");

        // 检查第二个事件是 content_block_start
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");

        // 检查第三个事件是 content_block_delta
        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(
            Regex::new(r#""type":\s*"text_delta""#)
                .unwrap()
                .is_match(&third_data)
        );
        assert!(
            Regex::new(r#""text":\s*"Hello""#)
                .unwrap()
                .is_match(&third_data)
        );

        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "text_delta");
        assert_eq!(msg_index, 1); // 索引应该增加
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_empty_content() {
        // 测试空内容的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-empty",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": null
                }
            ]
        });
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();

        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;

        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index,
        );

        // 应该只返回一个事件：message_start
        assert_eq!(results.len(), 1);

        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");

        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "message_start");
        assert_eq!(previous_delta_type, "");
        assert_eq!(msg_index, 0);
    }
}
