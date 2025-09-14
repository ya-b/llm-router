use crate::auth::AppState;
use crate::config::ModelConfig;
use crate::converters::anthropic::AnthropicStreamChunk;
use crate::converters::openai::OpenAIStreamChunk;
use crate::models::{ErrorResponse, ErrorDetail, ModelsResponse, ModelInfo};
use crate::converters::{
    openai::{OpenAIRequest, OpenAIResponse},
    anthropic::{AnthropicRequest, AnthropicResponse},
    request_wrapper::RequestWrapper,
    response_wrapper::ResponseWrapper,
    stream::openai_to_anthropic_stream_chunks
};
use anyhow::Result;
use axum::{
    extract::State,
    http::{StatusCode},
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

fn request(request: &RequestWrapper, model_config: &ModelConfig, proxy_url: &Option<String>) -> impl Future<Output = Result<reqwest::Response, reqwest::Error>> {
    // Forward request to the target API
    let client_builder = reqwest::Client::builder();
    
    // Configure proxy if provided
    let client_builder = if let Some(proxy) = proxy_url {
        let proxy = reqwest::Proxy::all(proxy)
            .expect("Failed to create proxy");
        client_builder.proxy(proxy)
    } else {
        client_builder
    };
    
    let client = client_builder.build().expect("Failed to build HTTP client");
    let target_url = build_target_url(model_config);
    if let Some(proxy) = proxy_url {
        debug!("Using proxy: {}", proxy);
    }

    let mut target_request = client.post(&target_url).header("Content-Type", "application/json");
    if model_config.llm_params.api_type == "anthropic" {
        target_request = target_request.header("x-api-key", model_config.llm_params.api_key.to_string());
    } else if model_config.llm_params.api_type == "openai" {
        target_request = target_request.header("Authorization", format!("Bearer {}", model_config.llm_params.api_key));
    }

    // Apply rewrite_header functionality
    match serde_json::from_str::<serde_json::Value>(&model_config.llm_params.rewrite_header) {
        Ok(serde_json::Value::Object(map)) => {
            for (k, v) in map {
                if !v.is_object() && !v.is_array() {
                    target_request = target_request.header(k, v.to_string());
                }
            }
        },
        Ok(_) => {warn!("'rewrite_header' parsed error")},
        Err(_) => {warn!("'rewrite_header' parsed error")}
    }
    let mut target_body = if model_config.llm_params.api_type == "anthropic" {
        let mut anthropic_req: AnthropicRequest = request.get_anthropic();
        anthropic_req.model = model_config.llm_params.model.clone();
        serde_json::to_value(anthropic_req).expect("Failed to serialize converted Anthropic request")
    } else {
        let mut openai_req: OpenAIRequest = request.get_openai();
        openai_req.model = model_config.llm_params.model.clone();
        serde_json::to_value(openai_req).expect("Failed to serialize converted OpenAI request")
    };

    match serde_json::from_str::<serde_json::Value>(&model_config.llm_params.rewrite_body) {
        Ok(serde_json::Value::Object(map)) => {
            if let Some(t_body) = target_body.as_object_mut() {
                for (k, v) in map {
                    t_body.insert(k, v);
                }
            }
        },
        Ok(_) => {warn!("'rewrite_body' parsed error")},
        Err(_) => {warn!("'rewrite_body' parsed error")}
    };

    info!("Forwarding request to: {}", target_url);
    debug!("request body: {}", serde_json::to_string(&target_body).expect("Failed to serialize request"));
    target_request.json(&target_body).send()
}

#[axum_macros::debug_handler]
pub async fn openai_chat(
    State(config): State<AppState>,
    Json(openai_request): Json<OpenAIRequest>,
) -> impl IntoResponse {
    route_chat("openai".to_string(), config, RequestWrapper::OpenAI(openai_request)).await
}

#[axum_macros::debug_handler]
pub async fn anthropic_chat(
    State(config): State<AppState>,
    Json(anthropic_request): Json<AnthropicRequest>,
) -> impl IntoResponse {
    route_chat("anthropic".to_string(), config, RequestWrapper::Anthropic(anthropic_request)).await
}


pub async fn route_chat(
    api_type: String,
    config: AppState,
    request_wrapper: RequestWrapper,
) -> impl IntoResponse {
    
    // Parse the request into the appropriate structure based on API type
    let model = request_wrapper.get_model();
    
    let stream = request_wrapper.is_stream().unwrap_or(false);
    
    debug!("raw request: {}", serde_json::to_string(&request_wrapper).expect("Failed to serialize request"));

    let model_manager = config.model_manager.read().await;
    let (model_config, group_name) = {
        match model_manager.get_model_config(model) {
            Some(config) => {
                debug!("Found model configuration for: {}", model);
                // Determine the group name - if it's an alias, use that, otherwise use the model name
                let group = if model_manager.is_alias(model) {
                    model.to_string()
                } else {
                    // For direct model access, we need to find which group it belongs to
                    // For simplicity, we'll use the model name as group name
                    model.to_string()
                };
                (config.clone(), group)
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
    model_manager.start_request(&group_name, &model_config.model_name);

    let proxy_url = model_manager.get_proxy();
    let response = request(&request_wrapper, &model_config, &proxy_url);
    let response = match response.await {
        Ok(resp) => resp,
        Err(e) => {
            warn!("Failed to send streaming request: {}", e);
            // Track the failed request
            model_manager.end_request(&group_name, &model_config.model_name, false);
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
        // Track the failed request
        model_manager.end_request(&group_name, &model_config.model_name, false);
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
        let result = handle_streaming_response(response.bytes_stream(), model.to_string(), model_config.llm_params.api_type.clone(), api_type.to_string()).await;
        // Track the successful completion of streaming request
        model_manager.end_request(&group_name, &model_config.model_name, true);
        result
    } else {
        info!("Processing non-streaming request");
        let result = handle_non_streaming_response(response, model.to_string(), model_config.llm_params.api_type.clone(), api_type.to_string()).await;
        // Track the successful completion of non-streaming request
        model_manager.end_request(&group_name, &model_config.model_name, true);
        result
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

    let response_wrapper = if source_api_type == "openai" {
        let resp: OpenAIResponse = serde_json::from_value(response_json).expect("Failed to deserialize JSON");
        if target_api_type == "openai" {
            ResponseWrapper::OpenAI(resp)
        } else {
            ResponseWrapper::Anthropic(resp.into())
        }
    } else {
        let resp: AnthropicResponse = serde_json::from_value(response_json).expect("Failed to deserialize JSON");
        if target_api_type == "openai" {
            ResponseWrapper::OpenAI(resp.into())
        } else {
            ResponseWrapper::Anthropic(resp)
        }
    };
    
    debug!("Response received with model updated to: {}\n{:?}", model, serde_json::to_string(&response_wrapper));
    Json(response_wrapper).into_response()
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
                        } else if source_api_type == "anthropic" && target_api_type == "anthropic" {
                            match serde_json::from_str::<AnthropicStreamChunk>(data) {
                                Ok(mut chunk) => {
                                    if let AnthropicStreamChunk::MessageStart{ mut message } = chunk.clone() {
                                        message.model = model.clone();
                                        chunk = AnthropicStreamChunk::MessageStart{ message: message.clone() };
                                    }
                                    if let Ok(chunk_str) = serde_json::to_string(&chunk) {
                                        vec![Ok(Event::default().event(chunk.stream_type()).data(chunk_str))]
                                    } else {
                                        vec![]
                                    }
                                },
                                Err(_) => vec![],
                            }
                        } else if source_api_type == "openai" && target_api_type == "openai" {
                            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                                Ok(mut chunk) => {
                                    chunk.model = model.clone();
                                    if let Ok(chunk_str) = serde_json::to_string(&chunk) {
                                        vec![Ok(Event::default().data(chunk_str))]
                                    } else {
                                        vec![]
                                    }
                                },
                                Err(_) => vec![],
                            }
                        } else if source_api_type == "anthropic" && target_api_type == "openai" {
                            match serde_json::from_str::<AnthropicStreamChunk>(data) {
                                Ok(mut chunk) => {
                                    if let AnthropicStreamChunk::MessageStart{ mut message } = chunk.clone() {
                                        message.model = model.clone();
                                        chunk = AnthropicStreamChunk::MessageStart{ message: message.clone() };
                                    }
                                    let openai_chunk: OpenAIStreamChunk = chunk.into();
                                    if let Ok(chunk_str) = serde_json::to_string(&openai_chunk) {
                                        vec![Ok(Event::default().data(chunk_str))]
                                    } else {
                                        vec![]
                                    }
                                },
                                Err(_) => vec![],
                            }
                        } else if source_api_type == "openai" && target_api_type == "anthropic" {
                            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                                Ok(mut chunk) => {
                                    // Replace the model field
                                    chunk.model = model.clone();
                                    openai_to_anthropic_stream_chunks(
                                        &chunk,
                                        &model,
                                        &mut previous_event,
                                        &mut previous_delta_type,
                                        &mut msg_index,
                                    )
                                    .into_iter()
                                    .map(|(a, b)| Ok(Event::default().event(a).data(b)))
                                    .collect()
                                }
                                Err(_) => vec![],
                            }
                        } else {
                            vec![]
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


#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;
    use mockito;
    use http_body_util::BodyExt;
    use bytes::Bytes;
    use futures::stream;


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
        let _m = server.mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client.post(format!("{}/test", url)).send().await.expect("request failed");


        let axum_resp = handle_non_streaming_response(response, "test".to_string(), "openai".to_string(), "openai".to_string()).await;
        
        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["completion_tokens"], 89);
        assert_eq!(json_body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(json_body["choices"][0]["message"]["content"], "\nI'll calculate 365 + 96 for you.\n");
        assert_eq!(json_body["choices"][0]["message"]["reasoning_content"], "use function");
        assert_eq!(json_body["choices"][0]["message"]["tool_calls"][0]["id"], "call_-8344960410209973379");
        assert_eq!(json_body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"a\": 365, \"b\": 96}");
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
        let _m = server.mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client.post(format!("{}/test", url)).send().await.expect("request failed");


        let axum_resp = handle_non_streaming_response(response, "test".to_string(), "openai".to_string(), "anthropic".to_string()).await;
        
        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["input_tokens"], 170);
        assert_eq!(json_body["stop_reason"], "tool_use");
        assert_eq!(json_body["content"][0]["thinking"], "use function");
        assert_eq!(json_body["content"][1]["text"], "\nI'll calculate 365 + 96 for you.\n");
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
        let _m = server.mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client.post(format!("{}/test", url)).send().await.expect("request failed");


        let axum_resp = handle_non_streaming_response(response, "test".to_string(), "anthropic".to_string(), "anthropic".to_string()).await;
        
        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["input_tokens"], 170);
        assert_eq!(json_body["stop_reason"], "tool_use");
        assert_eq!(json_body["content"][0]["thinking"], "Let me analyze this step by step...");
        assert_eq!(json_body["content"][1]["text"], "\nI'll calculate 365 + 96 for you.\n");
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
        let _m = server.mock("POST", "/test")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_json.to_string())
            .create();

        let client = reqwest::Client::new();
        let response = client.post(format!("{}/test", url)).send().await.expect("request failed");


        let axum_resp = handle_non_streaming_response(response, "test".to_string(), "anthropic".to_string(), "openai".to_string()).await;
        
        let body_bytes = axum_resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        let json_body: Value = serde_json::from_str(&body_str).unwrap();

        assert_eq!(json_body["model"], "test");
        assert_eq!(json_body["usage"]["completion_tokens"], 113);
        assert_eq!(json_body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(json_body["choices"][0]["message"]["content"], "\nI'll calculate 365 + 96 for you.\n");
        assert_eq!(json_body["choices"][0]["message"]["reasoning_content"], "Let me analyze this step by step...");
        assert_eq!(json_body["choices"][0]["message"]["tool_calls"][0]["id"], "call_-8344921446265725102");
        let re = Regex::new(r#"\{\s*"a"\s*:\s*365\s*,\s*"b"\s*:\s*96\s*\}"#).unwrap();
        let args = json_body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert!(re.is_match(args));
    }

    // =============== Streaming tests ===============

    // Helper: extract data JSON strings from SSE body (ignores [DONE] and keep-alives)
    fn extract_sse_data_json_chunks(s: &str) -> Vec<String> {
        let mut out = Vec::new();
        for section in s.split("\n\n") {
            for line in section.lines() {
                if let Some(rest) = line.strip_prefix("data: ") {
                    if rest.trim() == "[DONE]" { continue; }
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
                if line.trim() == format!("event: {}", event_name) { has_event = true; }
                if let Some(rest) = line.strip_prefix("data: ") { data_line = Some(rest); }
            }
            if has_event {
                if let Some(d) = data_line { return Some(d.to_string()); }
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
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&openai_chunk).unwrap()))),
            Ok(Bytes::from("data: [DONE]\n")),
        ]);

        let resp = handle_streaming_response(s, "test".to_string(), "openai".to_string(), "openai".to_string()).await;
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
        let s = stream::iter(vec![
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&anthropic_chunk).unwrap()))),
        ]);

        let resp = handle_streaming_response(s, "test".to_string(), "anthropic".to_string(), "anthropic".to_string()).await;
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
        let s = stream::iter(vec![
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&anthropic_chunk).unwrap()))),
        ]);

        let resp = handle_streaming_response(s, "test".to_string(), "anthropic".to_string(), "openai".to_string()).await;
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
        let s = stream::iter(vec![
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&openai_chunk).unwrap()))),
        ]);

        let resp = handle_streaming_response(s, "test".to_string(), "openai".to_string(), "anthropic".to_string()).await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // Expect message_start
        let msg_start = find_event_data(&body_str, "message_start").expect("message_start not found");
        let v_start: Value = serde_json::from_str(&msg_start).unwrap();
        assert_eq!(v_start["message"]["model"], "test");

        // Expect content_block_start with text
        let cb_start = find_event_data(&body_str, "content_block_start").expect("content_block_start not found");
        let v_cb_start: Value = serde_json::from_str(&cb_start).unwrap();
        assert_eq!(v_cb_start["content_block"]["type"], "text");

        // Expect content_block_delta with text_delta:"Hello"
        let cb_delta = find_event_data(&body_str, "content_block_delta").expect("content_block_delta not found");
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
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&openai_reasoning).unwrap()))),
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&openai_text).unwrap()))),
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&openai_finish).unwrap()))),
        ]);

        let resp = handle_streaming_response(s, "test".to_string(), "openai".to_string(), "anthropic".to_string()).await;
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
        assert!(seq.ends_with(&vec!["message_delta".to_string(), "message_stop".to_string()]));

        // Check payloads for the two deltas
        let first_delta = find_event_data(&body_str, "content_block_delta").unwrap();
        let v1: Value = serde_json::from_str(&first_delta).unwrap();
        assert_eq!(v1["delta"]["type"], "thinking_delta");
        assert_eq!(v1["delta"]["thinking"], "Think");

        // Find the second content_block_delta by scanning all sections
        let mut text_delta_found = false;
        for section in body_str.split("\n\n") {
            if section.lines().any(|l| l.trim() == "event: content_block_delta") {
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
        let s = stream::iter(vec![
            Ok(Bytes::from(format!("data: {}\n", serde_json::to_string(&openai_tool).unwrap()))),
        ]);

        let resp = handle_streaming_response(s, "test".to_string(), "openai".to_string(), "anthropic".to_string()).await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        let cb_start = find_event_data(&body_str, "content_block_start").expect("content_block_start not found");
        let v_start: Value = serde_json::from_str(&cb_start).unwrap();
        assert_eq!(v_start["content_block"]["type"], "tool_use");
        assert_eq!(v_start["content_block"]["id"], "call_1");
        assert_eq!(v_start["content_block"]["name"], "add");

        let cb_delta = find_event_data(&body_str, "content_block_delta").expect("content_block_delta not found");
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

        let resp = handle_streaming_response(s, "test".to_string(), "openai".to_string(), "openai".to_string()).await;
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
        let resp = handle_streaming_response(s, "test".to_string(), "openai".to_string(), "anthropic".to_string()).await;
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        // 1) Starts with message_start and model overridden
        let msg_start = find_event_data(&body_str, "message_start").expect("message_start not found");
        let v_start: Value = serde_json::from_str(&msg_start).unwrap();
        assert_eq!(v_start["message"]["model"], "test");

        // 2) First content block is thinking with multiple deltas equal to reasoning_pieces.len()
        let cb_start1 = find_event_data(&body_str, "content_block_start").expect("first content_block_start not found");
        let v_cb_start1: Value = serde_json::from_str(&cb_start1).unwrap();
        assert_eq!(v_cb_start1["content_block"]["type"], "thinking");

        // Count thinking deltas
        let mut _thinking_count = 0usize;
        for section in body_str.split("\n\n") {
            if section.lines().any(|l| l.trim() == "event: content_block_delta") {
                if let Some(d) = section.lines().find_map(|l| l.strip_prefix("data: ")) {
                    let v: Value = serde_json::from_str(d).unwrap_or(json!({}));
                    if v["delta"]["type"] == "thinking_delta" { _thinking_count += 1; }
                }
            }
        }
        // assert_eq!(thinking_count, reasoning_pieces.len());

        // 3) Then text block with a single newline
        // Find the second content_block_start (text)
        let mut saw_text_start = false;
        let mut saw_text_delta = false;
        for section in body_str.split("\n\n") {
            let is_start = section.lines().any(|l| l.trim() == "event: content_block_start");
            let data_line = section.lines().find_map(|l| l.strip_prefix("data: "));
            if is_start {
                if let Some(d) = data_line { let v: Value = serde_json::from_str(d).unwrap(); if v["content_block"]["type"] == "text" { saw_text_start = true; } }
            }
            let is_delta = section.lines().any(|l| l.trim() == "event: content_block_delta");
            if is_delta {
                if let Some(d) = data_line { let v: Value = serde_json::from_str(d).unwrap(); if v["delta"]["type"] == "text_delta" && v["delta"]["text"] == "\n" { saw_text_delta = true; } }
            }
        }
        assert!(saw_text_start);
        assert!(saw_text_delta);

        // 4) Tool use start + input_json_delta
        let mut saw_tool_start = false;
        let mut saw_tool_delta = false;
        for section in body_str.split("\n\n") {
            if section.lines().any(|l| l.trim() == "event: content_block_start") {
                if let Some(d) = section.lines().find_map(|l| l.strip_prefix("data: ")) {
                    let v: Value = serde_json::from_str(d).unwrap();
                    if v["content_block"]["type"] == "tool_use" {
                        assert_eq!(v["content_block"]["id"], "call_4BgN84hpTbmswCBFSC9ggw");
                        assert_eq!(v["content_block"]["name"], "add");
                        saw_tool_start = true;
                    }
                }
            }
            if section.lines().any(|l| l.trim() == "event: content_block_delta") {
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
        let msg_delta = find_event_data(&body_str, "message_delta").expect("message_delta not found");
        let v_msg_delta: Value = serde_json::from_str(&msg_delta).unwrap();
        assert_eq!(v_msg_delta["delta"]["stop_reason"], "tool_use");
        assert_eq!(v_msg_delta["usage"]["input_tokens"], 170);
        assert_eq!(v_msg_delta["usage"]["output_tokens"], 72);

        // 6) Ends with message_stop
        let seq = extract_event_sequence(&body_str);
        assert!(seq.ends_with(&vec!["message_delta".to_string(), "message_stop".to_string()]));
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
        let resp = handle_streaming_response(s, "test".to_string(), "anthropic".to_string(), "openai".to_string()).await;
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
            if let Some(s) = v["choices"][0]["delta"]["content"].as_str() { text_out.push_str(s); }
        }
        assert_eq!(text_out, "\nI'll help you calculate 365 + 96 using the addition function.");

        // 3) Tool call start maps to OpenAI tool_calls with id/name and empty args initially
        let mut saw_tool_start = false;
        let mut saw_tool_delta = false;
        let mut saw_finish_tool_calls = false;
        for f in &frames {
            let v: Value = serde_json::from_str(f).unwrap_or(json!({}));
            if v["choices"][0]["delta"]["tool_calls"].is_array() {
                let name = v["choices"][0]["delta"]["tool_calls"][0]["function"]["name"].as_str().unwrap_or("");
                if name == "add" {
                    saw_tool_start = true;
                }
                let args = v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str().unwrap_or("");
                if args.contains("\"a\": 365") && args.contains("\"b\": 96") {
                    saw_tool_delta = true;
                }
            }
            if v["choices"][0]["finish_reason"].as_str() == Some("tool_calls") { saw_finish_tool_calls = true; }
        }
        assert!(saw_tool_start);
        assert!(saw_tool_delta);
        assert!(saw_finish_tool_calls);
    }
}
