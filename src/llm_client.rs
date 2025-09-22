use crate::config::{ApiType, ModelConfig};
use crate::converters::request_wrapper::RequestWrapper;
use anyhow::Result;
use reqwest::header::{HeaderName, HeaderValue};
use std::future::Future;
use std::sync::Arc;
use tracing::{debug, info, warn};
use crate::request_id::RequestId;

#[derive(Debug)]
pub struct LlmClient {
    http_client: Arc<reqwest::Client>,
}

impl LlmClient {
    pub fn new(http_client: Arc<reqwest::Client>) -> Self {
        Self { http_client }
    }

    fn build_target_url(model_config: &ModelConfig, request: &RequestWrapper) -> String {
        let api_base = &model_config.llm_params.api_base;
        match model_config.llm_params.api_type {
            ApiType::Anthropic => {
                let path = "v1/messages";
                if api_base.ends_with('/') { format!("{}{}", api_base, path) } else { format!("{}/{}", api_base, path) }
            }
            ApiType::OpenAI => {
                let path = "chat/completions";
                if api_base.ends_with('/') { format!("{}{}", api_base, path) } else { format!("{}/{}", api_base, path) }
            }
            ApiType::Gemini => {
                // Determine streaming and construct proper Gemini path
                let model = &model_config.llm_params.model;
                let is_stream = request.is_stream().unwrap_or(false);
                let path = if is_stream { format!("models/{}:streamGenerateContent", model) } else { format!("models/{}:generateContent", model) };
                let mut base = if api_base.ends_with('/') { format!("{}{}", api_base, path) } else { format!("{}/{}", api_base, path) };
                if !model_config.llm_params.api_key.is_empty() {
                    if is_stream { base = format!("{}?alt=sse&key={}", base, model_config.llm_params.api_key); }
                    else { base = format!("{}?key={}", base, model_config.llm_params.api_key); }
                } else if is_stream {
                    base = format!("{}?alt=sse", base);
                }
                base
            }
        }
    }

    pub fn forward_request(
        &self,
        request: &RequestWrapper,
        model_config: &ModelConfig,
        request_id: &RequestId,
    ) -> impl Future<Output = Result<reqwest::Response, reqwest::Error>> {
        // Prepare body per upstream api type to know if streaming is needed for Gemini
        let mut target_body = match model_config.llm_params.api_type {
            ApiType::Anthropic => {
                let mut anthropic_req = request.get_anthropic();
                anthropic_req.model = model_config.llm_params.model.clone();
                serde_json::to_value(anthropic_req).expect("Failed to serialize converted Anthropic request")
            }
            ApiType::OpenAI => {
                let mut openai_req = request.get_openai();
                openai_req.model = model_config.llm_params.model.clone();
                serde_json::to_value(openai_req).expect("Failed to serialize converted OpenAI request")
            }
            ApiType::Gemini => {
                let mut gemini_req = request.get_gemini();
                // Path uses model; body does not include model
                gemini_req.model = model_config.llm_params.model.clone();
                serde_json::to_value(gemini_req).expect("Failed to serialize converted Gemini request")
            }
        };

        // Build target URL (Gemini stream/non-stream handled inside)
        let target_url = Self::build_target_url(model_config, request);

        let mut target_request = self
            .http_client
            .post(&target_url)
            .header("Content-Type", "application/json");

        // Propagate request id upstream
        if let Ok(val) = HeaderValue::from_str(&request_id.0) {
            target_request = target_request.header("x-request-id", val);
        }

        match model_config.llm_params.api_type {
            ApiType::Anthropic => {
                target_request = target_request.header("x-api-key", model_config.llm_params.api_key.to_string());
            }
            ApiType::OpenAI => {
                target_request = target_request.header(
                    "Authorization",
                    format!("Bearer {}", model_config.llm_params.api_key),
                );
            }
            ApiType::Gemini => {
                // Gemini commonly uses API key query param; no auth header required.
                // For SSE streaming, hint Accept header
                if request.is_stream().unwrap_or(false) {
                    target_request = target_request.header("Accept", "text/event-stream");
                }
            }
        }

        // Apply rewrite_header functionality
        if let serde_json::Value::Object(map) = &model_config.llm_params.rewrite_header {
            for (k, v) in map {
                if v.is_object() || v.is_array() {
                    continue;
                }

                let name = match HeaderName::try_from(k.as_str()) {
                    Ok(n) => n,
                    Err(e) => {
                        warn!("Invalid header name in rewrite_header: {}: {}", k, e);
                        continue;
                    }
                };

                let value_str = if let Some(s) = v.as_str() {
                    s.to_string()
                } else {
                    v.to_string().trim_matches('"').to_string()
                };

                match HeaderValue::from_str(&value_str) {
                    Ok(val) => {
                        target_request = target_request.header(name.clone(), val);
                    }
                    Err(e) => {
                        warn!("Invalid header value for {}: {}", k, e);
                    }
                }
            }
        }

        if let serde_json::Value::Object(map) = &model_config.llm_params.rewrite_body {
            if let Some(t_body) = target_body.as_object_mut() {
                for (k, v) in map {
                    t_body.insert(k.clone(), v.clone());
                }
            }
        }

        info!("Forwarding request to: {}", target_url);
        debug!(
            "request body: {}",
            serde_json::to_string(&target_body).expect("Failed to serialize request")
        );
        target_request.json(&target_body).send()
    }
}
