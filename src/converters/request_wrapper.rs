use super::openai::OpenAIRequest;
use super::anthropic::AnthropicRequest;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestWrapper {
    OpenAI(OpenAIRequest),
    Anthropic(AnthropicRequest),
}

impl RequestWrapper {
    pub fn get_openai(&self) -> OpenAIRequest {
        match self {
            RequestWrapper::OpenAI(req) => req.clone(),
            RequestWrapper::Anthropic(req) => req.clone().into(),
        }
    }
    
    pub fn get_anthropic(&self) -> AnthropicRequest {
        match self {
            RequestWrapper::Anthropic(req) => req.clone(),
            RequestWrapper::OpenAI(req) => req.clone().into(),
        }
    }

    pub fn get_model(&self) -> &String {
        match self {
            RequestWrapper::OpenAI(req) => &req.model,
            RequestWrapper::Anthropic(req) => &req.model,
        }
    }

    pub fn is_stream(&self) -> &Option<bool> {
        match self {
            RequestWrapper::OpenAI(req) => &req.stream,
            RequestWrapper::Anthropic(req) => &req.stream,
        }
    }
}