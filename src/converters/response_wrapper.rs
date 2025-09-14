use super::openai::OpenAIResponse;
use super::anthropic::AnthropicResponse;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseWrapper {
    OpenAI(OpenAIResponse),
    Anthropic(AnthropicResponse),
}

impl ResponseWrapper {
    pub fn get_openai(&self) -> OpenAIResponse {
        match self {
            ResponseWrapper::OpenAI(resp) => resp.clone(),
            ResponseWrapper::Anthropic(resp) => resp.clone().into(),
        }
    }
    
    pub fn get_anthropic(&self) -> AnthropicResponse {
        match self {
            ResponseWrapper::Anthropic(resp) => resp.clone(),
            ResponseWrapper::OpenAI(resp) => resp.clone().into(),
        }
    }
}