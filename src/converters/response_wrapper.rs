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
}