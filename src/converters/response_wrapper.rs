use super::anthropic::AnthropicResponse;
use super::gemini::GeminiResponse;
use super::openai::OpenAIResponse;
use super::responses::ResponsesResponse;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseWrapper {
    OpenAI(OpenAIResponse),
    Anthropic(AnthropicResponse),
    Gemini(GeminiResponse),
    Responses(ResponsesResponse),
}

impl ResponseWrapper {}
