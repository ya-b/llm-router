use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_tool_call::OpenAIToolCall;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}