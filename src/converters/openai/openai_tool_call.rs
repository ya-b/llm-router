use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_tool_call_function::OpenAIToolCallFunction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    pub r#type: String,
    pub function: OpenAIToolCallFunction,
}