use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_stream_tool_call_function::OpenAIStreamToolCallFunction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStreamToolCall {
    pub index: i32,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: Option<OpenAIStreamToolCallFunction>,
}