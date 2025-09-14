use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_function::OpenAIFunction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    pub r#type: String,
    pub function: OpenAIFunction,
}