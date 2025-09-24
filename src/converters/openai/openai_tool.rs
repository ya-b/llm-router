use crate::converters::openai::openai_function::OpenAIFunction;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    pub r#type: String,
    pub function: OpenAIFunction,
}
