use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_response_message::OpenAIResponseMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIChoice {
    pub index: i32,
    pub message: OpenAIResponseMessage,
    pub finish_reason: String,
}