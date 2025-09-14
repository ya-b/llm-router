use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_stream_delta::OpenAIStreamDelta;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStreamChoice {
    pub index: i32,
    pub delta: Option<OpenAIStreamDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}