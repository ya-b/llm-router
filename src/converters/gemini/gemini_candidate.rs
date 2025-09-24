use crate::converters::gemini::{GeminiContent, GeminiFinishReason};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiCandidate {
    pub content: GeminiContent,
    #[serde(rename = "finishReason")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<GeminiFinishReason>,
    pub index: Option<u32>,
}
