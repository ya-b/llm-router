use crate::converters::gemini::{GeminiBlockReason, GeminiSafetyRating};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPromptFeedback {
    #[serde(rename = "blockReason")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<GeminiBlockReason>,
    #[serde(rename = "safetyRatings")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<GeminiSafetyRating>>,
}
