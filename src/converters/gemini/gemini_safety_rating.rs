use serde::{Deserialize, Serialize};
use crate::converters::gemini::{GeminiHarmCategory, GeminiHarmProbability};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiSafetyRating {
    #[serde(rename = "category")]
    pub category: GeminiHarmCategory,
    #[serde(rename = "probability")]
    pub probability: GeminiHarmProbability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked: Option<bool>,
}