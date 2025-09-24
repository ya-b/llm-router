use crate::converters::gemini::GeminiPart;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: Option<String>, // "user" or "model"
    pub parts: Vec<GeminiPart>,
}
