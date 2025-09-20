use serde::{Deserialize, Serialize};
use crate::converters::gemini::GeminiPart;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: Option<String>, // "user" or "model"
    pub parts: Vec<GeminiPart>,
}