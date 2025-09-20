use serde::{Deserialize, Serialize};
use crate::converters::gemini::GeminiFunctionDeclaration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}