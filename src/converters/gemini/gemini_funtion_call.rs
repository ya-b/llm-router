use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: Value,
    #[serde(rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
}
