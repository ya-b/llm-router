use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystemContent {
    Text(String),
    Array(Vec<AnthropicSystemContentObject>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicSystemContentObject {
    #[serde(rename = "text")]
    Text { text: String },
}