use serde::{Deserialize, Serialize};
use crate::converters::anthropic::{AnthropicContent, AnthropicUsage};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicStreamMessage {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContent>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
}