use serde::{Deserialize, Serialize};
use crate::converters::anthropic::AnthropicContent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}