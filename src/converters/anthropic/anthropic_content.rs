use serde::{Deserialize, Serialize};
use crate::converters::anthropic::AnthropicContentObject;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Array(Vec<AnthropicContentObject>),
}