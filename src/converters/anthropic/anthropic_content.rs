use crate::converters::anthropic::AnthropicContentObject;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Array(Vec<AnthropicContentObject>),
}
