use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_content_item::OpenAIContentItem;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    Text(String),
    Array(Vec<OpenAIContentItem>),
}