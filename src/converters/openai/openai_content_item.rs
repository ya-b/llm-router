use serde::{Deserialize, Serialize};
use crate::converters::openai::openai_image_url::OpenAIImageUrl;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIContentItem {
    pub r#type: String,
    pub text: Option<String>,
    pub image_url: Option<OpenAIImageUrl>,
}