use crate::converters::openai::openai_image_url::OpenAIImageUrl;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIContentItem {
    pub r#type: String,
    pub text: Option<String>,
    pub image_url: Option<OpenAIImageUrl>,
}
