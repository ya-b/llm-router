use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicImageSource {
    pub r#type: String,
    pub media_type: Option<String>,
    pub data: Option<String>,
    pub url: Option<String>
}