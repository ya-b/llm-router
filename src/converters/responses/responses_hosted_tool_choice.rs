use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesHostedToolChoice {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[serde(default)]
    pub extra_fields: HashMap<String, Value>,
}
