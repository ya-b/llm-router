use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesFunctionToolChoice {
    pub name: String,
    #[serde(rename = "type")]
    #[serde(default = "ResponsesFunctionToolChoice::default_type")]
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    #[serde(default)]
    pub extra_fields: HashMap<String, Value>,
}

impl ResponsesFunctionToolChoice {
    fn default_type() -> String {
        "function".to_string()
    }
}
