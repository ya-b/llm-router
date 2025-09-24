use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::responses_text_format_json_schema::ResponsesTextFormatJsonSchema;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTextFormat {
    Text {
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    JsonSchema(ResponsesTextFormatJsonSchema),
    JsonObject {
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(other)]
    Unknown,
}
