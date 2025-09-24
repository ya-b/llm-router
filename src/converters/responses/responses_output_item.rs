use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::responses_content::ResponsesContentPart;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        status: String,
        role: String,
        #[serde(default)]
        content: Vec<ResponsesContentPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<HashMap<String, Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<Value>,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    ToolUse {
        #[serde(flatten)]
        data: HashMap<String, Value>,
    },
    #[serde(other)]
    Unknown,
}
