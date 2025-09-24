use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesFileSearchFilter {
    #[serde(rename = "eq")]
    Eq {
        key: String,
        value: Value,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "ne")]
    Ne {
        key: String,
        value: Value,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "gt")]
    Gt {
        key: String,
        value: Value,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "gte")]
    Gte {
        key: String,
        value: Value,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "lt")]
    Lt {
        key: String,
        value: Value,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "lte")]
    Lte {
        key: String,
        value: Value,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "and")]
    And {
        filters: Vec<ResponsesFileSearchFilter>,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "or")]
    Or {
        filters: Vec<ResponsesFileSearchFilter>,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(other)]
    Unknown,
}
