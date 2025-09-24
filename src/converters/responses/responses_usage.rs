use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::responses_usage_detail::ResponsesUsageDetail;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<ResponsesUsageDetail>,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<ResponsesUsageDetail>,
    pub total_tokens: u64,
    #[serde(flatten)]
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}
