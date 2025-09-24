use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Common content part representation used across request and response payloads for the
/// OpenAI Responses API. The variants follow the official documentation and keep a
/// `HashMap` of extra fields so we don't lose new attributes as the API evolves.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    InputText {
        text: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    OutputText {
        text: String,
        annotations: Vec<ResponsesAnnotation>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    Refusal {
        refusal: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    FileSearchCall {
        id: String,
        queries: Vec<Value>,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        results: Option<Value>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    WebSearchCall {
        id: String,
        status: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    FunctionCall {
        arguments: String,
        call_id: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    ComputerCall {
        action: Value,
        call_id: String,
        id: String,
        pending_safety_checks: Vec<Value>,
        status: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    ComputerCallOutput {
        call_id: String,
        output: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        acknowledged_safety_checks: Option<Vec<Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    Reasoning {
        id: String,
        summary: Vec<ResponsesReasoningSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    ItemReference {
        id: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    FileCitation {
        file_id: String,
        index: u64,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    UrlCitation {
        start_index: u64,
        end_index: u64,
        title: String,
        url: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    FilePath {
        file_id: String,
        index: u64,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesAnnotation {
    FileCitation {
        file_id: String,
        index: u64,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    UrlCitation {
        start_index: u64,
        end_index: u64,
        title: String,
        url: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    FilePath {
        file_id: String,
        index: u64,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningSummary {
    SummaryText {
        text: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
    #[serde(other)]
    Unknown,
}
