use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::responses_file_search_filter::ResponsesFileSearchFilter;
use super::responses_file_search_ranking_options::ResponsesFileSearchRankingOptions;
use super::responses_user_location::ResponsesUserLocation;

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesTool {
    #[serde(rename = "file_search")]
    FileSearch {
        vector_store_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<ResponsesFileSearchFilter>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ranking_options: Option<ResponsesFileSearchRankingOptions>,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "function")]
    Function {
        name: String,
        parameters: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "web_search_preview", alias = "web_search_preview_2025_03_11")]
    WebSearchPreview {
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<ResponsesUserLocation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        domains: Option<Vec<String>>,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview {
        display_height: u32,
        display_width: u32,
        environment: String,
        #[serde(flatten)]
        #[serde(default)]
        extra: HashMap<String, Value>,
    },
    #[serde(other)]
    Unknown,
}
