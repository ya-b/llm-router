use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: Option<u32>,
    #[serde(rename = "promptTokensDetails")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<Vec<GeminiPromptTokensDetail>>,
    #[serde(rename = "thoughtsTokenCount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPromptTokensDetail {
    pub modality: Option<String>,
    #[serde(rename = "tokenCount")]
    pub token_count: Option<u32>,
}
