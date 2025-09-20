use serde::{Deserialize, Serialize};
use crate::converters::gemini::GeminiInlineData;
use crate::converters::gemini::gemini_funtion_call::GeminiFunctionCall;
use crate::converters::gemini::gemini_funtion_response::GeminiFunctionResponse;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GeminiPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "thoughtSignature")]
        thought_signature: Option<String>,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
        #[serde(rename = "thoughtSignature")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
    // Minimal shape; other variants omitted for first cut
}
