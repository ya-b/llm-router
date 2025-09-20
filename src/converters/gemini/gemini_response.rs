use crate::converters::openai::OpenAIResponse;
use serde::{Deserialize, Serialize};

use crate::converters::gemini::{
    gemini_candidate::GeminiCandidate, gemini_content::GeminiContent,
    gemini_finish_reason::GeminiFinishReason, gemini_funtion_call::GeminiFunctionCall,
    gemini_part::GeminiPart, gemini_prompt_feedback::GeminiPromptFeedback,
    gemini_usage::GeminiUsage,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<GeminiUsage>,
    #[serde(rename = "modelVersion")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(rename = "promptFeedback")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<GeminiPromptFeedback>,
    #[serde(rename = "responseId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

impl From<OpenAIResponse> for GeminiResponse {
    fn from(openai_resp: OpenAIResponse) -> Self {
        let mut parts: Vec<GeminiPart> = Vec::new();

        if let Some(reasoning) = &openai_resp.choices[0].message.reasoning_content {
            if !reasoning.trim().is_empty() {
                parts.push(GeminiPart::Text {
                    text: reasoning.clone(),
                    thought: Some(true),
                    thought_signature: None,
                });
            }
        }

        if let Some(content) = &openai_resp.choices[0].message.content {
            if !content.trim().is_empty() {
                parts.push(GeminiPart::Text {
                    text: content.clone(),
                    thought: None,
                    thought_signature: None,
                });
            }
        }

        if let Some(tool_calls) = &openai_resp.choices[0].message.tool_calls {
            for tc in tool_calls.iter() {
                let args = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                    .unwrap_or_else(|_| serde_json::json!({}));
                parts.push(GeminiPart::FunctionCall {
                    function_call: GeminiFunctionCall {
                        name: tc.function.name.clone(),
                        args,
                        thought_signature: None,
                    },
                    thought_signature: None,
                });
            }
        }

        let finish_reason = match openai_resp.choices[0].finish_reason.as_str() {
            "stop" => Some(GeminiFinishReason::Stop),
            "length" => Some(GeminiFinishReason::MaxTokens),
            // No perfect mapping for tool_calls; leave unspecified
            "tool_calls" => Some(GeminiFinishReason::FinishReasonUnspecified),
            _ => Some(GeminiFinishReason::FinishReasonUnspecified),
        };

        let candidate = GeminiCandidate {
            content: GeminiContent {
                role: Some("model".to_string()),
                parts,
            },
            finish_reason,
            index: None,
        };

        GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: openai_resp.usage.map(|u| GeminiUsage {
                prompt_token_count: Some(u.prompt_tokens),
                candidates_token_count: Some(u.completion_tokens),
                total_token_count: Some(u.total_tokens),
                prompt_tokens_details: None,
                thoughts_token_count: None,
            }),
            model_version: Some(openai_resp.model),
            prompt_feedback: None,
            response_id: Some(openai_resp.id),
        }
    }
}
