use serde::{Deserialize, Serialize};

use crate::converters::gemini::gemini_funtion_call::GeminiFunctionCall;
use crate::converters::gemini::{GeminiCandidate, GeminiFinishReason, GeminiUsage};
use crate::converters::gemini::{GeminiContent, GeminiPart};
use crate::converters::openai::{OpenAIStreamChoice, OpenAIStreamChunk};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiStreamChunk {
    pub candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<GeminiUsage>,
    #[serde(rename = "modelVersion")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(rename = "responseId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

impl From<OpenAIStreamChunk> for GeminiStreamChunk {
    fn from(openai_chunk: OpenAIStreamChunk) -> Self {
        let candidates = openai_chunk
            .choices
            .unwrap_or_default()
            .into_iter()
            .map(map_openai_choice_to_gemini_candidate)
            .collect();

        GeminiStreamChunk {
            candidates,
            usage_metadata: openai_chunk.usage.map(|u| GeminiUsage {
                prompt_token_count: Some(u.prompt_tokens as u32),
                candidates_token_count: Some(u.completion_tokens as u32),
                total_token_count: Some(u.total_tokens as u32),
                prompt_tokens_details: None,
                thoughts_token_count: None,
            }),
            model_version: Some(openai_chunk.model),
            response_id: Some(openai_chunk.id),
        }
    }
}

fn map_openai_choice_to_gemini_candidate(choice: OpenAIStreamChoice) -> GeminiCandidate {
    let mut parts: Vec<GeminiPart> = Vec::new();
    let mut role: Option<String> = None;

    if let Some(delta) = choice.delta {
        if let Some(r) = delta.role {
            role = Some(if r == "assistant" { "model".into() } else { r });
        }
        if let Some(reasoning) = delta.reasoning_content {
            if !reasoning.is_empty() {
                parts.push(GeminiPart::Text {
                    text: reasoning,
                    thought: Some(true),
                    thought_signature: None,
                });
            }
        }
        if let Some(content) = delta.content {
            if !content.is_empty() {
                parts.push(GeminiPart::Text {
                    text: content,
                    thought: None,
                    thought_signature: None,
                });
            }
        }
        if let Some(tool_calls) = delta.tool_calls {
            for tc in tool_calls.into_iter() {
                if let Some(func) = tc.function {
                    let name = func.name.unwrap_or_default();
                    let args_val = func
                        .arguments
                        .and_then(|a| serde_json::from_str::<serde_json::Value>(&a).ok())
                        .unwrap_or_else(|| serde_json::Value::String(String::new()));
                    parts.push(GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name,
                            args: args_val,
                            thought_signature: None,
                        },
                        thought_signature: None,
                    });
                }
            }
        }
    }

    let finish_reason = choice.finish_reason.and_then(map_openai_finish_reason);

    GeminiCandidate {
        content: GeminiContent { role, parts },
        finish_reason,
        index: Some(choice.index as u32),
    }
}

fn map_openai_finish_reason(r: String) -> Option<GeminiFinishReason> {
    match r.as_str() {
        "stop" => Some(GeminiFinishReason::Stop),
        "length" => Some(GeminiFinishReason::MaxTokens),
        // Tool call related stop doesn't have a direct mapping; keep as unspecified
        "tool_calls" => Some(GeminiFinishReason::FinishReasonUnspecified),
        _ => Some(GeminiFinishReason::FinishReasonUnspecified),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_str() {
        let text = "{\"candidates\": [{\"content\": {\"parts\": [{\"functionCall\": {\"name\": \"schedule_meeting\",\"args\": {\"attendees\": [\"Bob\",\"Alice\"],\"date\": \"2025-03-27\",\"topic\": \"Q3 planning\",\"time\": \"10:00\"}},\"thoughtSignature\": \"thoughtSignature value\"}],\"role\": \"model\"},\"finishReason\": \"STOP\",\"index\": 0}],\"usageMetadata\": {\"promptTokenCount\": 165,\"candidatesTokenCount\": 49,\"totalTokenCount\": 562,\"promptTokensDetails\": [{\"modality\": \"TEXT\",\"tokenCount\": 165}],\"thoughtsTokenCount\": 348},\"modelVersion\": \"gemini-2.5-pro\",\"responseId\": \"iJDOaOzkBM70jMcPxJmmyAw\"}";
        let info = match serde_json::from_str::<GeminiStreamChunk>(&text) {
            Ok(chunk) => "success".to_string(),
            Err(e) => {
                format!("{:?}", e)
            }
        };
        assert_eq!(info, "success");
    }
}
