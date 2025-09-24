use crate::converters::anthropic::{AnthropicContentObject, AnthropicResponse};
use crate::converters::gemini::{GeminiFinishReason, GeminiPart, GeminiResponse};
use crate::converters::helpers;
use crate::converters::openai::openai_completion_tokens_details::OpenAICompletionTokensDetails;
use crate::converters::openai::openai_prompt_tokens_details::OpenAIPromptTokensDetails;
use crate::converters::openai::{
    OpenAIChoice, OpenAIResponseMessage, OpenAIToolCall, OpenAIToolCallFunction, OpenAIUsage,
};
use crate::converters::responses::{
    ResponsesContentPart, ResponsesOutputItem, ResponsesReasoningSummary, ResponsesResponse,
    ResponsesUsage, ResponsesUsageDetail,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIResponse {
    pub id: String,
    pub object: Option<String>,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

impl From<AnthropicResponse> for OpenAIResponse {
    fn from(anthropic_resp: AnthropicResponse) -> Self {
        let mut reasoning_text = String::new();
        let mut content_text = String::new();
        let mut tool_calls = Vec::new();

        for content in anthropic_resp.content {
            match content {
                AnthropicContentObject::Text { text } => {
                    content_text.push_str(&text);
                }
                AnthropicContentObject::Thinking {
                    thinking,
                    signature: _,
                } => {
                    reasoning_text.push_str(&thinking);
                }
                AnthropicContentObject::RedactedThinking { data } => {
                    reasoning_text.push_str(
                        format!("<redacted_thinking>{}</redacted_thinking>", &data).as_str(),
                    );
                }
                AnthropicContentObject::Image { source: _ } => {
                    // 图片内容在响应中不太常见，暂不处理
                }
                AnthropicContentObject::ToolUse { id, name, input } => {
                    tool_calls.push(OpenAIToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: OpenAIToolCallFunction {
                            name,
                            arguments: serde_json::to_string(&input)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                    });
                }
                AnthropicContentObject::ToolResult {
                    tool_use_id: _,
                    content: _,
                } => {
                    // 工具结果在响应中不太常见，暂不处理
                }
            }
        }

        OpenAIResponse {
            id: anthropic_resp.id,
            object: Some("chat.completion".to_string()),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            model: anthropic_resp.model,
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIResponseMessage {
                    role: "assistant".to_string(),
                    content: if content_text.is_empty() {
                        None
                    } else {
                        Some(content_text)
                    },
                    reasoning_content: if reasoning_text.is_empty() {
                        None
                    } else {
                        Some(reasoning_text)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                },
                finish_reason: match anthropic_resp.stop_reason {
                    Some(s) => helpers::map_anthropic_stop_reason_to_openai(Some(&Value::String(
                        s.clone(),
                    )))
                    .as_str()
                    .unwrap_or("stop")
                    .to_string(),
                    None => "stop".to_string(),
                },
            }],
            usage: anthropic_resp.usage.map(|usage| OpenAIUsage {
                prompt_tokens: usage.input_tokens,
                completion_tokens: usage.output_tokens,
                total_tokens: usage.input_tokens + usage.output_tokens,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            }),
            service_tier: None,
            system_fingerprint: None,
        }
    }
}

impl From<GeminiResponse> for OpenAIResponse {
    fn from(resp: GeminiResponse) -> Self {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let (text, reasoning_text, tool_calls, finish_reason) =
            if let Some(first) = resp.candidates.get(0) {
                let mut t = String::new();
                let mut rt = String::new();
                let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
                let mut saw_tool_call = false;
                for (idx, p) in first.content.parts.iter().enumerate() {
                    match p {
                        GeminiPart::Text {
                            text,
                            thought,
                            thought_signature: _,
                        } => {
                            if let Some(true) = thought {
                                rt.push_str(&text);
                            } else {
                                t.push_str(&text);
                            }
                        }
                        GeminiPart::InlineData { inline_data: _ } => {}
                        GeminiPart::FunctionCall {
                            function_call,
                            thought_signature: _,
                        } => {
                            saw_tool_call = true;
                            tool_calls.push(OpenAIToolCall {
                                id: format!("tool_call_{}", idx),
                                r#type: "function".to_string(),
                                function: OpenAIToolCallFunction {
                                    name: function_call.name.clone(),
                                    arguments: serde_json::to_string(&function_call.args)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            });
                        }
                        GeminiPart::FunctionResponse {
                            function_response: _,
                        } => {}
                    }
                }
                let fr = if saw_tool_call {
                    "tool_calls".to_string()
                } else {
                    match first.finish_reason.as_ref() {
                        Some(GeminiFinishReason::Stop) => "stop".to_string(),
                        Some(GeminiFinishReason::MaxTokens) => "length".to_string(),
                        _ => "stop".to_string(),
                    }
                };
                (
                    Some(t),
                    Some(rt),
                    if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    fr,
                )
            } else {
                (None, None, None, "stop".to_string())
            };

        OpenAIResponse {
            id: format!("gen-{}", now_secs),
            object: Some("chat.completion".to_string()),
            created: now_secs,
            model: resp.model_version.unwrap_or_else(|| "gemini".to_string()),
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIResponseMessage {
                    role: "assistant".to_string(),
                    content: text,
                    reasoning_content: match reasoning_text {
                        Some(s) if !s.is_empty() => Some(s),
                        _ => None,
                    },
                    tool_calls,
                },
                finish_reason,
            }],
            usage: resp.usage_metadata.as_ref().map(|u| OpenAIUsage {
                prompt_tokens: u.prompt_token_count.unwrap_or(0),
                completion_tokens: u.candidates_token_count.unwrap_or(0),
                total_tokens: u.total_token_count.unwrap_or(0),
                completion_tokens_details: None,
                prompt_tokens_details: None,
            }),
            system_fingerprint: None,
            service_tier: None,
        }
    }
}

impl From<ResponsesResponse> for OpenAIResponse {
    fn from(resp: ResponsesResponse) -> Self {
        let mut choices = Vec::new();
        let mut pending_tool_calls: Vec<OpenAIToolCall> = Vec::new();

        for item in resp.output.into_iter() {
            match item {
                ResponsesOutputItem::Message {
                    role,
                    content,
                    status,
                    ..
                } => {
                    let mut fields = extract_message_fields(content);

                    if fields.tool_calls.is_empty() && !pending_tool_calls.is_empty() {
                        fields.tool_calls.extend(pending_tool_calls.drain(..));
                    }

                    let finish_reason = if !fields.tool_calls.is_empty() {
                        "tool_calls".to_string()
                    } else {
                        map_responses_status_to_finish_reason(Some(status.as_str()))
                    };

                    choices.push(OpenAIChoice {
                        index: choices.len() as i32,
                        message: OpenAIResponseMessage {
                            role,
                            content: fields.text,
                            reasoning_content: fields.reasoning,
                            tool_calls: if fields.tool_calls.is_empty() {
                                None
                            } else {
                                Some(fields.tool_calls)
                            },
                        },
                        finish_reason,
                    });
                }
                ResponsesOutputItem::FunctionCall {
                    call_id: _,
                    name,
                    arguments,
                    id,
                    ..
                } => {
                    pending_tool_calls.push(OpenAIToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: OpenAIToolCallFunction { name, arguments },
                    });
                }
                ResponsesOutputItem::ToolUse { .. } | ResponsesOutputItem::Unknown => {
                    // These variants do not have a direct OpenAI Chat Completion equivalent.
                }
            }
        }

        if !pending_tool_calls.is_empty() {
            if let Some(last_choice) = choices.last_mut() {
                let mut calls = last_choice.message.tool_calls.take().unwrap_or_default();
                calls.extend(pending_tool_calls);
                last_choice.message.tool_calls = Some(calls);
                if last_choice.finish_reason != "tool_calls" {
                    last_choice.finish_reason = "tool_calls".to_string();
                }
            }
        }

        if choices.is_empty() {
            choices.push(OpenAIChoice {
                index: 0,
                message: OpenAIResponseMessage {
                    role: "assistant".to_string(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: map_responses_status_to_finish_reason(Some(resp.status.as_str())),
            });
        }

        OpenAIResponse {
            id: resp.id,
            object: resp.object.or_else(|| Some("chat.completion".to_string())),
            created: resp.created_at,
            model: resp.model,
            choices,
            usage: resp.usage.map(convert_usage_to_openai),
            system_fingerprint: None,
            service_tier: None,
        }
    }
}

struct ExtractedFields {
    text: Option<String>,
    reasoning: Option<String>,
    tool_calls: Vec<OpenAIToolCall>,
}

fn extract_message_fields(parts: Vec<ResponsesContentPart>) -> ExtractedFields {
    let mut text_buffer = String::new();
    let mut reasoning_segments: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();

    for part in parts {
        match part {
            ResponsesContentPart::InputText { text, .. }
            | ResponsesContentPart::OutputText { text, .. }
            | ResponsesContentPart::Refusal { refusal: text, .. } => {
                if !text_buffer.is_empty() {
                    text_buffer.push_str("\n");
                }
                text_buffer.push_str(&text);
            }
            ResponsesContentPart::Reasoning { summary, .. } => {
                let segment = summary
                    .into_iter()
                    .filter_map(|item| match item {
                        ResponsesReasoningSummary::SummaryText { text, .. } => Some(text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !segment.is_empty() {
                    reasoning_segments.push(segment);
                }
            }
            ResponsesContentPart::FunctionCall {
                call_id,
                name,
                arguments,
                id,
                ..
            } => {
                tool_calls.push(OpenAIToolCall {
                    id: id.unwrap_or_else(|| call_id.clone()),
                    r#type: "function".to_string(),
                    function: OpenAIToolCallFunction { name, arguments },
                });
            }
            ResponsesContentPart::FunctionCallOutput { .. }
            | ResponsesContentPart::InputImage { .. }
            | ResponsesContentPart::InputFile { .. }
            | ResponsesContentPart::FileSearchCall { .. }
            | ResponsesContentPart::WebSearchCall { .. }
            | ResponsesContentPart::ComputerCall { .. }
            | ResponsesContentPart::ComputerCallOutput { .. }
            | ResponsesContentPart::ItemReference { .. }
            | ResponsesContentPart::FileCitation { .. }
            | ResponsesContentPart::UrlCitation { .. }
            | ResponsesContentPart::FilePath { .. }
            | ResponsesContentPart::Unknown => {}
        }
    }

    ExtractedFields {
        text: if text_buffer.is_empty() {
            None
        } else {
            Some(text_buffer)
        },
        reasoning: if reasoning_segments.is_empty() {
            None
        } else {
            Some(reasoning_segments.join("\n"))
        },
        tool_calls,
    }
}

fn map_responses_status_to_finish_reason(status: Option<&str>) -> String {
    match status {
        Some("incomplete") => "length".to_string(),
        Some("cancelled") => "cancelled".to_string(),
        Some("failed") => "failed".to_string(),
        Some("requires_action") => "tool_calls".to_string(),
        _ => "stop".to_string(),
    }
}

fn convert_usage_to_openai(usage: ResponsesUsage) -> OpenAIUsage {
    let prompt_tokens = usage.input_tokens.try_into().unwrap_or(u32::MAX);
    let completion_tokens = usage.output_tokens.try_into().unwrap_or(u32::MAX);
    let total_tokens = usage.total_tokens.try_into().unwrap_or(u32::MAX);

    OpenAIUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        completion_tokens_details: usage
            .output_tokens_details
            .and_then(convert_usage_detail_to_completion_detail),
        prompt_tokens_details: usage
            .input_tokens_details
            .and_then(convert_usage_detail_to_prompt_detail),
    }
}

fn convert_usage_detail_to_completion_detail(
    detail: ResponsesUsageDetail,
) -> Option<OpenAICompletionTokensDetails> {
    let ResponsesUsageDetail {
        reasoning_tokens, ..
    } = detail;

    reasoning_tokens.map(|value| OpenAICompletionTokensDetails {
        reasoning_tokens: Some(value as u32),
        audio_tokens: None,
        accepted_prediction_tokens: None,
        rejected_prediction_tokens: None,
    })
}

fn convert_usage_detail_to_prompt_detail(
    detail: ResponsesUsageDetail,
) -> Option<OpenAIPromptTokensDetails> {
    let ResponsesUsageDetail { cached_tokens, .. } = detail;

    cached_tokens.map(|value| OpenAIPromptTokensDetails {
        audio_tokens: None,
        cached_tokens: Some(value as u32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_to_openai_response() {
        // 测试基本的文本响应
        let json_response = json!({
            "id": "msg_123",
            "model": "test111",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello, how can I help you today?"
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 9,
                "output_tokens": 12
            }
        });
        let anthropic_response: AnthropicResponse =
            serde_json::from_value(json_response).expect("Failed to parse Anthropic response");

        let openai_response: OpenAIResponse = anthropic_response.into();

        assert_eq!(openai_response.object.unwrap(), "chat.completion");
        assert_eq!(openai_response.choices[0].message.role, "assistant");
        assert_eq!(
            openai_response.choices[0].message.content.as_ref().unwrap(),
            "Hello, how can I help you today?"
        );
        assert_eq!(openai_response.choices[0].finish_reason, "stop");
        if let Some(usage) = &openai_response.usage {
            assert_eq!(usage.prompt_tokens, 9);
            assert_eq!(usage.completion_tokens, 12);
            assert_eq!(usage.total_tokens, 21);
        } else {
            panic!("Expected AnthropicUsage::usage");
        }
    }

    #[test]
    fn test_anthropic_to_openai_response_with_thinking() {
        // 测试包含推理内容的响应
        let json_response = json!({
            "id": "msg_456",
            "model": "test111",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I need to think about this step by step."
                },
                {
                    "type": "text",
                    "text": "The answer is 42."
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 20
            }
        });

        let anthropic_response: AnthropicResponse =
            serde_json::from_value(json_response).expect("Failed to parse Anthropic response");

        let openai_response: OpenAIResponse = anthropic_response.into();

        assert_eq!(openai_response.object.unwrap(), "chat.completion");
        assert_eq!(openai_response.choices[0].message.role, "assistant");
        assert_eq!(
            openai_response.choices[0].message.content.as_ref().unwrap(),
            "The answer is 42."
        );
        assert_eq!(
            openai_response.choices[0]
                .message
                .reasoning_content
                .as_ref()
                .unwrap(),
            "I need to think about this step by step."
        );
        assert_eq!(openai_response.choices[0].finish_reason, "stop");
        if let Some(usage) = &openai_response.usage {
            assert_eq!(usage.prompt_tokens, 15);
            assert_eq!(usage.completion_tokens, 20);
            assert_eq!(usage.total_tokens, 35);
        } else {
            panic!("Expected AnthropicUsage::usage");
        }
    }

    #[test]
    fn test_anthropic_to_openai_response_with_tool_calls() {
        // 测试包含工具调用的响应
        let json_response = json!({
            "id": "msg_789",
            "model": "test111",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you get the weather."
                },
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {
                        "location": "San Francisco, CA"
                    }
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 25,
                "output_tokens": 30
            }
        });

        let anthropic_response: AnthropicResponse =
            serde_json::from_value(json_response).expect("Failed to parse Anthropic response");

        let openai_response: OpenAIResponse = anthropic_response.into();

        assert_eq!(openai_response.object.unwrap(), "chat.completion");
        assert_eq!(openai_response.choices[0].message.role, "assistant");
        assert_eq!(
            openai_response.choices[0].message.content.as_ref().unwrap(),
            "I'll help you get the weather."
        );
        assert_eq!(
            openai_response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()[0]
                .id,
            "tool_123"
        );
        assert_eq!(
            openai_response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()[0]
                .function
                .name,
            "get_weather"
        );
        assert_eq!(
            openai_response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()[0]
                .function
                .arguments,
            "{\"location\":\"San Francisco, CA\"}"
        );
        assert_eq!(openai_response.choices[0].finish_reason, "tool_calls");
        if let Some(usage) = &openai_response.usage {
            assert_eq!(usage.prompt_tokens, 25);
            assert_eq!(usage.completion_tokens, 30);
            assert_eq!(usage.total_tokens, 55);
        } else {
            panic!("Expected AnthropicUsage::usage");
        }
    }

    #[test]
    fn test_anthropic_to_openai_response_empty_content() {
        // 测试空内容的响应
        let json_response = json!({
            "id": "msg_empty",
            "model": "test111",
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn"
        });

        let anthropic_response: AnthropicResponse =
            serde_json::from_value(json_response).expect("Failed to parse Anthropic response");

        let openai_response: OpenAIResponse = anthropic_response.into();

        assert_eq!(openai_response.object.unwrap(), "chat.completion");
        assert_eq!(openai_response.choices[0].message.role, "assistant");
        assert_eq!(openai_response.choices[0].message.content, None);
        assert_eq!(openai_response.choices[0].finish_reason, "stop");
    }

    #[test]
    fn test_anthropic_to_openai_response_with_redacted_thinking() {
        // 测试包含编辑后推理内容的响应
        let json_response = json!({
            "id": "msg_redacted",
            "model": "test111",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "redacted_thinking",
                    "data": "sensitive information"
                },
                {
                    "type": "text",
                    "text": "I can't share that information."
                }
            ],
            "stop_reason": "end_turn"
        });

        let anthropic_response: AnthropicResponse =
            serde_json::from_value(json_response).expect("Failed to parse Anthropic response");

        let openai_response: OpenAIResponse = anthropic_response.into();

        assert_eq!(openai_response.object.unwrap(), "chat.completion");
        assert_eq!(openai_response.choices[0].message.role, "assistant");
        assert_eq!(
            openai_response.choices[0].message.content.as_ref().unwrap(),
            "I can't share that information."
        );
        assert_eq!(
            openai_response.choices[0]
                .message
                .reasoning_content
                .as_ref()
                .unwrap(),
            "<redacted_thinking>sensitive information</redacted_thinking>"
        );
        assert_eq!(openai_response.choices[0].finish_reason, "stop");
    }

    #[test]
    fn test_anthropic_to_openai_response_max_tokens_stop_reason() {
        // 测试 max_tokens 停止原因
        let json_response = json!({
            "id": "msg_max",
            "model": "test111",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "This is a truncated response because"
                }
            ],
            "stop_reason": "max_tokens"
        });

        let anthropic_response: AnthropicResponse =
            serde_json::from_value(json_response).expect("Failed to parse Anthropic response");

        let openai_response: OpenAIResponse = anthropic_response.into();

        assert_eq!(openai_response.object.unwrap(), "chat.completion");
        assert_eq!(openai_response.choices[0].message.role, "assistant");
        assert_eq!(
            openai_response.choices[0].message.content.as_ref().unwrap(),
            "This is a truncated response because"
        );
        assert_eq!(openai_response.choices[0].finish_reason, "length");
    }
}
