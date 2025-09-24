use crate::converters::anthropic::{
    AnthropicContentBlock, AnthropicStreamChunk, AnthropicStreamDelta,
};
use crate::converters::gemini::{
    GeminiCandidate, GeminiFinishReason, GeminiPart, GeminiStreamChunk,
};
use crate::converters::helpers;
use crate::converters::openai::{
    OpenAIStreamChoice, OpenAIStreamDelta, OpenAIStreamToolCall, OpenAIStreamToolCallFunction,
    OpenAIUsage,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: Option<String>,
    pub created: u64,
    pub model: String,
    pub choices: Option<Vec<OpenAIStreamChoice>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

impl From<AnthropicStreamChunk> for OpenAIStreamChunk {
    fn from(anthropic_chunk: AnthropicStreamChunk) -> Self {
        let id = match &anthropic_chunk {
            AnthropicStreamChunk::MessageStart { message } => message.id.clone(),
            AnthropicStreamChunk::ContentBlockStart { .. } => "chatcmpl-default".to_string(),
            AnthropicStreamChunk::ContentBlockDelta { .. } => "chatcmpl-default".to_string(),
            AnthropicStreamChunk::ContentBlockStop { .. } => "chatcmpl-default".to_string(),
            AnthropicStreamChunk::MessageDelta { .. } => "chatcmpl-default".to_string(),
            AnthropicStreamChunk::MessageStop => "chatcmpl-default".to_string(),
            AnthropicStreamChunk::Ping => "chatcmpl-default".to_string(),
        };

        let mut delta = OpenAIStreamDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };

        let mut finish_reason = None;
        let mut usage = None;

        // 根据 chunk 类型处理
        match anthropic_chunk {
            AnthropicStreamChunk::MessageStart { message: _ } => {
                // 消息开始，设置角色
                delta.role = Some("assistant".to_string());
                delta.content = Some("".to_string());
                delta.reasoning_content = Some("".to_string());
            }
            AnthropicStreamChunk::ContentBlockStart {
                index: _,
                content_block,
            } => {
                // 内容块开始
                match content_block {
                    AnthropicContentBlock::Text { text: _ } => {
                        // 文本块开始，通常不需要特殊处理
                        delta.content = Some("".to_string());
                    }
                    AnthropicContentBlock::Thinking { .. } => {
                        // 推理块开始
                        delta.reasoning_content = Some("".to_string());
                    }
                    AnthropicContentBlock::ToolUse { id, name, input: _ } => {
                        // 工具使用块开始
                        delta.tool_calls = Some(vec![OpenAIStreamToolCall {
                            index: 0,
                            id: Some(id),
                            r#type: Some("function".to_string()),
                            function: Some(OpenAIStreamToolCallFunction {
                                name: Some(name),
                                arguments: Some("".to_string()),
                            }),
                        }]);
                    }
                }
            }
            AnthropicStreamChunk::ContentBlockDelta {
                index: _,
                delta: chunk_delta,
            } => {
                // 处理内容块增量
                match chunk_delta {
                    AnthropicStreamDelta::ThinkingDelta { thinking } => {
                        delta.reasoning_content = Some(thinking);
                    }
                    AnthropicStreamDelta::TextDelta { text } => {
                        delta.content = Some(text);
                    }
                    AnthropicStreamDelta::InputJsonDelta {
                        partial_json,
                        name,
                        id,
                    } => {
                        // 工具参数增量
                        delta.tool_calls = Some(vec![OpenAIStreamToolCall {
                            index: 0,
                            id,
                            r#type: Some("function".to_string()),
                            function: Some(OpenAIStreamToolCallFunction {
                                name,
                                arguments: partial_json,
                            }),
                        }]);
                    }
                }
            }
            AnthropicStreamChunk::ContentBlockStop { .. } => {
                // 内容块结束，通常不需要特殊处理
                delta = OpenAIStreamDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                };
            }
            AnthropicStreamChunk::MessageDelta {
                delta: chunk_delta,
                usage: chunk_usage,
            } => {
                // 处理消息级增量，主要是停止原因
                if let Some(stop_reason) = chunk_delta.stop_reason {
                    finish_reason = Some(
                        helpers::map_anthropic_stop_reason_to_openai(Some(&Value::String(
                            stop_reason,
                        )))
                        .as_str()
                        .unwrap_or("stop")
                        .to_string(),
                    );
                }
                usage = chunk_usage.map(|u| OpenAIUsage {
                    prompt_tokens: u.input_tokens,
                    completion_tokens: u.output_tokens,
                    total_tokens: u.input_tokens + u.output_tokens,
                    completion_tokens_details: None,
                    prompt_tokens_details: None,
                });
            }
            AnthropicStreamChunk::MessageStop => {
                // 消息结束
                finish_reason = Some("stop".to_string());
                delta = OpenAIStreamDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                };
            }
            AnthropicStreamChunk::Ping => {
                // 心跳包，返回空的增量
                delta = OpenAIStreamDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                };
            }
        }

        // 构建 OpenAI 流式响应块
        OpenAIStreamChunk {
            id,
            object: Some("chat.completion.chunk".to_string()),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            model: "claude-3-opus".to_string(), // 默认模型，实际应该从请求中获取
            choices: Some(vec![OpenAIStreamChoice {
                index: 0,
                delta: Some(delta),
                finish_reason,
            }]),
            usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_start() {
        // 测试 message_start 类型的 Anthropic 流式响应块
        let json_chunk = json!({
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-opus-20240229"
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(openai_chunk["choices"][0]["delta"]["content"], "");
        assert_eq!(openai_chunk["choices"][0]["delta"]["reasoning_content"], "");
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
        // 验证 ID 被正确设置
        assert!(openai_chunk["id"].as_str().unwrap().contains("msg_123"));
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_start_text() {
        // 测试 content_block_start 类型的 Anthropic 流式响应块（文本内容）
        let json_chunk = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"]["content"], "");
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_start_tool_use() {
        // 测试 content_block_start 类型的 Anthropic 流式响应块（工具调用）
        let json_chunk = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "tool_123",
                "name": "get_weather",
                "input": {}
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["tool_calls"][0]["id"],
            "tool_123"
        );
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["tool_calls"][0]["type"],
            "function"
        );
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            ""
        );
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_delta_thinking() {
        // 测试 content_block_delta 类型的 Anthropic 流式响应块（推理内容）
        let json_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "I need to think about this step by step."
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["reasoning_content"],
            "I need to think about this step by step."
        );
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_delta_text() {
        // 测试 content_block_delta 类型的 Anthropic 流式响应块（文本内容）
        let json_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello, how can I help you today?"
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["content"],
            "Hello, how can I help you today?"
        );
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_delta_tool_use() {
        // 测试 content_block_delta 类型的 Anthropic 流式响应块（工具调用参数）
        let json_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": "{\"location\": \"San Francisco, CA\"}"
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            openai_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "{\"location\": \"San Francisco, CA\"}"
        );
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_stop() {
        // 测试 content_block_stop 类型的 Anthropic 流式响应块
        let json_chunk = json!({
            "type": "content_block_stop",
            "index": 0
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_delta() {
        // 测试 message_delta 类型的 Anthropic 流式响应块
        let json_chunk = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn"
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_stop() {
        // 测试 message_stop 类型的 Anthropic 流式响应块
        let json_chunk = json!({
            "type": "message_stop"
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_ping() {
        // 测试 ping 类型的 Anthropic 流式响应块
        let json_chunk = json!({
            "type": "ping"
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_delta_max_tokens() {
        // 测试 message_delta 类型的 Anthropic 流式响应块（max_tokens 停止原因）
        let json_chunk = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "max_tokens"
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_delta_tool_use() {
        // 测试 message_delta 类型的 Anthropic 流式响应块（tool_use 停止原因）
        let json_chunk = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use"
            }
        });

        let anthropic_chunk: AnthropicStreamChunk =
            serde_json::from_value(json_chunk.clone()).unwrap();
        let openai_chunk_struct: OpenAIStreamChunk = anthropic_chunk.into();
        let openai_chunk = serde_json::to_value(openai_chunk_struct.clone()).unwrap_or_default();

        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "tool_calls");
    }
}

impl From<GeminiStreamChunk> for OpenAIStreamChunk {
    fn from(gemini_chunk: GeminiStreamChunk) -> Self {
        // Map id and model from Gemini if present
        let id = gemini_chunk
            .response_id
            .unwrap_or_else(|| "chatcmpl-default".to_string());
        let model = gemini_chunk
            .model_version
            .unwrap_or_else(|| "gemini-1.5-pro".to_string());

        // Map candidates to OpenAI choices
        let choices: Vec<OpenAIStreamChoice> = gemini_chunk
            .candidates
            .into_iter()
            .enumerate()
            .map(|(idx, cand)| map_gemini_candidate_to_openai_choice(cand, idx as i32))
            .collect();

        // Map usage if available
        let usage = gemini_chunk.usage_metadata.map(|u| OpenAIUsage {
            prompt_tokens: u.prompt_token_count.unwrap_or(0),
            completion_tokens: u.candidates_token_count.unwrap_or(0),
            total_tokens: u.total_token_count.unwrap_or(
                u.prompt_token_count.unwrap_or(0) + u.candidates_token_count.unwrap_or(0),
            ),
            completion_tokens_details: None,
            prompt_tokens_details: None,
        });

        OpenAIStreamChunk {
            id,
            object: Some("chat.completion.chunk".to_string()),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            model,
            choices: Some(choices),
            usage,
        }
    }
}

fn map_gemini_candidate_to_openai_choice(
    candidate: GeminiCandidate,
    index: i32,
) -> OpenAIStreamChoice {
    // Aggregate delta content from parts
    let mut role: Option<String> = None;
    let mut content_acc = String::new();
    let mut reasoning_acc = String::new();
    let mut tool_calls: Vec<OpenAIStreamToolCall> = Vec::new();

    if let Some(r) = candidate.content.role {
        // Gemini uses "model" for assistant
        role = Some(if r == "model" {
            "assistant".to_string()
        } else {
            r
        });
    }

    for part in candidate.content.parts.into_iter() {
        match part {
            GeminiPart::Text { text, thought, .. } => {
                if thought.unwrap_or(false) {
                    reasoning_acc.push_str(&text);
                } else {
                    content_acc.push_str(&text);
                }
            }
            GeminiPart::FunctionCall { function_call, .. } => {
                // Map to OpenAI tool call
                let args_str = match serde_json::to_string(&function_call.args) {
                    Ok(s) => s,
                    Err(_) => String::new(),
                };
                let idx = tool_calls.len() as i32;
                tool_calls.push(OpenAIStreamToolCall {
                    index: idx,
                    id: None,
                    r#type: Some("function".to_string()),
                    function: Some(OpenAIStreamToolCallFunction {
                        name: Some(function_call.name),
                        arguments: Some(args_str),
                    }),
                });
            }
            // FunctionResponse and InlineData don't have a direct delta mapping here; ignore
            _ => {}
        }
    }

    let delta = OpenAIStreamDelta {
        role,
        content: if content_acc.is_empty() {
            None
        } else {
            Some(content_acc)
        },
        reasoning_content: if reasoning_acc.is_empty() {
            None
        } else {
            Some(reasoning_acc)
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    };

    let finish_reason = candidate
        .finish_reason
        .and_then(map_gemini_finish_reason_to_openai);

    OpenAIStreamChoice {
        index,
        delta: Some(delta),
        finish_reason,
    }
}

fn map_gemini_finish_reason_to_openai(fr: GeminiFinishReason) -> Option<String> {
    use GeminiFinishReason as GFR;
    let s = match fr {
        GFR::Stop => "stop",
        GFR::MaxTokens => "length",
        // Tool-related
        GFR::UnexpectedToolCall | GFR::TooManyToolCalls => "tool_calls",
        // Safety/content filter related
        GFR::Safety | GFR::Blocklist | GFR::ProhibitedContent | GFR::ImageSafety | GFR::Spii => {
            "content_filter"
        }
        // Others map to unspecified; do not set
        _ => return None,
    };
    Some(s.to_string())
}
