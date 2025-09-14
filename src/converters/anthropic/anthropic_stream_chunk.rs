use crate::converters::helpers;
use crate::converters::{
    anthropic::{
        AnthropicContentBlock, AnthropicMessageDelta, AnthropicStreamDelta, AnthropicStreamMessage,
        AnthropicUsage,
    },
    openai::OpenAIStreamChunk,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamChunk {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicStreamMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: i32,
        content_block: AnthropicContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: i32,
        delta: AnthropicStreamDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: i32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: Option<AnthropicUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
}

impl AnthropicStreamChunk {

    pub fn stream_type(&self) -> &'static str {
        match self {
            AnthropicStreamChunk::MessageStart { .. } => "message_start",
            AnthropicStreamChunk::ContentBlockStart { .. } => "content_block_start",
            AnthropicStreamChunk::ContentBlockDelta { .. } => "content_block_delta",
            AnthropicStreamChunk::ContentBlockStop { .. } => "content_block_stop",
            AnthropicStreamChunk::MessageDelta { .. } => "message_delta",
            AnthropicStreamChunk::MessageStop => "message_stop",
            AnthropicStreamChunk::Ping => "ping",
        }
    }
}

impl From<OpenAIStreamChunk> for AnthropicStreamChunk {
    fn from(openai_chunk: OpenAIStreamChunk) -> Self {
        // 如果没有 choices，返回 ping 心跳包
        if openai_chunk
            .choices
            .as_ref()
            .map(|v| v.is_empty())
            .unwrap_or(true)
        {
            return AnthropicStreamChunk::Ping;
        }

        let first_choice = match openai_chunk
            .choices
            .as_ref()
            .and_then(|v| v.first())
        {
            Some(c) => c,
            None => return AnthropicStreamChunk::Ping,
        };

        // 处理使用统计
        let usage = openai_chunk.usage.map(|usage| AnthropicUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        });

        // 处理内容增量
        let delta = &first_choice.delta;

        // 处理推理内容
        if let Some(reasoning_content) = delta
            .as_ref()
            .and_then(|d| d.reasoning_content.as_ref())
        {
            if !reasoning_content.is_empty() {
                return AnthropicStreamChunk::ContentBlockDelta {
                    index: 0,
                    delta: AnthropicStreamDelta::ThinkingDelta {
                        thinking: reasoning_content.clone(),
                    },
                };
            }
        }

        // 处理文本内容
        if let Some(text_content) = delta.as_ref().and_then(|d| d.content.as_ref()) {
            if !text_content.is_empty() {
                return AnthropicStreamChunk::ContentBlockDelta {
                    index: 0,
                    delta: AnthropicStreamDelta::TextDelta {
                        text: text_content.clone(),
                    },
                };
            }
        }

        // 处理工具调用增量
        if let Some(tool_calls) = delta.as_ref().and_then(|d| d.tool_calls.as_ref()) {
            for tool_call in tool_calls.iter() {
                if let Some(function) = tool_call.function.as_ref() {
                    return AnthropicStreamChunk::ContentBlockDelta {
                        index: 0,
                        delta: AnthropicStreamDelta::InputJsonDelta {
                            partial_json: function.arguments.clone(),
                            name: function.name.clone(),
                            id: tool_call.id.clone(),
                        },
                    };
                }
            }
        }

        // 检查是否是停止消息
        if let Some(finish_reason) = &first_choice.finish_reason {
            if !finish_reason.is_empty() {
                return AnthropicStreamChunk::MessageDelta {
                    delta: AnthropicMessageDelta {
                        stop_reason: Some(
                            helpers::map_openai_finish_reason_to_anthropic(&Value::String(
                                finish_reason.clone(),
                            ))
                            .as_str()
                            .unwrap_or("end_turn")
                            .to_string(),
                        ),
                    },
                    usage,
                };
            }
        }

        // 默认返回 ping 心跳包
        AnthropicStreamChunk::Ping
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use super::*;

    fn openai_to_anthropic_stream_chunk(chunk: &Value) -> Value {
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(chunk.clone()).unwrap();
        let anthropic_stream_chunk: AnthropicStreamChunk = openai_chunk.into();
        serde_json::to_value(anthropic_stream_chunk).unwrap_or_default()
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunk_with_text_content() {
        // 测试包含文本内容的流式响应块
        let openai_chunk = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": null
                }
            ]
        });

        let anthropic_chunk = openai_to_anthropic_stream_chunk(&openai_chunk);
        
        assert_eq!(anthropic_chunk["type"], "content_block_delta");
        assert_eq!(anthropic_chunk["delta"]["type"], "text_delta");
        assert_eq!(anthropic_chunk["delta"]["text"], "Hello");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunk_with_reasoning_content() {
        // 测试包含推理内容的流式响应块
        let openai_chunk = json!({
            "id": "chatcmpl-456",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "I need to think about this."
                    },
                    "finish_reason": null
                }
            ]
        });

        let anthropic_chunk = openai_to_anthropic_stream_chunk(&openai_chunk);
        
        assert_eq!(anthropic_chunk["type"], "content_block_delta");
        assert_eq!(anthropic_chunk["delta"]["type"], "thinking_delta");
        assert_eq!(anthropic_chunk["delta"]["thinking"], "I need to think about this.");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunk_with_tool_calls() {
        // 测试包含工具调用的流式响应块
        let openai_chunk = json!({
            "id": "chatcmpl-789",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\": \"San Francisco\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": null
                }
            ]
        });

        let anthropic_chunk = openai_to_anthropic_stream_chunk(&openai_chunk);
        
        assert_eq!(anthropic_chunk["type"], "content_block_delta");
        assert_eq!(anthropic_chunk["delta"]["type"], "input_json_delta");
        assert_eq!(anthropic_chunk["delta"]["name"], "get_weather");
        assert_eq!(anthropic_chunk["delta"]["id"], "call_abc123");
        assert_eq!(anthropic_chunk["delta"]["partial_json"], "{\"location\": \"San Francisco\"}");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunk_with_finish_reason() {
        // 测试包含完成原因的流式响应块
        let openai_chunk = json!({
            "id": "chatcmpl-finish",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        });

        let anthropic_chunk = openai_to_anthropic_stream_chunk(&openai_chunk);
        
        assert_eq!(anthropic_chunk["type"], "message_delta");
        assert_eq!(anthropic_chunk["delta"]["stop_reason"], "end_turn");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunk_with_usage() {
        // 测试包含使用统计的流式响应块
        let openai_chunk = json!({
            "id": "chatcmpl-usage",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Hello"
                    },
                    "finish_reason": null
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let anthropic_chunk = openai_to_anthropic_stream_chunk(&openai_chunk);
        
        assert_eq!(anthropic_chunk["type"], "content_block_delta");
        assert_eq!(anthropic_chunk["delta"]["type"], "text_delta");
        assert_eq!(anthropic_chunk["delta"]["text"], "Hello");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunk_empty_delta() {
        // 测试空增量的流式响应块
        let openai_chunk = json!({
            "id": "chatcmpl-empty",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": null
                }
            ]
        });

        let anthropic_chunk = openai_to_anthropic_stream_chunk(&openai_chunk);
        
        // 空增量应该返回 ping 心跳包
        assert_eq!(anthropic_chunk["type"], "ping");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunk_no_choices() {
        // 测试没有 choices 的流式响应块
        let openai_chunk = json!({
            "id": "chatcmpl-nochoices",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4"
        });

        let anthropic_chunk = openai_to_anthropic_stream_chunk(&openai_chunk);
        
        // 没有 choices 应该返回 ping 心跳包
        assert_eq!(anthropic_chunk["type"], "ping");
    }
}
