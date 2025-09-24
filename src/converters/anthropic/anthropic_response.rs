use crate::converters::anthropic::{AnthropicContentObject, AnthropicUsage};
use crate::converters::helpers;
use crate::converters::openai::OpenAIResponse;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContentObject>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
}

impl From<OpenAIResponse> for AnthropicResponse {
    fn from(openai_resp: OpenAIResponse) -> Self {
        let mut content_objects = Vec::new();

        if let Some(reasoning_content) = &openai_resp.choices[0].message.reasoning_content {
            if !reasoning_content.trim().is_empty() {
                content_objects.push(AnthropicContentObject::Thinking {
                    thinking: reasoning_content.clone(),
                    signature: None,
                });
            }
        }

        if let Some(content) = &openai_resp.choices[0].message.content {
            if !content.trim().is_empty() {
                content_objects.push(AnthropicContentObject::Text {
                    text: content.clone(),
                });
            }
        }

        if let Some(tool_calls) = &openai_resp.choices[0].message.tool_calls {
            for tool_call in tool_calls {
                let input = serde_json::from_str(&tool_call.function.arguments)
                    .unwrap_or_else(|_| serde_json::json!({}));
                content_objects.push(AnthropicContentObject::ToolUse {
                    id: tool_call.id.clone(),
                    name: tool_call.function.name.clone(),
                    input,
                });
            }
        }

        AnthropicResponse {
            id: openai_resp.id,
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: content_objects,
            model: openai_resp.model.clone(),
            stop_sequence: None,
            stop_reason: Some(
                helpers::map_openai_finish_reason_to_anthropic(&Value::String(
                    openai_resp.choices[0].finish_reason.clone(),
                ))
                .as_str()
                .unwrap_or("end_turn")
                .to_string(),
            ),
            usage: openai_resp.usage.map(|usage| AnthropicUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_openai_to_anthropic_response() {
        // 测试基本的文本响应
        let json_response = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, how can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        });

        let openai_response: OpenAIResponse = serde_json::from_value(json_response).expect("error");

        let anthropic_response: AnthropicResponse = openai_response.into();

        assert_eq!(anthropic_response.id, "chatcmpl-123");
        assert_eq!(anthropic_response.r#type, "message");
        assert_eq!(anthropic_response.role, "assistant");
        if let AnthropicContentObject::Text { text } = &anthropic_response.content[0] {
            assert_eq!(text, "Hello, how can I help you today?");
        } else {
            panic!("Expected AnthropicContentObject::Text");
        }
        assert_eq!(anthropic_response.stop_reason.unwrap(), "end_turn");
        if let Some(usage) = &anthropic_response.usage {
            assert_eq!(usage.input_tokens, 9);
            assert_eq!(usage.output_tokens, 12);
        } else {
            panic!("Expected AnthropicUsage::usage");
        }
    }

    #[test]
    fn test_openai_to_anthropic_response_with_reasoning() {
        // 测试包含推理内容的响应
        let json_response = json!({
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "I need to think about this step by step.",
                        "content": "The answer is 42."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "total_tokens": 35
            }
        });

        let openai_response: OpenAIResponse = serde_json::from_value(json_response).expect("error");

        let anthropic_response: AnthropicResponse = openai_response.into();

        assert_eq!(anthropic_response.id, "chatcmpl-456");
        assert_eq!(anthropic_response.r#type, "message");
        assert_eq!(anthropic_response.role, "assistant");
        if let AnthropicContentObject::Thinking {
            thinking,
            signature: _,
        } = &anthropic_response.content[0]
        {
            assert_eq!(thinking, "I need to think about this step by step.");
        } else {
            panic!("Expected AnthropicContentObject::Text");
        }
        if let AnthropicContentObject::Text { text } = &anthropic_response.content[1] {
            assert_eq!(text, "The answer is 42.");
        } else {
            panic!("Expected AnthropicContentObject::Text");
        }
        assert_eq!(anthropic_response.stop_reason.unwrap(), "end_turn");
        if let Some(usage) = &anthropic_response.usage {
            assert_eq!(usage.input_tokens, 15);
            assert_eq!(usage.output_tokens, 20);
        } else {
            panic!("Expected AnthropicUsage::usage");
        }
    }

    #[test]
    fn test_openai_to_anthropic_response_with_tool_calls() {
        // 测试包含工具调用的响应
        let json_response = json!({
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'll help you get the weather.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\": \"San Francisco, CA\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 30,
                "total_tokens": 55
            }
        });

        let openai_response: OpenAIResponse = serde_json::from_value(json_response).expect("error");

        let anthropic_response: AnthropicResponse = openai_response.into();

        assert_eq!(anthropic_response.id, "chatcmpl-789");
        assert_eq!(anthropic_response.r#type, "message");
        assert_eq!(anthropic_response.role, "assistant");
        if let AnthropicContentObject::Text { text } = &anthropic_response.content[0] {
            assert_eq!(text, "I'll help you get the weather.");
        } else {
            panic!("Expected AnthropicContentObject::Text");
        }
        if let AnthropicContentObject::ToolUse { id, name, input } = &anthropic_response.content[1]
        {
            assert_eq!(id, "call_abc123");
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "San Francisco, CA");
        } else {
            panic!("Expected AnthropicContentObject::Text");
        }
        assert_eq!(anthropic_response.stop_reason.unwrap(), "tool_use");
        if let Some(usage) = &anthropic_response.usage {
            assert_eq!(usage.input_tokens, 25);
            assert_eq!(usage.output_tokens, 30);
        } else {
            panic!("Expected AnthropicUsage::usage");
        }
    }

    #[test]
    fn test_openai_to_anthropic_response_empty_content() {
        // 测试空内容的响应
        let json_response = json!({
            "id": "chatcmpl-empty",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null
                    },
                    "finish_reason": "stop"
                }
            ]
        });

        let openai_response: OpenAIResponse = serde_json::from_value(json_response).expect("error");

        let anthropic_response: AnthropicResponse = openai_response.into();

        assert_eq!(anthropic_response.id, "chatcmpl-empty");
        assert_eq!(anthropic_response.r#type, "message");
        assert_eq!(anthropic_response.role, "assistant");
        // 空内容不应该被添加到 content 数组中
        assert!(anthropic_response.content.is_empty());
        assert_eq!(anthropic_response.stop_reason.unwrap(), "end_turn");
    }

    #[test]
    fn test_openai_to_anthropic_response_max_tokens_finish_reason() {
        // 测试 max_tokens 停止原因
        let json_response = json!({
            "id": "chatcmpl-max",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a truncated response because"
                    },
                    "finish_reason": "length"
                }
            ]
        });

        let openai_response: OpenAIResponse = serde_json::from_value(json_response).expect("error");

        let anthropic_response: AnthropicResponse = openai_response.into();

        assert_eq!(anthropic_response.id, "chatcmpl-max");
        assert_eq!(anthropic_response.r#type, "message");
        assert_eq!(anthropic_response.role, "assistant");
        if let AnthropicContentObject::Text { text } = &anthropic_response.content[0] {
            assert_eq!(text, "This is a truncated response because");
        } else {
            panic!("Expected AnthropicContentObject::Text");
        }
        assert_eq!(anthropic_response.stop_reason.unwrap(), "max_tokens");
    }
}
