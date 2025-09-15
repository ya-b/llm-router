use crate::converters::anthropic::{
    AnthropicContent, AnthropicContentObject, AnthropicRequest, AnthropicSystemContent,
    AnthropicSystemContentObject,
};
use crate::converters::openai::{
    OpenAIContent, OpenAIContentItem, OpenAIFunction, OpenAIImageUrl, OpenAIMessage, OpenAITool,
    OpenAIToolCall, OpenAIToolCallFunction,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl From<AnthropicRequest> for OpenAIRequest {
    fn from(anthropic_request: AnthropicRequest) -> Self {
        let mut messages = Vec::new();

        // 添加系统消息
        if let Some(system) = anthropic_request.system {
            let system_content = match system {
                AnthropicSystemContent::Text(text) => OpenAIContent::Text(text),
                AnthropicSystemContent::Array(arr) => {
                    let items: Vec<OpenAIContentItem> = arr
                        .into_iter()
                        .filter_map(|obj| match obj {
                            AnthropicSystemContentObject::Text { text } => Some(OpenAIContentItem {
                                r#type: "text".to_string(),
                                text: Some(text),
                                image_url: None,
                            }),
                        })
                        .collect();
                    OpenAIContent::Array(items)
                }
            };

            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: system_content,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        // 转换普通消息
        if let Some(anthropic_messages) = anthropic_request.messages {
            for message in anthropic_messages {
                let mut content_items = Vec::new();
                let mut tool_calls = Vec::new();
                let mut text_content = String::new();

                match &message.content {
                    AnthropicContent::Text(text) => {
                        text_content.push_str(&text);
                    }
                    AnthropicContent::Array(array) => {
                        for item in array.iter() {
                            match item {
                                AnthropicContentObject::Text { text } => {
                                    content_items.push(OpenAIContentItem {
                                        r#type: "text".to_string(),
                                        text: Some(text.clone()),
                                        image_url: None,
                                    });
                                }
                                AnthropicContentObject::Thinking {
                                    thinking: _,
                                    signature: _,
                                } => {}
                                AnthropicContentObject::RedactedThinking { data: _ } => {}
                                AnthropicContentObject::Image { source } => {
                                    let image_url = if &source.r#type == "base64" {
                                        format!(
                                            "data:{:?};base64,{:?}",
                                            source.media_type, source.data
                                        )
                                    } else {
                                        source.url.clone().expect("url error")
                                    };
                                    content_items.push(OpenAIContentItem {
                                        r#type: "image_url".to_string(),
                                        text: None,
                                        image_url: Some(OpenAIImageUrl { url: image_url }),
                                    });
                                }
                                AnthropicContentObject::ToolUse { id, name, input } => {
                                    tool_calls.push(OpenAIToolCall {
                                        id: id.clone(),
                                        r#type: "function".to_string(),
                                        function: OpenAIToolCallFunction {
                                            name: name.clone(),
                                            arguments: serde_json::to_string(&input)
                                                .unwrap_or_else(|_| "{}".to_string()),
                                        },
                                    });
                                }
                                AnthropicContentObject::ToolResult {
                                    tool_use_id,
                                    content,
                                } => {
                                    messages.push(OpenAIMessage {
                                        role: "tool".to_string(),
                                        content: OpenAIContent::Text(content.clone()),
                                        tool_calls: None,
                                        tool_call_id: Some(tool_use_id.clone()),
                                        reasoning_content: None,
                                    });
                                }
                            }
                        }
                    }
                };

                if !content_items.is_empty() {
                    messages.push(OpenAIMessage {
                        role: message.role.clone(),
                        content: OpenAIContent::Array(content_items),
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                } else {
                    messages.push(OpenAIMessage {
                        role: message.role,
                        content: OpenAIContent::Text(text_content),
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                }
            }
        }

        let openai_request = OpenAIRequest {
            model: anthropic_request.model,
            messages,
            max_tokens: Some(anthropic_request.max_tokens),
            temperature: None,
            tools: anthropic_request.tools.map(|tools| {
                tools
                    .into_iter()
                    .map(|tool| OpenAITool {
                        r#type: "function".to_string(),
                        function: OpenAIFunction {
                            name: tool.name,
                            description: tool.description,
                            parameters: tool.input_schema,
                        },
                    })
                    .collect()
            }),
            stream: anthropic_request.stream,
            extra_fields: anthropic_request.extra_fields,
        };

        openai_request
    }
}
