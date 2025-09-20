use crate::converters::anthropic::{
    AnthropicContent, AnthropicContentObject, AnthropicImageSource, AnthropicMessage,
    AnthropicMetadata, AnthropicSystemContent, AnthropicTool,
};
use crate::converters::openai::{OpenAIContent, OpenAIRequest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<AnthropicMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

// 转换实现
impl From<OpenAIRequest> for AnthropicRequest {
    fn from(openai_request: OpenAIRequest) -> Self {
        let mut anthropic_request = AnthropicRequest {
            model: openai_request.model,
            max_tokens: openai_request.max_tokens.unwrap_or(4096),
            messages: None,
            system: None,
            tools: None,
            metadata: None,
            stream: openai_request.stream,
            temperature: openai_request.temperature,
            extra_fields: std::collections::HashMap::new(),
        };

        // 处理消息
        let mut messages = Vec::new();
        let mut system_message: Option<AnthropicSystemContent> = None;

        for message in openai_request.messages {
            if message.role == "system" {
                if let OpenAIContent::Text(text) = message.content {
                    system_message = Some(AnthropicSystemContent::Text(text));
                }
            } else {
                let mut content = Vec::new();

                let text_for_tool_result = match &message.content {
                    OpenAIContent::Text(text) => {
                        content.push(AnthropicContentObject::Text { text: text.clone() });
                        text.clone()
                    }
                    OpenAIContent::Array(array) => {
                        // 这里需要处理多模态内容
                        for item in array.iter() {
                            match item.r#type.as_str() {
                                "text" => {
                                    if let Some(text) = &item.text {
                                        content.push(AnthropicContentObject::Text {
                                            text: text.clone(),
                                        });
                                    }
                                }
                                "image_url" => {
                                    if let Some(image_url) = &item.image_url {
                                        // 处理 base64 图片
                                        if image_url.url.starts_with("data:") {
                                            let parts: Vec<&str> =
                                                image_url.url.split(',').collect();
                                            if parts.len() == 2 {
                                                let media_type = parts[0]
                                                    .replace("data:", "")
                                                    .replace(";base64", "");
                                                content.push(AnthropicContentObject::Image {
                                                    source: AnthropicImageSource {
                                                        r#type: "base64".to_string(),
                                                        media_type: Some(media_type),
                                                        data: Some(parts[1].to_string()),
                                                        url: None,
                                                    },
                                                });
                                            }
                                        } else if image_url.url.starts_with("http") {
                                            content.push(AnthropicContentObject::Image {
                                                source: AnthropicImageSource {
                                                    r#type: "url".to_string(),
                                                    media_type: None,
                                                    data: None,
                                                    url: Some(image_url.url.clone()),
                                                },
                                            });
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        String::new()
                    }
                };

                // 处理工具调用结果
                if let Some(tool_call_id) = message.tool_call_id {
                    content.push(AnthropicContentObject::ToolResult {
                        tool_use_id: tool_call_id,
                        content: text_for_tool_result,
                    });
                }

                messages.push(AnthropicMessage {
                    role: if message.role == "assistant" {
                        "assistant"
                    } else {
                        "user"
                    }
                    .to_string(),
                    content: AnthropicContent::Array(content),
                });
            }
        }

        if !messages.is_empty() {
            anthropic_request.messages = Some(messages);
        }
        anthropic_request.system = system_message;

        // 处理工具调用
        if let Some(tools) = openai_request.tools {
            let anthropic_tools = tools
                .into_iter()
                .map(|tool| AnthropicTool {
                    name: tool.function.name,
                    description: tool.function.description,
                    input_schema: tool.function.parameters,
                })
                .collect();
            anthropic_request.tools = Some(anthropic_tools);
        }

        // 复制额外字段
        for (key, value) in openai_request.extra_fields {
            anthropic_request.extra_fields.insert(key, value);
        }

        anthropic_request
    }
}
