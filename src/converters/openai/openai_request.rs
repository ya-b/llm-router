use crate::converters::anthropic::{
    AnthropicContent, AnthropicContentObject, AnthropicRequest, AnthropicSystemContent,
    AnthropicSystemContentObject,
};
use crate::converters::gemini::{GeminiPart, GeminiRequest};
use crate::converters::openai::{
    OpenAIContent, OpenAIContentItem, OpenAIFunction, OpenAIImageUrl, OpenAIMessage, OpenAITool,
    OpenAIToolCall, OpenAIToolCallFunction,
};
use crate::converters::responses::{
    ResponsesContentPart, ResponsesInput, ResponsesMessage, ResponsesMessageContent,
    ResponsesReasoningSummary, ResponsesRequest, ResponsesTextConfig, ResponsesTextFormat,
    ResponsesTextFormatJsonSchema, ResponsesTool,
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
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    pub r#type: String, // e.g. "json_object" or "json_schema"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<OpenAIJSONSchemaSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIJSONSchemaSpec {
    pub name: String,
    pub schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
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
                            AnthropicSystemContentObject::Text { text } => {
                                Some(OpenAIContentItem {
                                    r#type: "text".to_string(),
                                    text: Some(text),
                                    image_url: None,
                                })
                            }
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
            response_format: None,
            tools: anthropic_request.tools.map(|tools| {
                tools
                    .into_iter()
                    .map(|tool| OpenAITool {
                        r#type: "function".to_string(),
                        function: OpenAIFunction {
                            name: tool.name,
                            description: tool.description,
                            parameters: tool.input_schema,
                            strict: None,
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

impl From<GeminiRequest> for OpenAIRequest {
    fn from(g: GeminiRequest) -> Self {
        let mut messages = Vec::new();

        if let Some(sys) = g.system_instruction {
            // Flatten system instruction text
            let mut text = String::new();
            for p in sys.parts.iter() {
                if let GeminiPart::Text { text: t, .. } = p {
                    text.push_str(&t);
                }
            }
            if !text.is_empty() {
                messages.push(crate::converters::openai::OpenAIMessage {
                    role: "system".to_string(),
                    content: OpenAIContent::Text(text),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
        }

        for c in g.contents.into_iter() {
            let role = match c.role.as_deref() {
                Some("model") => "assistant",
                _ => "user",
            };
            let mut text = String::new();
            for p in c.parts.into_iter() {
                if let GeminiPart::Text { text: t, .. } = p {
                    text.push_str(&t);
                }
            }
            messages.push(crate::converters::openai::OpenAIMessage {
                role: role.to_string(),
                content: OpenAIContent::Text(text),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        // Structured output mapping from Gemini generationConfig -> OpenAI
        let (resp_mime, resp_schema) = if let Some(gc) = &g.generation_config {
            (gc.response_mime_type.as_deref(), gc.response_schema.clone())
        } else {
            (None, None)
        };
        let response_format = match (resp_mime, resp_schema) {
            (Some(mime), Some(schema)) if mime.eq_ignore_ascii_case("application/json") => {
                Some(OpenAIResponseFormat {
                    r#type: "json_schema".to_string(),
                    json_schema: Some(OpenAIJSONSchemaSpec {
                        name: "structured_output".to_string(),
                        schema,
                        strict: Some(true),
                    }),
                })
            }
            (Some(mime), None) if mime.eq_ignore_ascii_case("application/json") => {
                Some(OpenAIResponseFormat {
                    r#type: "json_object".to_string(),
                    json_schema: None,
                })
            }
            _ => None,
        };

        OpenAIRequest {
            model: g.model,
            messages,
            max_tokens: g
                .generation_config
                .as_ref()
                .and_then(|gc| gc.max_output_tokens),
            temperature: g.generation_config.as_ref().and_then(|gc| gc.temperature),
            response_format,
            tools: None,
            stream: g.stream,
            extra_fields: g.extra_fields,
        }
    }
}

impl From<ResponsesRequest> for OpenAIRequest {
    fn from(responses: ResponsesRequest) -> Self {
        let ResponsesRequest {
            model,
            input,
            include: _,
            instructions,
            max_output_tokens,
            metadata: _,
            parallel_tool_calls: _,
            previous_response_id: _,
            reasoning: _,
            service_tier: _,
            store: _,
            stream,
            temperature,
            text,
            tool_choice: _,
            tools,
            top_p: _,
            truncation: _,
            user: _,
            extra_fields,
        } = responses;

        let mut messages = Vec::new();

        if let Some(instr) = instructions.filter(|s| !s.trim().is_empty()) {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: OpenAIContent::Text(instr),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        messages.extend(convert_responses_input_to_openai_messages(input));

        let response_format = responses_text_config_to_response_format(text);

        let tool_list = tools.and_then(|ts| {
            let converted: Vec<OpenAITool> = ts
                .into_iter()
                .filter_map(responses_tool_to_openai_tool)
                .collect();
            if converted.is_empty() {
                None
            } else {
                Some(converted)
            }
        });

        OpenAIRequest {
            model,
            messages,
            max_tokens: max_output_tokens,
            temperature,
            response_format,
            tools: tool_list,
            stream,
            extra_fields,
        }
    }
}

fn convert_responses_input_to_openai_messages(input: ResponsesInput) -> Vec<OpenAIMessage> {
    match input {
        ResponsesInput::Text(text) => vec![OpenAIMessage {
            role: "user".to_string(),
            content: OpenAIContent::Text(text),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        ResponsesInput::Messages(messages) => messages
            .into_iter()
            .flat_map(convert_responses_message_to_openai)
            .collect(),
        ResponsesInput::ContentParts(parts) => {
            let message = ResponsesMessage {
                role: "user".to_string(),
                content: ResponsesMessageContent::Parts(parts),
                r#type: None,
                id: None,
                status: None,
                name: None,
                metadata: None,
                extra_fields: HashMap::new(),
            };
            convert_responses_message_to_openai(message)
        }
    }
}

fn convert_responses_message_to_openai(message: ResponsesMessage) -> Vec<OpenAIMessage> {
    match message.content {
        ResponsesMessageContent::Text(text) => vec![OpenAIMessage {
            role: message.role,
            content: OpenAIContent::Text(text),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        ResponsesMessageContent::Parts(parts) => {
            let mut text_buffer = String::new();
            let mut media_items: Vec<OpenAIContentItem> = Vec::new();
            let mut reasoning_segments: Vec<String> = Vec::new();
            let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
            let mut followup_messages: Vec<OpenAIMessage> = Vec::new();

            for part in parts {
                match part {
                    ResponsesContentPart::InputText { text, .. }
                    | ResponsesContentPart::OutputText { text, .. } => {
                        if !text_buffer.is_empty() {
                            text_buffer.push_str("\n");
                        }
                        text_buffer.push_str(&text);
                    }
                    ResponsesContentPart::Reasoning { summary, .. } => {
                        let collected = summary
                            .into_iter()
                            .filter_map(|s| match s {
                                ResponsesReasoningSummary::SummaryText { text, .. } => Some(text),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !collected.is_empty() {
                            reasoning_segments.push(collected);
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
                    ResponsesContentPart::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        followup_messages.push(OpenAIMessage {
                            role: "tool".to_string(),
                            content: OpenAIContent::Text(output),
                            tool_calls: None,
                            tool_call_id: Some(call_id),
                            reasoning_content: None,
                        });
                    }
                    ResponsesContentPart::InputImage {
                        image_url, file_id, ..
                    } => {
                        if let Some(url) =
                            image_url.or_else(|| file_id.map(|id| format!("openai://file/{}", id)))
                        {
                            media_items.push(OpenAIContentItem {
                                r#type: "image_url".to_string(),
                                text: None,
                                image_url: Some(OpenAIImageUrl { url }),
                            });
                        }
                    }
                    ResponsesContentPart::Refusal { refusal, .. } => {
                        if !text_buffer.is_empty() {
                            text_buffer.push_str("\n");
                        }
                        text_buffer.push_str(&refusal);
                    }
                    ResponsesContentPart::InputFile {
                        filename, file_id, ..
                    } => {
                        let mut placeholder = String::from("[file]");
                        if let Some(name) = filename {
                            placeholder = format!("[file:{}]", name);
                        } else if let Some(id) = file_id {
                            placeholder = format!("[file_id:{}]", id);
                        }
                        if !text_buffer.is_empty() {
                            text_buffer.push_str("\n");
                        }
                        text_buffer.push_str(&placeholder);
                    }
                    ResponsesContentPart::FileSearchCall { .. }
                    | ResponsesContentPart::WebSearchCall { .. }
                    | ResponsesContentPart::ComputerCall { .. }
                    | ResponsesContentPart::ComputerCallOutput { .. }
                    | ResponsesContentPart::ItemReference { .. }
                    | ResponsesContentPart::FileCitation { .. }
                    | ResponsesContentPart::UrlCitation { .. }
                    | ResponsesContentPart::FilePath { .. }
                    | ResponsesContentPart::Unknown => {
                        // Unsupported part types are skipped.
                    }
                }
            }

            let reasoning = reasoning_segments.join("\n");
            let mut content = if media_items.is_empty() {
                OpenAIContent::Text(text_buffer.clone())
            } else {
                let mut items = media_items;
                if !text_buffer.is_empty() {
                    items.insert(
                        0,
                        OpenAIContentItem {
                            r#type: "text".to_string(),
                            text: Some(text_buffer.clone()),
                            image_url: None,
                        },
                    );
                }
                OpenAIContent::Array(items)
            };

            // Ensure we have at least empty text content when required
            if matches!(content, OpenAIContent::Text(ref t) if t.is_empty())
                && tool_calls.is_empty()
                && reasoning.is_empty()
            {
                content = OpenAIContent::Text(text_buffer);
            }

            let mut messages = Vec::new();
            if !matches!(&content, OpenAIContent::Text(t) if t.is_empty())
                || !tool_calls.is_empty()
                || !reasoning.is_empty()
            {
                messages.push(OpenAIMessage {
                    role: message.role,
                    content,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    reasoning_content: if reasoning.is_empty() {
                        None
                    } else {
                        Some(reasoning)
                    },
                });
            }
            messages.extend(followup_messages);
            messages
        }
    }
}

fn responses_tool_to_openai_tool(tool: ResponsesTool) -> Option<OpenAITool> {
    match tool {
        ResponsesTool::Function {
            name,
            parameters,
            strict,
            description,
            ..
        } => Some(OpenAITool {
            r#type: "function".to_string(),
            function: OpenAIFunction {
                name,
                description: description.unwrap_or_default(),
                parameters,
                strict,
            },
        }),
        _ => None,
    }
}

fn responses_text_config_to_response_format(
    text: Option<ResponsesTextConfig>,
) -> Option<OpenAIResponseFormat> {
    let config = text?;
    match config.format {
        Some(ResponsesTextFormat::JsonObject { .. }) => Some(OpenAIResponseFormat {
            r#type: "json_object".to_string(),
            json_schema: None,
        }),
        Some(ResponsesTextFormat::JsonSchema(ResponsesTextFormatJsonSchema {
            name,
            schema,
            strict,
            ..
        })) => Some(OpenAIResponseFormat {
            r#type: "json_schema".to_string(),
            json_schema: Some(OpenAIJSONSchemaSpec {
                name,
                schema,
                strict,
            }),
        }),
        _ => None,
    }
}
