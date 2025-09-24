use crate::converters::openai::openai_request::OpenAIResponseFormat;
use crate::converters::openai::{OpenAIContent, OpenAIMessage, OpenAIRequest, OpenAITool};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::responses_content::{ResponsesContentPart, ResponsesReasoningSummary};
use super::responses_message::{ResponsesMessage, ResponsesMessageContent};
use super::responses_reasoning_config::ResponsesReasoningConfig;
use super::responses_text_config::ResponsesTextConfig;
use super::responses_text_format::ResponsesTextFormat;
use super::responses_text_format_json_schema::ResponsesTextFormatJsonSchema;
use super::responses_tool::ResponsesTool;
use super::responses_tool_choice::ResponsesToolChoice;

/// Top-level request payload for the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(flatten)]
    #[serde(default)]
    pub extra_fields: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Messages(Vec<ResponsesMessage>),
    ContentParts(Vec<ResponsesContentPart>),
}

impl From<OpenAIRequest> for ResponsesRequest {
    fn from(openai: OpenAIRequest) -> Self {
        let OpenAIRequest {
            model,
            messages,
            max_tokens,
            temperature,
            response_format,
            tools,
            stream,
            extra_fields,
        } = openai;

        let mut instructions_segments: Vec<String> = Vec::new();
        let mut converted_messages: Vec<ResponsesMessage> = Vec::new();

        for message in messages {
            if message.role == "system" {
                let instr = openai_content_to_text(message.content);
                if !instr.trim().is_empty() {
                    instructions_segments.push(instr);
                }
            } else {
                converted_messages.extend(convert_openai_message_to_responses(message));
            }
        }

        let instructions = if instructions_segments.is_empty() {
            None
        } else {
            Some(instructions_segments.join("\n"))
        };

        let input = if converted_messages.is_empty() {
            ResponsesInput::Text(String::new())
        } else {
            ResponsesInput::Messages(converted_messages)
        };

        let text_config = response_format_to_responses_text_config(response_format);

        let responses_tools = tools.and_then(|ts| {
            let converted: Vec<ResponsesTool> = ts
                .into_iter()
                .filter_map(openai_tool_to_responses_tool)
                .collect();
            if converted.is_empty() {
                None
            } else {
                Some(converted)
            }
        });

        ResponsesRequest {
            model,
            input,
            include: None,
            instructions,
            max_output_tokens: max_tokens,
            metadata: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            service_tier: None,
            store: None,
            stream,
            temperature,
            text: text_config,
            tool_choice: None,
            tools: responses_tools,
            top_p: None,
            truncation: None,
            user: None,
            extra_fields,
        }
    }
}

fn openai_content_to_text(content: OpenAIContent) -> String {
    match content {
        OpenAIContent::Text(text) => text,
        OpenAIContent::Array(items) => items
            .into_iter()
            .filter_map(|item| {
                if item.r#type == "text" {
                    item.text
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn convert_openai_message_to_responses(message: OpenAIMessage) -> Vec<ResponsesMessage> {
    if message.role == "tool" {
        let call_id = message
            .tool_call_id
            .unwrap_or_else(|| "tool_call".to_string());
        let output_text = openai_content_to_text(message.content);
        let part = ResponsesContentPart::FunctionCallOutput {
            call_id,
            output: output_text,
            id: None,
            status: None,
            extra: HashMap::new(),
        };
        return vec![ResponsesMessage {
            role: "tool".to_string(),
            content: ResponsesMessageContent::Parts(vec![part]),
            r#type: None,
            id: None,
            status: None,
            name: None,
            metadata: None,
            extra_fields: HashMap::new(),
        }];
    }

    let mut text_buffer = String::new();
    let mut parts: Vec<ResponsesContentPart> = Vec::new();
    let mut insert_index = 0usize;

    match message.content {
        OpenAIContent::Text(text) => {
            text_buffer = text;
        }
        OpenAIContent::Array(items) => {
            for item in items {
                match item.r#type.as_str() {
                    "text" => {
                        if let Some(text) = item.text {
                            text_buffer.push_str(&text);
                        }
                    }
                    "image_url" => {
                        if let Some(image) = item.image_url {
                            parts.push(ResponsesContentPart::InputImage {
                                detail: None,
                                file_id: None,
                                image_url: Some(image.url),
                                extra: HashMap::new(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(reasoning) = message.reasoning_content {
        parts.insert(
            insert_index,
            ResponsesContentPart::Reasoning {
                id: "reasoning".to_string(),
                summary: vec![ResponsesReasoningSummary::SummaryText {
                    text: reasoning,
                    extra: HashMap::new(),
                }],
                encrypted_content: None,
                status: None,
                extra: HashMap::new(),
            },
        );
        insert_index += 1;
    }

    if let Some(tool_calls) = message.tool_calls {
        for call in tool_calls {
            parts.push(ResponsesContentPart::FunctionCall {
                id: Some(call.id.clone()),
                call_id: call.id,
                name: call.function.name,
                arguments: call.function.arguments,
                status: None,
                extra: HashMap::new(),
            });
        }
    }

    if parts.is_empty() {
        return vec![ResponsesMessage {
            role: message.role,
            content: ResponsesMessageContent::Text(text_buffer),
            r#type: None,
            id: None,
            status: None,
            name: None,
            metadata: None,
            extra_fields: HashMap::new(),
        }];
    }

    if !text_buffer.is_empty() {
        parts.insert(insert_index, text_part_for_role(&message.role, text_buffer));
    }

    vec![ResponsesMessage {
        role: message.role,
        content: ResponsesMessageContent::Parts(parts),
        r#type: None,
        id: None,
        status: None,
        name: None,
        metadata: None,
        extra_fields: HashMap::new(),
    }]
}

fn text_part_for_role(role: &str, text: String) -> ResponsesContentPart {
    if role == "assistant" {
        ResponsesContentPart::OutputText {
            text,
            annotations: Vec::new(),
            extra: HashMap::new(),
        }
    } else {
        ResponsesContentPart::InputText {
            text,
            extra: HashMap::new(),
        }
    }
}

fn response_format_to_responses_text_config(
    response_format: Option<OpenAIResponseFormat>,
) -> Option<ResponsesTextConfig> {
    let format = response_format?;
    match format.r#type.as_str() {
        "json_object" => Some(ResponsesTextConfig {
            format: Some(ResponsesTextFormat::JsonObject {
                extra: HashMap::new(),
            }),
            extra_fields: HashMap::new(),
        }),
        "json_schema" => format.json_schema.map(|spec| ResponsesTextConfig {
            format: Some(ResponsesTextFormat::JsonSchema(
                ResponsesTextFormatJsonSchema {
                    name: spec.name,
                    schema: spec.schema,
                    description: None,
                    strict: spec.strict,
                    extra_fields: HashMap::new(),
                },
            )),
            extra_fields: HashMap::new(),
        }),
        _ => None,
    }
}

fn openai_tool_to_responses_tool(tool: OpenAITool) -> Option<ResponsesTool> {
    if tool.r#type != "function" {
        return None;
    }

    let OpenAITool { function, .. } = tool;

    let description = if function.description.is_empty() {
        None
    } else {
        Some(function.description)
    };

    Some(ResponsesTool::Function {
        name: function.name,
        parameters: function.parameters,
        strict: function.strict,
        description,
        extra: HashMap::new(),
    })
}
