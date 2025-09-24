use crate::converters::anthropic::{
    AnthropicContent, AnthropicContentBlock, AnthropicContentObject, AnthropicStreamChunk,
    AnthropicStreamDelta, AnthropicStreamMessage, AnthropicUsage,
};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;
use std::collections::HashMap;

use super::responses_content::ResponsesContentPart;
use super::responses_content::ResponsesReasoningSummary;
use super::responses_output_item::ResponsesOutputItem;
use super::responses_response::ResponsesResponse;
use super::responses_usage::ResponsesUsage;

/// Streaming event envelope used when `stream: true` is set on a Responses request.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesStreamChunk {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub payload: ResponsesStreamEventPayload,
}

impl<'de> Deserialize<'de> for ResponsesStreamChunk {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw_value = Value::deserialize(deserializer)?;
        let mut raw_object = match raw_value {
            Value::Object(map) => map,
            _ => {
                return Err(D::Error::custom(
                    "expected ResponsesStreamChunk to deserialize from an object",
                ));
            }
        };

        let event_type = match raw_object.remove("type") {
            Some(Value::String(s)) => s,
            Some(_) => {
                return Err(D::Error::custom(
                    "expected `type` field to be a string for ResponsesStreamChunk",
                ));
            }
            None => return Err(D::Error::missing_field("type")),
        };

        let payload_value = Value::Object(raw_object);

        let payload = match event_type.as_str() {
            "response.content_part.done" => {
                let data: ContentPartEnvelope =
                    serde_json::from_value(payload_value).map_err(D::Error::custom)?;
                ResponsesStreamEventPayload::ContentPartDone {
                    item_id: data.item_id,
                    output_index: data.output_index,
                    content_index: data.content_index,
                    part: data.part,
                }
            }
            "response.output_item.done" => {
                let data: OutputItemEnvelope =
                    serde_json::from_value(payload_value).map_err(D::Error::custom)?;
                ResponsesStreamEventPayload::OutputItemDone {
                    output_index: data.output_index,
                    item: data.item,
                }
            }
            _ => serde_json::from_value(payload_value).map_err(D::Error::custom)?,
        };

        Ok(Self {
            event_type,
            payload,
        })
    }
}

#[derive(Deserialize)]
struct ContentPartEnvelope {
    item_id: String,
    output_index: usize,
    content_index: usize,
    part: ResponsesContentPart,
}

#[derive(Deserialize)]
struct OutputItemEnvelope {
    output_index: usize,
    item: ResponsesOutputItem,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesStreamEventPayload {
    Response {
        response: ResponsesResponse,
    },
    OutputItem {
        output_index: usize,
        item: ResponsesOutputItem,
    },
    ContentPart {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ResponsesContentPart,
    },
    OutputTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    OutputTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    ContentPartDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ResponsesContentPart,
    },
    OutputItemDone {
        output_index: usize,
        item: ResponsesOutputItem,
    },
    Generic {
        #[serde(flatten)]
        #[serde(default)]
        fields: HashMap<String, Value>,
    },
}

impl From<AnthropicStreamChunk> for ResponsesStreamChunk {
    fn from(chunk: AnthropicStreamChunk) -> Self {
        match chunk {
            AnthropicStreamChunk::MessageStart { message } => ResponsesStreamChunk {
                event_type: "response.created".to_string(),
                payload: ResponsesStreamEventPayload::Response {
                    response: anthropic_message_to_responses_response(message),
                },
            },
            AnthropicStreamChunk::ContentBlockStart {
                index,
                content_block,
            } => {
                if let Some(part) = anthropic_block_to_responses_part(&content_block) {
                    ResponsesStreamChunk {
                        event_type: "response.content_part.added".to_string(),
                        payload: ResponsesStreamEventPayload::ContentPart {
                            item_id: format!("item-{}", index),
                            output_index: index.max(0) as usize,
                            content_index: index.max(0) as usize,
                            part,
                        },
                    }
                } else {
                    ResponsesStreamChunk {
                        event_type: "response.generic".to_string(),
                        payload: ResponsesStreamEventPayload::Generic {
                            fields: generic_payload_from_block(index, content_block),
                        },
                    }
                }
            }
            AnthropicStreamChunk::ContentBlockDelta { index, delta } => match delta {
                AnthropicStreamDelta::TextDelta { text } => ResponsesStreamChunk {
                    event_type: "response.output_text.delta".to_string(),
                    payload: ResponsesStreamEventPayload::OutputTextDelta {
                        item_id: format!("item-{}", index),
                        output_index: index.max(0) as usize,
                        content_index: index.max(0) as usize,
                        delta: text,
                    },
                },
                AnthropicStreamDelta::ThinkingDelta { thinking } => ResponsesStreamChunk {
                    event_type: "response.content_part.delta".to_string(),
                    payload: ResponsesStreamEventPayload::Generic {
                        fields: generic_delta_payload(index, "thinking_delta", thinking.into()),
                    },
                },
                AnthropicStreamDelta::InputJsonDelta {
                    partial_json,
                    name,
                    id,
                } => ResponsesStreamChunk {
                    event_type: "response.content_part.delta".to_string(),
                    payload: ResponsesStreamEventPayload::Generic {
                        fields: generic_tool_delta_payload(index, partial_json, name, id),
                    },
                },
            },
            AnthropicStreamChunk::ContentBlockStop { index } => ResponsesStreamChunk {
                event_type: "response.content_part.done".to_string(),
                payload: ResponsesStreamEventPayload::ContentPartDone {
                    item_id: format!("item-{}", index),
                    output_index: index.max(0) as usize,
                    content_index: index.max(0) as usize,
                    part: ResponsesContentPart::Unknown,
                },
            },
            AnthropicStreamChunk::MessageDelta { delta, usage } => {
                let mut extra = HashMap::new();
                if let Some(u) = usage {
                    if let Ok(serialized) =
                        serde_json::to_value(anthropic_usage_to_responses_usage(u))
                    {
                        extra.insert("usage".to_string(), serialized);
                    }
                }

                ResponsesStreamChunk {
                    event_type: "response.output_item.done".to_string(),
                    payload: ResponsesStreamEventPayload::OutputItemDone {
                        output_index: 0,
                        item: ResponsesOutputItem::Message {
                            id: "item-0".to_string(),
                            status: map_stop_reason_to_responses_status(
                                delta.stop_reason.as_deref(),
                            ),
                            role: "assistant".to_string(),
                            content: Vec::new(),
                            metadata: None,
                            name: None,
                            extra,
                        },
                    },
                }
            }
            AnthropicStreamChunk::MessageStop => ResponsesStreamChunk {
                event_type: "response.completed".to_string(),
                payload: ResponsesStreamEventPayload::Generic {
                    fields: {
                        let mut map = HashMap::new();
                        map.insert("status".to_string(), Value::String("completed".to_string()));
                        map
                    },
                },
            },
            AnthropicStreamChunk::Ping => ResponsesStreamChunk {
                event_type: "response.heartbeat".to_string(),
                payload: ResponsesStreamEventPayload::Generic {
                    fields: HashMap::new(),
                },
            },
        }
    }
}

fn anthropic_block_to_responses_part(
    block: &AnthropicContentBlock,
) -> Option<ResponsesContentPart> {
    match block {
        AnthropicContentBlock::Text { text } => Some(ResponsesContentPart::OutputText {
            text: text.clone(),
            annotations: Vec::new(),
            extra: HashMap::new(),
        }),
        AnthropicContentBlock::Thinking { thinking, .. } => Some(ResponsesContentPart::Reasoning {
            id: "reasoning-0".to_string(),
            summary: vec![ResponsesReasoningSummary::SummaryText {
                text: thinking.clone(),
                extra: HashMap::new(),
            }],
            encrypted_content: None,
            status: None,
            extra: HashMap::new(),
        }),
        AnthropicContentBlock::ToolUse { id, name, input } => {
            let arguments = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
            Some(ResponsesContentPart::FunctionCall {
                arguments,
                call_id: id.clone(),
                name: name.clone(),
                id: Some(id.clone()),
                status: None,
                extra: HashMap::new(),
            })
        }
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn anthropic_message_to_responses_response(message: AnthropicStreamMessage) -> ResponsesResponse {
    let mut content_parts = Vec::new();
    for entry in &message.content {
        content_parts.extend(anthropic_content_to_responses_parts(entry));
    }

    let item = ResponsesOutputItem::Message {
        id: format!("item-{}", message.id),
        status: "in_progress".to_string(),
        role: message.role.clone(),
        content: content_parts,
        metadata: None,
        name: None,
        extra: HashMap::new(),
    };

    ResponsesResponse {
        id: message.id,
        object: Some(message.r#type),
        created_at: current_timestamp(),
        status: "in_progress".to_string(),
        error: None,
        incomplete_details: None,
        instructions: None,
        max_output_tokens: None,
        model: message.model,
        output: vec![item],
        parallel_tool_calls: None,
        previous_response_id: None,
        reasoning: None,
        store: None,
        temperature: None,
        text: None,
        tool_choice: None,
        tools: None,
        top_p: None,
        truncation: None,
        usage: message.usage.map(anthropic_usage_to_responses_usage),
        user: None,
        metadata: None,
        extra: HashMap::new(),
    }
}

fn generic_delta_payload(index: i32, kind: &str, value: Value) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    map.insert("index".to_string(), Value::from(index));
    map.insert("kind".to_string(), Value::String(kind.to_string()));
    map.insert("value".to_string(), value);
    map
}

fn generic_payload_from_block(
    index: i32,
    content_block: AnthropicContentBlock,
) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    map.insert("index".to_string(), Value::from(index));
    map.insert(
        "content_block".to_string(),
        serde_json::to_value(content_block).unwrap_or(Value::Null),
    );
    map
}

fn generic_tool_delta_payload(
    index: i32,
    partial_json: Option<String>,
    name: Option<String>,
    id: Option<String>,
) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    map.insert("index".to_string(), Value::from(index));
    if let Some(json_str) = partial_json {
        map.insert("partial_json".to_string(), Value::String(json_str));
    }
    if let Some(name) = name {
        map.insert("name".to_string(), Value::String(name));
    }
    if let Some(id) = id {
        map.insert("id".to_string(), Value::String(id));
    }
    map
}

fn map_stop_reason_to_responses_status(reason: Option<&str>) -> String {
    match reason.unwrap_or("end_turn") {
        "tool_use" => "requires_action".to_string(),
        "max_tokens" => "incomplete".to_string(),
        "abort" | "user_abort" => "cancelled".to_string(),
        "error" => "failed".to_string(),
        _ => "completed".to_string(),
    }
}

fn anthropic_usage_to_responses_usage(usage: AnthropicUsage) -> ResponsesUsage {
    ResponsesUsage {
        input_tokens: usage.input_tokens as u64,
        input_tokens_details: None,
        output_tokens: usage.output_tokens as u64,
        output_tokens_details: None,
        total_tokens: (usage.input_tokens as u64) + (usage.output_tokens as u64),
        extra: HashMap::new(),
    }
}

fn anthropic_content_to_responses_parts(content: &AnthropicContent) -> Vec<ResponsesContentPart> {
    match content {
        AnthropicContent::Text(text) => vec![ResponsesContentPart::OutputText {
            text: text.clone(),
            annotations: Vec::new(),
            extra: HashMap::new(),
        }],
        AnthropicContent::Array(objects) => objects
            .iter()
            .enumerate()
            .filter_map(|(idx, obj)| anthropic_object_to_responses_part(idx, obj))
            .collect(),
    }
}

fn anthropic_object_to_responses_part(
    idx: usize,
    object: &AnthropicContentObject,
) -> Option<ResponsesContentPart> {
    match object {
        AnthropicContentObject::Text { text } => Some(ResponsesContentPart::OutputText {
            text: text.clone(),
            annotations: Vec::new(),
            extra: HashMap::new(),
        }),
        AnthropicContentObject::Thinking { thinking, .. }
        | AnthropicContentObject::RedactedThinking { data: thinking } => {
            Some(ResponsesContentPart::Reasoning {
                id: format!("reasoning-{}", idx),
                summary: vec![ResponsesReasoningSummary::SummaryText {
                    text: thinking.clone(),
                    extra: HashMap::new(),
                }],
                encrypted_content: None,
                status: None,
                extra: HashMap::new(),
            })
        }
        AnthropicContentObject::ToolUse { id, name, input } => {
            let arguments = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
            Some(ResponsesContentPart::FunctionCall {
                arguments,
                call_id: id.clone(),
                name: name.clone(),
                id: Some(id.clone()),
                status: None,
                extra: HashMap::new(),
            })
        }
        AnthropicContentObject::ToolResult {
            tool_use_id,
            content,
        } => Some(ResponsesContentPart::FunctionCallOutput {
            call_id: tool_use_id.clone(),
            output: content.clone(),
            id: None,
            status: None,
            extra: HashMap::new(),
        }),
        AnthropicContentObject::Image { source } => Some(ResponsesContentPart::InputImage {
            detail: None,
            file_id: None,
            image_url: source.url.clone(),
            extra: HashMap::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::anthropic::AnthropicMessageDelta;
    use serde_json::{Value, json};

    #[test]
    fn test_message_start_to_response_created() {
        let message = AnthropicStreamMessage {
            id: "resp_123".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContent::Text("Hello".to_string())],
            model: "claude-3".to_string(),
            stop_reason: None,
            usage: Some(AnthropicUsage {
                input_tokens: 5,
                output_tokens: 7,
            }),
        };

        let chunk = AnthropicStreamChunk::MessageStart {
            message: message.clone(),
        };

        let responses_chunk: ResponsesStreamChunk = chunk.into();

        assert_eq!(responses_chunk.event_type, "response.created");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::Response { response } => {
                assert_eq!(response.id, message.id);
                assert_eq!(response.status, "in_progress");
                assert_eq!(response.model, message.model);
                assert_eq!(response.output.len(), 1);
                let usage = response.usage.expect("usage to be present");
                assert_eq!(usage.input_tokens, 5);
                assert_eq!(usage.output_tokens, 7);
            }
            _ => panic!("expected response payload"),
        }
    }

    #[test]
    fn test_content_block_start_to_content_part_added() {
        let chunk = AnthropicStreamChunk::ContentBlockStart {
            index: 0,
            content_block: AnthropicContentBlock::Text {
                text: "hola".to_string(),
            },
        };

        let responses_chunk: ResponsesStreamChunk = chunk.into();

        assert_eq!(responses_chunk.event_type, "response.content_part.added");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::ContentPart {
                item_id,
                output_index,
                content_index,
                part,
            } => {
                assert_eq!(item_id, "item-0");
                assert_eq!(output_index, 0);
                assert_eq!(content_index, 0);
                match part {
                    ResponsesContentPart::OutputText { text, .. } => {
                        assert_eq!(text, "hola");
                    }
                    _ => panic!("expected output text part"),
                }
            }
            _ => panic!("expected content part payload"),
        }
    }

    #[test]
    fn test_thinking_delta_to_generic_payload() {
        let chunk = AnthropicStreamChunk::ContentBlockDelta {
            index: 2,
            delta: AnthropicStreamDelta::ThinkingDelta {
                thinking: "ponder".to_string(),
            },
        };

        let responses_chunk: ResponsesStreamChunk = chunk.into();

        assert_eq!(responses_chunk.event_type, "response.content_part.delta");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::Generic { fields } => {
                assert_eq!(fields.get("index"), Some(&Value::from(2)));
                assert_eq!(
                    fields.get("kind"),
                    Some(&Value::String("thinking_delta".to_string()))
                );
                assert_eq!(
                    fields.get("value"),
                    Some(&Value::String("ponder".to_string()))
                );
            }
            _ => panic!("expected generic payload"),
        }
    }

    #[test]
    fn test_input_json_delta_to_generic_payload() {
        let chunk = AnthropicStreamChunk::ContentBlockDelta {
            index: 5,
            delta: AnthropicStreamDelta::InputJsonDelta {
                partial_json: Some("{\"foo\":1}".to_string()),
                name: Some("tool".to_string()),
                id: Some("call_1".to_string()),
            },
        };

        let responses_chunk: ResponsesStreamChunk = chunk.into();

        assert_eq!(responses_chunk.event_type, "response.content_part.delta");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::Generic { fields } => {
                assert_eq!(fields.get("index"), Some(&Value::from(5)));
                assert_eq!(
                    fields.get("partial_json"),
                    Some(&Value::String("{\"foo\":1}".to_string()))
                );
                assert_eq!(fields.get("name"), Some(&Value::String("tool".to_string())));
                assert_eq!(fields.get("id"), Some(&Value::String("call_1".to_string())));
            }
            _ => panic!("expected generic payload"),
        }
    }

    #[test]
    fn test_content_block_stop_to_content_part_done() {
        let chunk = AnthropicStreamChunk::ContentBlockStop { index: 3 };
        let responses_chunk: ResponsesStreamChunk = chunk.into();

        assert_eq!(responses_chunk.event_type, "response.content_part.done");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::ContentPartDone {
                item_id,
                output_index,
                content_index,
                part,
            } => {
                assert_eq!(item_id, "item-3");
                assert_eq!(output_index, 3);
                assert_eq!(content_index, 3);
                assert!(matches!(part, ResponsesContentPart::Unknown));
            }
            _ => panic!("expected content part done payload"),
        }
    }

    #[test]
    fn test_message_delta_to_output_item_done() {
        let chunk = AnthropicStreamChunk::MessageDelta {
            delta: AnthropicMessageDelta {
                stop_reason: Some("end_turn".to_string()),
            },
            usage: Some(AnthropicUsage {
                input_tokens: 11,
                output_tokens: 13,
            }),
        };

        let responses_chunk: ResponsesStreamChunk = chunk.into();

        assert_eq!(responses_chunk.event_type, "response.output_item.done");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::OutputItemDone { output_index, item } => {
                assert_eq!(output_index, 0);
                match item {
                    ResponsesOutputItem::Message { status, extra, .. } => {
                        assert_eq!(status, "completed");
                        assert!(extra.contains_key("usage"));
                    }
                    _ => panic!("expected message output item"),
                }
            }
            _ => panic!("expected output item done payload"),
        }
    }

    #[test]
    fn test_message_stop_to_response_completed() {
        let chunk = AnthropicStreamChunk::MessageStop;
        let responses_chunk: ResponsesStreamChunk = chunk.into();

        assert_eq!(responses_chunk.event_type, "response.completed");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::Generic { fields } => {
                assert_eq!(
                    fields.get("status"),
                    Some(&Value::String("completed".to_string()))
                );
            }
            _ => panic!("expected generic payload"),
        }
    }

    #[test]
    fn test_anthropic_text_delta_to_responses() {
        let anthropic_chunk = AnthropicStreamChunk::ContentBlockDelta {
            index: 1,
            delta: AnthropicStreamDelta::TextDelta {
                text: "World".to_string(),
            },
        };

        let responses_chunk: ResponsesStreamChunk = anthropic_chunk.into();

        assert_eq!(responses_chunk.event_type, "response.output_text.delta");
        match responses_chunk.payload {
            ResponsesStreamEventPayload::OutputTextDelta {
                item_id,
                output_index,
                content_index,
                delta,
            } => {
                assert_eq!(item_id, "item-1");
                assert_eq!(output_index, 1);
                assert_eq!(content_index, 1);
                assert_eq!(delta, "World");
            }
            _ => panic!("expected output text delta payload"),
        }
    }

    #[test]
    fn test_heartbeat_event_type_matches_payload_type() {
        let responses_chunk: ResponsesStreamChunk = AnthropicStreamChunk::Ping.into();
        let serialized = serde_json::to_value(&responses_chunk).expect("serialize chunk");
        assert_eq!(serialized["type"], "response.heartbeat");
    }

    // --- Deserialization tests for ResponsesStreamChunk types ---

    #[test]
    fn test_deserialize_response_created() {
        let data = json!({
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 1u64,
                "status": "in_progress",
                "model": "gpt-4o",
                "output": []
            }
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.created");
        match chunk.payload {
            ResponsesStreamEventPayload::Response { response } => {
                assert_eq!(response.id, "resp_1");
                assert_eq!(response.status, "in_progress");
                assert_eq!(response.model, "gpt-4o");
            }
            _ => panic!("expected Response payload"),
        }
    }

    #[test]
    fn test_deserialize_response_in_progress() {
        let data = json!({
            "type": "response.in_progress",
            "response": {
                "id": "resp_2",
                "object": "response",
                "created_at": 2u64,
                "status": "in_progress",
                "model": "gpt-4o-mini",
                "output": []
            }
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.in_progress");
        match chunk.payload {
            ResponsesStreamEventPayload::Response { response } => {
                assert_eq!(response.id, "resp_2");
                assert_eq!(response.status, "in_progress");
                assert_eq!(response.model, "gpt-4o-mini");
            }
            _ => panic!("expected Response payload"),
        }
    }

    #[test]
    fn test_deserialize_output_item_added() {
        let data = json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "item_1",
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.output_item.added");
        match chunk.payload {
            ResponsesStreamEventPayload::OutputItem { output_index, item } => {
                assert_eq!(output_index, 0);
                match item {
                    ResponsesOutputItem::Message {
                        id, status, role, ..
                    } => {
                        assert_eq!(id, "item_1");
                        assert_eq!(status, "in_progress");
                        assert_eq!(role, "assistant");
                    }
                    _ => panic!("expected message output item"),
                }
            }
            _ => panic!("expected OutputItem payload"),
        }
    }

    #[test]
    fn test_deserialize_content_part_added() {
        let data = json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "Hello",
                "annotations": []
            }
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.content_part.added");
        match chunk.payload {
            ResponsesStreamEventPayload::ContentPart {
                item_id,
                output_index,
                content_index,
                part,
            } => {
                assert_eq!(item_id, "msg_1");
                assert_eq!(output_index, 0);
                assert_eq!(content_index, 0);
                match part {
                    ResponsesContentPart::OutputText { text, .. } => assert_eq!(text, "Hello"),
                    _ => panic!("expected OutputText part"),
                }
            }
            _ => panic!("expected ContentPart payload"),
        }
    }

    #[test]
    fn test_deserialize_output_text_delta() {
        let data = json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "你"
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.output_text.delta");
        match chunk.payload {
            ResponsesStreamEventPayload::OutputTextDelta {
                item_id,
                output_index,
                content_index,
                delta,
            } => {
                assert_eq!(item_id, "msg_1");
                assert_eq!(output_index, 0);
                assert_eq!(content_index, 0);
                assert_eq!(delta, "你");
            }
            _ => panic!("expected OutputTextDelta payload"),
        }
    }

    #[test]
    fn test_deserialize_output_text_done() {
        let data = json!({
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": "你好！"
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.output_text.done");
        match chunk.payload {
            ResponsesStreamEventPayload::OutputTextDone {
                item_id,
                output_index,
                content_index,
                text,
            } => {
                assert_eq!(item_id, "msg_1");
                assert_eq!(output_index, 0);
                assert_eq!(content_index, 0);
                assert_eq!(text, "你好！");
            }
            _ => panic!("expected OutputTextDone payload"),
        }
    }

    #[test]
    fn test_deserialize_content_part_done() {
        let data = json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 1,
            "part": {
                "type": "output_text",
                "text": "Done",
                "annotations": []
            }
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.content_part.done");
        match chunk.payload {
            ResponsesStreamEventPayload::ContentPartDone {
                item_id,
                output_index,
                content_index,
                part,
            } => {
                assert_eq!(item_id, "msg_1");
                assert_eq!(output_index, 0);
                assert_eq!(content_index, 1);
                match part {
                    ResponsesContentPart::OutputText { text, .. } => assert_eq!(text, "Done"),
                    _ => panic!("expected OutputText part"),
                }
            }
            _ => panic!("expected ContentPartDone payload"),
        }
    }

    #[test]
    fn test_deserialize_output_item_done() {
        let data = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "item_9",
                "status": "completed",
                "role": "assistant",
                "content": []
            }
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.output_item.done");
        match chunk.payload {
            ResponsesStreamEventPayload::OutputItemDone { output_index, item } => {
                assert_eq!(output_index, 0);
                match item {
                    ResponsesOutputItem::Message { status, .. } => assert_eq!(status, "completed"),
                    _ => panic!("expected message output item"),
                }
            }
            _ => panic!("expected OutputItemDone payload"),
        }
    }

    #[test]
    fn test_deserialize_content_part_delta_thinking_generic() {
        let data = json!({
            "type": "response.content_part.delta",
            "index": 2,
            "kind": "thinking_delta",
            "value": "ponder"
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.content_part.delta");
        match chunk.payload {
            ResponsesStreamEventPayload::Generic { fields } => {
                assert_eq!(fields.get("index"), Some(&Value::from(2)));
                assert_eq!(
                    fields.get("kind"),
                    Some(&Value::String("thinking_delta".to_string()))
                );
                assert_eq!(
                    fields.get("value"),
                    Some(&Value::String("ponder".to_string()))
                );
            }
            _ => panic!("expected Generic payload"),
        }
    }

    #[test]
    fn test_deserialize_content_part_delta_tool_use_generic() {
        let data = json!({
            "type": "response.content_part.delta",
            "index": 1,
            "partial_json": "{\"foo\":1}",
            "name": "tool",
            "id": "call_1"
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.content_part.delta");
        match chunk.payload {
            ResponsesStreamEventPayload::Generic { fields } => {
                assert_eq!(fields.get("index"), Some(&Value::from(1)));
                assert_eq!(
                    fields.get("partial_json"),
                    Some(&Value::String("{\"foo\":1}".to_string()))
                );
                assert_eq!(fields.get("name"), Some(&Value::String("tool".to_string())));
                assert_eq!(fields.get("id"), Some(&Value::String("call_1".to_string())));
            }
            _ => panic!("expected Generic payload"),
        }
    }

    #[test]
    fn test_deserialize_response_completed_generic() {
        let data = json!({
            "type": "response.completed",
            "status": "completed"
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.completed");
        match chunk.payload {
            ResponsesStreamEventPayload::Generic { fields } => {
                assert_eq!(
                    fields.get("status"),
                    Some(&Value::String("completed".to_string()))
                );
            }
            _ => panic!("expected Generic payload"),
        }
    }

    #[test]
    fn test_deserialize_response_heartbeat_generic() {
        let data = json!({
            "type": "response.heartbeat"
        });

        let chunk: ResponsesStreamChunk = serde_json::from_value(data).expect("deserialize");
        assert_eq!(chunk.event_type, "response.heartbeat");
        match chunk.payload {
            ResponsesStreamEventPayload::Generic { fields } => {
                assert!(fields.is_empty());
            }
            _ => panic!("expected Generic payload"),
        }
    }
}
