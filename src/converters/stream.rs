use super::openai::OpenAIStreamChunk;
use super::anthropic::{
    AnthropicContentBlock, AnthropicStreamChunk, AnthropicStreamDelta, AnthropicStreamMessage,
};
use serde_json::json;
use crate::config::ApiType;

/// 将单行 SSE `data:` 载荷从 source -> target 转换为输出帧集合。
/// 返回的 Vec 中，(None, data) 表示 OpenAI 风格的无事件名数据帧；
/// (Some(event_name), data) 表示 Anthropic 风格的具名事件帧。
pub fn convert_sse_data_line(
    source_api_type: ApiType,
    target_api_type: ApiType,
    data: &str,
    model: &String,
    previous_event: &mut String,
    previous_delta_type: &mut String,
    msg_index: &mut i32,
) -> Vec<(Option<String>, String)> {
    match (source_api_type, target_api_type) {
        (ApiType::OpenAI, ApiType::OpenAI) => {
            if let Ok(mut chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                chunk.model = model.clone();
                if let Ok(s) = serde_json::to_string(&chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        (ApiType::Anthropic, ApiType::Anthropic) => {
            if let Ok(mut chunk) = serde_json::from_str::<AnthropicStreamChunk>(data) {
                if let AnthropicStreamChunk::MessageStart { mut message } = chunk.clone() {
                    let mut patched = message.clone();
                    patched.model = model.clone();
                    chunk = AnthropicStreamChunk::MessageStart { message: patched };
                }
                if let Ok(s) = serde_json::to_string(&chunk) {
                    return vec![(Some(chunk.stream_type().to_string()), s)];
                }
            }
            vec![]
        }
        (ApiType::Anthropic, ApiType::OpenAI) => {
            if let Ok(mut chunk) = serde_json::from_str::<AnthropicStreamChunk>(data) {
                if let AnthropicStreamChunk::MessageStart { mut message } = chunk.clone() {
                    let mut patched = message.clone();
                    patched.model = model.clone();
                    chunk = AnthropicStreamChunk::MessageStart { message: patched };
                }
                let openai_chunk: OpenAIStreamChunk = chunk.into();
                if let Ok(s) = serde_json::to_string(&openai_chunk) {
                    return vec![(None, s)];
                }
            }
            vec![]
        }
        (ApiType::OpenAI, ApiType::Anthropic) => {
            if let Ok(mut chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                chunk.model = model.clone();
                return openai_to_anthropic_stream_chunks(
                    &chunk,
                    model,
                    previous_event,
                    previous_delta_type,
                    msg_index,
                )
                .into_iter()
                .map(|(event, payload)| (Some(event), payload))
                .collect();
            }
            vec![]
        }
    }
}

fn handle_content_block_start_typed(
    anthropic_delta: &AnthropicStreamChunk,
    index: i32,
) -> Option<AnthropicStreamChunk> {
    match anthropic_delta {
        AnthropicStreamChunk::ContentBlockDelta { index: _, delta } => match delta {
            AnthropicStreamDelta::InputJsonDelta { name, id, .. } => {
                if let (Some(name), Some(id)) = (name.as_ref(), id.as_ref()) {
                    Some(AnthropicStreamChunk::ContentBlockStart {
                        index,
                        content_block: AnthropicContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: json!({}),
                        },
                    })
                } else {
                    None
                }
            }
            AnthropicStreamDelta::ThinkingDelta { .. } => Some(AnthropicStreamChunk::ContentBlockStart {
                index,
                content_block: AnthropicContentBlock::Thinking {
                    thinking: "".to_string(),
                    // Upstream struct requires a signature; keep empty to preserve tests
                    signature: "".to_string(),
                },
            }),
            AnthropicStreamDelta::TextDelta { .. } => Some(AnthropicStreamChunk::ContentBlockStart {
                index,
                content_block: AnthropicContentBlock::Text {
                    text: "".to_string(),
                },
            }),
        },
        _ => None,
    }
}

fn delta_kind(anthropic_delta: &AnthropicStreamChunk) -> Option<&'static str> {
    match anthropic_delta {
        AnthropicStreamChunk::ContentBlockDelta { delta, .. } => match delta {
            AnthropicStreamDelta::InputJsonDelta { .. } => Some("input_json_delta"),
            AnthropicStreamDelta::ThinkingDelta { .. } => Some("thinking_delta"),
            AnthropicStreamDelta::TextDelta { .. } => Some("text_delta"),
        },
        AnthropicStreamChunk::MessageDelta { .. } => Some("message_delta"),
        _ => None,
    }
}

/// 将 OpenAI 流式响应块转换为 Anthropic 流式事件序列（不使用 serde_json::Value 作为输入）。
pub fn openai_to_anthropic_stream_chunks(
    chunk: &OpenAIStreamChunk,
    model: &String,
    previous_event: &mut String,
    previous_delta_type: &mut String,
    msg_index: &mut i32,
) -> Vec<(String, String)> {
    let mut results: Vec<(String, String)> = vec![];

    // 初始 message_start
    if previous_event.is_empty() {
        let start = AnthropicStreamChunk::MessageStart {
            message: AnthropicStreamMessage {
                id: chunk.id.clone(),
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: model.clone(),
                stop_reason: None,
                usage: None,
            },
        };
        if let Ok(s) = serde_json::to_string(&start) {
            results.push(("message_start".to_string(), s));
        }
        previous_event.clear();
        previous_event.push_str("message_start");
    }

    // 提取 OpenAI delta 信息
    let (mut is_finish, mut is_reasoning_empty, mut is_content_empty, mut is_tool_calls_empty) =
        (false, true, true, true);
    if let Some(first_choice) = chunk.choices.as_ref().and_then(|v| v.first()) {
        is_finish = first_choice.finish_reason.as_ref().is_some();
        if let Some(delta) = first_choice.delta.as_ref() {
            is_reasoning_empty = delta
                .reasoning_content
                .as_ref()
                .map(|s| s.is_empty())
                .unwrap_or(true);
            is_content_empty = delta
                .content
                .as_ref()
                .map(|s| s.is_empty())
                .unwrap_or(true);
            is_tool_calls_empty = delta
                .tool_calls
                .as_ref()
                .map(|arr| arr.is_empty())
                .unwrap_or(true);
        }
    }

    if is_finish {
        // 在结束前先发送 content_block_stop（与旧逻辑保持一致）
        if let Ok(s) = serde_json::to_string(&AnthropicStreamChunk::ContentBlockStop { index: *msg_index }) {
            results.push(("content_block_stop".to_string(), s));
        }
    }

    // 没有任何内容增量且未结束，直接返回（仅包含 message_start）
    if is_reasoning_empty && is_content_empty && is_tool_calls_empty && !is_finish {
        return results;
    }

    // 将 OpenAI 块转换为 Anthropic 块（单个增量或消息级增量）
    let base_chunk: AnthropicStreamChunk = chunk.clone().into();
    let event_type = base_chunk.stream_type();
    let current_delta_type = delta_kind(&base_chunk).unwrap_or("");

    if previous_event == "message_start" {
        // 发送 content_block_start
        if let Some(start) = handle_content_block_start_typed(&base_chunk, *msg_index) {
            if let Ok(s) = serde_json::to_string(&start) {
                results.push(("content_block_start".to_string(), s));
            }
        }

        // 发送对应的增量事件（设置正确 index；tool_use 去掉 id/name）
        match &base_chunk {
            AnthropicStreamChunk::ContentBlockDelta { delta, .. } => {
                let delta_for_emit = match delta.clone() {
                    AnthropicStreamDelta::InputJsonDelta { partial_json, .. } => {
                        AnthropicStreamDelta::InputJsonDelta { partial_json, name: None, id: None }
                    }
                    other => other,
                };
                let emit = AnthropicStreamChunk::ContentBlockDelta { index: *msg_index, delta: delta_for_emit };
                if let Ok(s) = serde_json::to_string(&emit) {
                    results.push((event_type.to_string(), s));
                }
                previous_delta_type.clear();
                previous_delta_type.push_str(current_delta_type);
                previous_event.clear();
                previous_event.push_str("content_block_delta");
            }
            AnthropicStreamChunk::MessageDelta { .. } => {
                if let Ok(s) = serde_json::to_string(&base_chunk) {
                    results.push((event_type.to_string(), s));
                }
                if let Ok(s) = serde_json::to_string(&AnthropicStreamChunk::MessageStop) {
                    results.push(("message_stop".to_string(), s));
                }
            }
            _ => {}
        }
        return results;
    }

    if previous_event == "content_block_delta" {
        match &base_chunk {
            AnthropicStreamChunk::ContentBlockDelta { delta, .. } => {
                let new_delta_type = current_delta_type;
                if new_delta_type == previous_delta_type {
                    // 同一内容类型，直接追加增量
                    let delta_for_emit = match delta.clone() {
                        AnthropicStreamDelta::InputJsonDelta { partial_json, .. } => {
                            AnthropicStreamDelta::InputJsonDelta { partial_json, name: None, id: None }
                        }
                        other => other,
                    };
                    let emit = AnthropicStreamChunk::ContentBlockDelta { index: *msg_index, delta: delta_for_emit };
                    if let Ok(s) = serde_json::to_string(&emit) {
                        results.push(("content_block_delta".to_string(), s));
                    }
                } else if matches!(new_delta_type, "input_json_delta" | "thinking_delta" | "text_delta") {
                    // 切换内容类型，先停止当前内容块
                    if let Ok(s) = serde_json::to_string(&AnthropicStreamChunk::ContentBlockStop { index: *msg_index }) {
                        results.push(("content_block_stop".to_string(), s));
                    }
                    *msg_index += 1;

                    // 新内容块开始
                    if let Some(start) = handle_content_block_start_typed(&base_chunk, *msg_index) {
                        if let Ok(s) = serde_json::to_string(&start) {
                            results.push(("content_block_start".to_string(), s));
                        }
                    }

                    // 发送新的增量
                    let delta_for_emit = match delta.clone() {
                        AnthropicStreamDelta::InputJsonDelta { partial_json, .. } => {
                            AnthropicStreamDelta::InputJsonDelta { partial_json, name: None, id: None }
                        }
                        other => other,
                    };
                    let emit = AnthropicStreamChunk::ContentBlockDelta { index: *msg_index, delta: delta_for_emit };
                    if let Ok(s) = serde_json::to_string(&emit) {
                        results.push(("content_block_delta".to_string(), s));
                    }

                    previous_delta_type.clear();
                    previous_delta_type.push_str(new_delta_type);
                }
            }
            AnthropicStreamChunk::MessageDelta { .. } => {
                if let Ok(s) = serde_json::to_string(&base_chunk) {
                    results.push(("message_delta".to_string(), s));
                }
                if let Ok(s) = serde_json::to_string(&AnthropicStreamChunk::MessageStop) {
                    results.push(("message_stop".to_string(), s));
                }
            }
            _ => {}
        }
    }

    results
}


#[cfg(test)]
mod tests {
    use regex::Regex;
    use super::*;
    use serde_json::json;

    #[test]
    fn test_openai_to_anthropic_stream_chunks_message_start() {
        // 测试初始消息开始的情况
        let openai_chunk_json = json!({
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
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index
        );
        
        // 应该返回两个事件：message_start 和 content_block_delta
        assert_eq!(results.len(), 3);
        
        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");
        let first_data = first_event.1;
        assert!(Regex::new(r#""type":\s*"message_start""#).unwrap().is_match(&first_data));
        assert!(Regex::new(r#""id":\s*"chatcmpl-123""#).unwrap().is_match(&first_data));
        assert!(Regex::new(r#""model":\s*"claude-3-opus""#).unwrap().is_match(&first_data));
        
        // 检查第二个事件是 content_block_delta
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");

        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(Regex::new(r#""type":\s*"content_block_delta""#).unwrap().is_match(&third_data));
        assert!(Regex::new(r#""type":\s*"text_delta""#).unwrap().is_match(&third_data));
        assert!(Regex::new(r#""text":\s*"Hello""#).unwrap().is_match(&third_data));
        
        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "text_delta");
        assert_eq!(msg_index, 0);
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_with_reasoning_content() {
        // 测试包含推理内容的情况
        let openai_chunk_json = json!({
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
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index
        );
        
        // 应该返回两个事件：message_start 和 content_block_delta
        assert_eq!(results.len(), 3);
        
        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");
        
        // 检查第二个事件是 content_block_delta
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");

        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(Regex::new(r#""type":\s*"thinking_delta""#).unwrap().is_match(&third_data));
        assert!(Regex::new(r#""thinking":\s*"I need to think about this.""#).unwrap().is_match(&third_data));
        
        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "thinking_delta");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_with_tool_calls() {
        // 测试包含工具调用的情况
        let openai_chunk_json = json!({
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
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index
        );
        
        // 应该返回三个事件：message_start, content_block_start, 和 content_block_delta
        assert_eq!(results.len(), 3);
        
        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");
        
        // 检查第二个事件是 content_block_start
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");
        let second_data = second_event.1;
        assert!(Regex::new(r#""type":\s*"content_block_start""#).unwrap().is_match(&second_data));
        assert!(Regex::new(r#""type":\s*"tool_use""#).unwrap().is_match(&second_data));
        assert!(Regex::new(r#""id":\s*"call_abc123""#).unwrap().is_match(&second_data));
        assert!(Regex::new(r#""name":\s*"get_weather""#).unwrap().is_match(&second_data));
        
        // 检查第三个事件是 content_block_delta
        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(Regex::new(r#""type":\s*"input_json_delta""#).unwrap().is_match(&third_data));
        assert!(Regex::new(r#""partial_json":\s*"\{\\"location\\":\s*\\"San Francisco\\"\}""#).unwrap().is_match(&third_data));
        
        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "input_json_delta");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_with_finish_reason() {
        // 测试包含完成原因的情况
        let openai_chunk_json = json!({
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
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "content_block_delta".to_string();
        let mut previous_delta_type = "text_delta".to_string();
        let mut msg_index = 0;
        
        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index
        );
        
        // 应该返回三个事件：content_block_stop, message_delta, 和 message_stop
        assert_eq!(results.len(), 3);
        
        // 检查第一个事件是 content_block_stop
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "content_block_stop");
        let first_data = first_event.1;
        assert!(Regex::new(r#""type":\s*"content_block_stop""#).unwrap().is_match(&first_data));
        assert!(Regex::new(r#""index":\s*0"#).unwrap().is_match(&first_data));
        
        // 检查第二个事件是 message_delta
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "message_delta");
        let second_data = second_event.1;
        assert!(Regex::new(r#""type":\s*"message_delta""#).unwrap().is_match(&second_data));
        assert!(Regex::new(r#""stop_reason":\s*"end_turn""#).unwrap().is_match(&second_data));
        
        // 检查第三个事件是 message_stop
        let third_event = results[2].clone();
        assert_eq!(third_event.0, "message_stop");
        let third_data = third_event.1;
        assert!(Regex::new(r#""type":\s*"message_stop""#).unwrap().is_match(&third_data));
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_content_type_switch() {
        // 测试内容类型切换的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-switch",
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
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "content_block_delta".to_string();
        let mut previous_delta_type = "thinking_delta".to_string(); // 前一个是推理内容
        let mut msg_index = 0;
        
        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index
        );
        
        // 应该返回三个事件：content_block_stop, content_block_start, 和 content_block_delta
        assert_eq!(results.len(), 3);
        
        // 检查第一个事件是 content_block_stop
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "content_block_stop");
        
        // 检查第二个事件是 content_block_start
        let second_event = results[1].clone();
        assert_eq!(second_event.0, "content_block_start");
        
        // 检查第三个事件是 content_block_delta
        let third_event = results[2].clone();
        assert_eq!(third_event.0, "content_block_delta");
        let third_data = third_event.1;
        assert!(Regex::new(r#""type":\s*"text_delta""#).unwrap().is_match(&third_data));
        assert!(Regex::new(r#""text":\s*"Hello""#).unwrap().is_match(&third_data));
        
        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "content_block_delta");
        assert_eq!(previous_delta_type, "text_delta");
        assert_eq!(msg_index, 1); // 索引应该增加
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_empty_content() {
        // 测试空内容的情况
        let openai_chunk_json = json!({
            "id": "chatcmpl-empty",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": null
                }
            ]
        });
        let openai_chunk: OpenAIStreamChunk = serde_json::from_value(openai_chunk_json).unwrap();
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = openai_to_anthropic_stream_chunks(
            &openai_chunk,
            &model,
            &mut previous_event,
            &mut previous_delta_type,
            &mut msg_index
        );
        
        // 应该只返回一个事件：message_start
        assert_eq!(results.len(), 1);
        
        // 检查第一个事件是 message_start
        let first_event = results[0].clone();
        assert_eq!(first_event.0, "message_start");
        
        // 检查状态变量是否被正确更新
        assert_eq!(previous_event, "message_start");
        assert_eq!(previous_delta_type, "");
        assert_eq!(msg_index, 0);
    }
}
