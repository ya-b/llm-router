use serde_json::{json, Value};

pub struct ApiConverter;

impl ApiConverter {
    /// 将 OpenAI 请求转换为 Anthropic 请求
    pub fn openai_to_anthropic_request(openai_request: &Value) -> Value {
        let mut anthropic_request = json!({});

        // 处理 max_tokens
        if let Some(max_tokens) = openai_request.get("max_tokens") {
            anthropic_request["max_tokens"] = max_tokens.clone();
        } else {
            // Anthropic 要求必须有 max_tokens
            anthropic_request["max_tokens"] = json!(4096);
        }

        // 处理消息
        if let Some(messages) = openai_request.get("messages") {
            if let Some(messages_array) = messages.as_array() {
                anthropic_request["messages"] = Self::convert_openai_messages_to_anthropic(messages_array);
            }
        }

        // 处理系统消息
        if let Some(messages) = openai_request.get("messages") {
            if let Some(messages_array) = messages.as_array() {
                if let Some(system_message) = Self::extract_system_message(messages_array) {
                    anthropic_request["system"] = system_message;
                }
            }
        }

        // 处理工具调用
        if let Some(tools) = openai_request.get("tools") {
            anthropic_request["tools"] = Self::convert_openai_tools_to_anthropic(tools);
        }

        let processed_keys = vec!["max_tokens".to_string(), "messages".to_string(), "system".to_string(), "tools".to_string()];

        if let (Value::Object(map1), Value::Object(map2)) = (&mut anthropic_request, openai_request) {
            for (k, v) in map2 {
                if !processed_keys.contains(&k.to_string()) {
                    map1.insert(k.clone(), v.clone()); // 逐个复制
                }
            }
        }

        anthropic_request
    }

    /// 将 Anthropic 请求转换为 OpenAI 请求
    pub fn anthropic_to_openai_request(anthropic_request: &Value) -> Value {
        let mut openai_request = json!({});

        // 处理消息和系统消息
        let mut messages = Vec::new();
        
        // 添加系统消息
        if let Some(system) = anthropic_request.get("system") {
            messages.push(json!({
                "role": "system",
                "content": system
            }));
        }

        // 转换普通消息
        if let Some(anthropic_messages) = anthropic_request.get("messages") {
            if let Some(messages_array) = anthropic_messages.as_array() {
                messages.extend(Self::convert_anthropic_messages_to_openai(messages_array));
            }
        }

        openai_request["messages"] = json!(messages);

        // 处理工具调用
        if let Some(tools) = anthropic_request.get("tools") {
            openai_request["tools"] = Self::convert_anthropic_tools_to_openai(tools);
        }

        let processed_keys = vec!["messages".to_string(), "system".to_string(), "tools".to_string()];

        if let (Value::Object(map1), Value::Object(map2)) = (&mut openai_request, anthropic_request) {
            for (k, v) in map2 {
                if !processed_keys.contains(&k.to_string()) {
                    map1.insert(k.clone(), v.clone()); // 逐个复制
                }
            }
        }

        openai_request
    }

    /// 将 OpenAI 响应转换为 Anthropic 响应
    pub fn openai_to_anthropic_response(openai_response: &Value) -> Value {
        let mut anthropic_response = json!({});

        // 处理基本字段
        if let Some(id) = openai_response.get("id") {
            anthropic_response["id"] = id.clone();
        }

        anthropic_response["type"] = json!("message");
        anthropic_response["role"] = json!("assistant");

        // 处理内容
        if let Some(choices) = openai_response.get("choices") {
            if let Some(choices_array) = choices.as_array() {
                if let Some(first_choice) = choices_array.first() {
                    let mut content = Vec::new();

                    // 处理文本内容
                    if let Some(message) = first_choice.get("message") {
                        if let Some(reasoning_content) = message.get("reasoning_content") {
                            if !reasoning_content.is_null() && reasoning_content.as_str().unwrap_or("").trim() != "" {
                                content.push(json!({
                                    "type": "thinking",
                                    "thinking": reasoning_content,
                                    "signature": ""
                                }));
                            }
                        }
                        if let Some(text_content) = message.get("content") {
                            if !text_content.is_null() && text_content.as_str().unwrap_or("").trim() != "" {
                                content.push(json!({
                                    "type": "text",
                                    "text": text_content
                                }));
                            }
                        }

                        // 处理工具调用
                        if let Some(tool_calls) = message.get("tool_calls") {
                            if let Some(tool_calls_array) = tool_calls.as_array() {
                                for tool_call in tool_calls_array {
                                    content.push(json!({
                                        "type": "tool_use",
                                        "id": tool_call.get("id").unwrap_or(&json!("")),
                                        "name": tool_call.get("function").and_then(|f| f.get("name")).unwrap_or(&json!("")),
                                        "input": serde_json::from_str::<Value>(
                                            tool_call.get("function")
                                                .and_then(|f| f.get("arguments"))
                                                .and_then(|a| a.as_str())
                                                .unwrap_or("{}")
                                        ).unwrap_or(json!({}))
                                    }));
                                }
                            }
                        }
                    }

                    anthropic_response["content"] = json!(content);

                    // 处理停止原因
                    if let Some(finish_reason) = first_choice.get("finish_reason") {
                        anthropic_response["stop_reason"] = Self::map_openai_finish_reason_to_anthropic(finish_reason);
                    }
                }
            }
        }

        // 处理使用统计
        if let Some(usage) = openai_response.get("usage") {
            anthropic_response["usage"] = json!({
                "input_tokens": usage.get("prompt_tokens").unwrap_or(&json!(0)),
                "output_tokens": usage.get("completion_tokens").unwrap_or(&json!(0))
            });
        } else {
            anthropic_response["usage"] = json!({
                "input_tokens": 0,
                "output_tokens": 0
            });
        }

        anthropic_response
    }

    /// 将 Anthropic 响应转换为 OpenAI 响应
    pub fn anthropic_to_openai_response(anthropic_response: &Value) -> Value {
        let mut openai_response = json!({});

        // 处理基本字段
        if let Some(id) = anthropic_response.get("id") {
            openai_response["id"] = id.clone();
        } else {
            openai_response["id"] = json!("chatcmpl-".to_owned() + &uuid::Uuid::new_v4().to_string());
        }

        openai_response["object"] = json!("chat.completion");
        openai_response["created"] = json!(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());

        // 处理选择
        let mut message = json!({
            "role": "assistant"
        });

        let mut reasoning_text = String::new();
        let mut content_text = String::new();
        let mut tool_calls = Vec::new();

        if let Some(content) = anthropic_response.get("content") {
            if let Some(content_array) = content.as_array() {
                for item in content_array {
                    if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
                        match item_type {
                            "thinking" => {
                                if let Some(text) = item.get("thinking").and_then(|t| t.as_str()) {
                                    reasoning_text.push_str(text);
                                }
                            }
                            "redacted_thinking" => {
                                if let Some(text) = item.get("data").and_then(|t| t.as_str()) {
                                    let data = format!("'<redacted_thinking>{}</redacted_thinking>'", text.to_string());
                                    reasoning_text.push_str(&data);
                                }
                            }
                            "text" => {
                                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                    content_text.push_str(text);
                                }
                            }
                            "tool_use" => {
                                let tool_call = json!({
                                    "id": item.get("id").unwrap_or(&json!("")),
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name").unwrap_or(&json!("")),
                                        "arguments": serde_json::to_string(item.get("input").unwrap_or(&json!({})))
                                            .unwrap_or("{}".to_string())
                                    }
                                });
                                tool_calls.push(tool_call);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        message["reasoning_content"] = json!(reasoning_text);
        message["content"] = json!(content_text);
        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
        }

        let choice = json!({
            "index": 0,
            "message": message,
            "finish_reason": Self::map_anthropic_stop_reason_to_openai(
                anthropic_response.get("stop_reason")
            )
        });

        openai_response["choices"] = json!([choice]);

        // 处理使用统计
        if let Some(usage) = anthropic_response.get("usage") {
            openai_response["usage"] = json!({
                "prompt_tokens": usage.get("input_tokens").unwrap_or(&json!(0)),
                "completion_tokens": usage.get("output_tokens").unwrap_or(&json!(0)),
                "total_tokens": usage.get("input_tokens").unwrap_or(&json!(0)).as_u64().unwrap_or(0) +
                               usage.get("output_tokens").unwrap_or(&json!(0)).as_u64().unwrap_or(0)
            });
        }

        openai_response
    }

    /// 将 OpenAI 流式响应块转换为 Anthropic 流式响应块
    pub fn openai_to_anthropic_stream_chunk(chunk: &Value) -> Value {
        let mut anthropic_chunk = json!({});

        // 处理基本字段
        if let Some(id) = chunk.get("id") {
            anthropic_chunk["id"] = id.clone();
        }

        // 处理 choices
        if let Some(choices) = chunk.get("choices") {
            if let Some(choices_array) = choices.as_array() {
                if let Some(first_choice) = choices_array.first() {
                    
                    if let Some(usage) = chunk.get("usage") {
                        if let Some(prompt_tokens) = usage.get("prompt_tokens") {
                            if let Some(completion_tokens) = usage.get("completion_tokens") {
                                anthropic_chunk["usage"] = json!({
                                    "input_tokens": prompt_tokens,
                                    "output_tokens": completion_tokens
                                });
                            }
                        }
                    }

                    // 处理内容增量
                    if let Some(delta) = first_choice.get("delta") {
                        let mut delta_content = json!({});

                        // 处理文本内容
                        if let Some(reasoning_content) = delta.get("reasoning_content") {
                            if let Some(reasoning_str) = reasoning_content.as_str() {
                                if !reasoning_str.is_empty() {
                                    delta_content = json!({
                                        "type": "thinking_delta",
                                        "thinking": reasoning_str
                                    });
                                }
                            }
                        }

                        // 处理文本内容
                        if let Some(text_content) = delta.get("content") {
                            if let Some(text_str) = text_content.as_str() {
                                if !text_str.is_empty() {
                                    delta_content = json!({
                                        "type": "text_delta",
                                        "text": text_str
                                    });
                                }
                            }
                        }

                        // 处理工具调用增量
                        if let Some(tool_calls) = delta.get("tool_calls") {
                            if let Some(tool_calls_array) = tool_calls.as_array() {
                                for tool_call in tool_calls_array {
                                    if let Some(function) = tool_call.get("function") {
                                        delta_content = json!({"type": "input_json_delta"});
                                        if let Some(name) = function.get("name") {
                                            delta_content["name"] = json!(name);
                                        }
                                        if let Some(id) = tool_call.get("id") {
                                            delta_content["id"] = json!(id);
                                        }
                                        if let Some(arguments) = function.get("arguments") {
                                            if let Some(args_str) = arguments.as_str() {
                                                if !args_str.is_empty() {
                                                    delta_content["partial_json"] = json!(args_str);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if delta_content != json!({}) {
                            anthropic_chunk["type"] = json!("content_block_delta");
                            anthropic_chunk["delta"] = delta_content;
                            return anthropic_chunk;
                        }
                    }
                    

                    // 检查是否是停止消息
                    if let Some(finish_reason) = first_choice.get("finish_reason") {
                        if !finish_reason.is_null() {
                            anthropic_chunk["type"] = json!("message_delta");
                            anthropic_chunk["delta"] = json!({
                                "stop_reason": Self::map_openai_finish_reason_to_anthropic(finish_reason)
                            });
                            return anthropic_chunk;
                        }
                    }
                }
            }
        }

        // 默认返回 ping 心跳包
        json!({
            "type": "ping"
        })
    }

    /// 处理内容块开始的通用逻辑
    fn handle_content_block_start(chunk: &Value, delta_type: &str, msg_index: &i32, results: &mut Vec<(String, String)>, previous_delta_type: &mut String, json_copy: &mut Value) {
        let mut content_block_start = json!({"type": "content_block_start", "index": msg_index.clone()});
        
        if delta_type == "input_json_delta" {
            if let Some(delta) = chunk.get("delta") {
                if let Some(id) = delta.get("id") {
                    if let Some(id_str) = id.as_str() {
                        if let Some(name) = delta.get("name") {
                            if let Some(name_str) = name.as_str() {
                                if !id_str.is_empty() && !name_str.is_empty() {
                                    let tool_use_block = json!({"type":"content_block_start","index": msg_index.clone(), "content_block": {"type":"tool_use", "input": {}, "id": id, "name": name}});
                                    if let Ok(modified_data) = serde_json::to_string(&tool_use_block) {
                                        results.push(("content_block_start".to_string(), modified_data));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if let Some(obj) = json_copy.as_object_mut() {
                obj.remove("id");
                if let Some(delta) = chunk.get("delta") {
                    obj.insert("delta".to_string(), json!({"type":"input_json_delta","partial_json": delta.get("partial_json")}));
                }
            }
            previous_delta_type.clear();
            previous_delta_type.push_str("input_json_delta");
        } else if delta_type == "thinking_delta" {
            content_block_start["content_block"] = json!({"type": "thinking", "thinking": ""});
            if let Ok(modified_data) = serde_json::to_string(&content_block_start) {
                results.push(("content_block_start".to_string(), modified_data));
            }
            previous_delta_type.clear();
            previous_delta_type.push_str("thinking_delta");
        } else {
            content_block_start["content_block"] = json!({"type": "text", "text": ""});
            if let Ok(modified_data) = serde_json::to_string(&content_block_start) {
                results.push(("content_block_start".to_string(), modified_data));
            }
            previous_delta_type.clear();
            previous_delta_type.push_str("text_delta");
        }
    }

    /// 将 OpenAI 流式响应块转换为 Anthropic 流式响应块
    pub fn openai_to_anthropic_stream_chunks(chunk: &Value, model: &String, previous_event: &mut String, previous_delta_type: &mut String, msg_index: &mut i32) -> std::vec::Vec<(String, String)> {
        let mut results: std::vec::Vec<(String, String)> = vec![];
        if previous_event == "" {
            results.push(("message_start".to_string(), format!("{{\"type\": \"message_start\", \"message\": {{\"id\": \"{}\", \"type\": \"message\", \"role\": \"assistant\", \"content\": [], \"model\": \"{}\"}}}}", chunk.get("id").and_then(|s| s.as_str()).unwrap_or(""), model)));
            previous_event.clear();
            previous_event.push_str("message_start");
        }
        let (mut is_finish, mut is_reasoning_empty, mut is_content_empty, mut is_tool_calls_empty) =
            (false, true, true, true);

        if let Some(first_choice) = chunk.get("choices").and_then(|c| c.as_array()).and_then(|arr| arr.first()) {
            is_finish = first_choice.get("finish_reason").map_or(false, |v| !v.is_null());
            if let Some(delta) = first_choice.get("delta") {
                is_reasoning_empty = delta.get("reasoning_content").and_then(|v| v.as_str()).map_or(true, |s| s.is_empty());
                is_content_empty = delta.get("content").and_then(|v| v.as_str()).map_or(true, |s| s.is_empty());
                is_tool_calls_empty = delta.get("tool_calls").and_then(|v| v.as_array()).map_or(true, |arr| arr.is_empty());
            }
        }
        if is_finish {
            results.push(("content_block_stop".to_string(), format!("{{\"type\": \"content_block_stop\", \"index\": {}}}", msg_index.clone())));
        }
        if is_reasoning_empty && is_content_empty && is_tool_calls_empty && !is_finish {
            return results;
        }
        let mut chunk = Self::openai_to_anthropic_stream_chunk(&chunk);
        if let Some(obj) = chunk.as_object_mut() {
            if is_finish {
                obj.remove("index");
                obj.remove("id");
            } else {
                obj.insert("index".to_string(), json!(msg_index));
            }
        }
        let event_type = chunk.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let delta_type = chunk.get("delta").and_then(|t| t.get("type")).and_then(|t| t.as_str()).unwrap_or("");
        if previous_event == "message_start" {
            let mut json_copy = chunk.clone();
            Self::handle_content_block_start(&chunk, delta_type, msg_index, &mut results, previous_delta_type, &mut json_copy);
            if let Ok(modified_data) = serde_json::to_string(&json_copy) {
                results.push((event_type.to_string(), modified_data));
            }
            previous_event.clear();
            previous_event.push_str("content_block_delta");
            return results;
        } else if previous_event == "content_block_delta" {
            let content_keys = vec!["input_json_delta".to_string(), "thinking_delta".to_string(), "text_delta".to_string()];
            let mut json_copy = chunk.clone();
            if delta_type == previous_delta_type {
                if let Ok(modified_data) = serde_json::to_string(&json_copy) {
                    results.push((event_type.to_string(), modified_data));
                }
            } else if content_keys.contains(&delta_type.to_string()) {
                let stop_block = json!({"type": "content_block_stop", "index": msg_index.clone()});
                if let Ok(modified_data) = serde_json::to_string(&stop_block) {
                    results.push(("content_block_stop".to_string(), modified_data));
                }
                *msg_index += 1;
                if let Some(obj) = json_copy.as_object_mut() {
                    obj.insert("index".to_string(), json!(msg_index.clone()));
                }
                Self::handle_content_block_start(&chunk, delta_type, msg_index, &mut results, previous_delta_type, &mut json_copy);
                if let Ok(modified_data) = serde_json::to_string(&json_copy) {
                    results.push((event_type.to_string(), modified_data));
                }
            } else if event_type == "message_delta" {
                if let Ok(modified_data) = serde_json::to_string(&chunk) {
                    results.push((event_type.to_string(), modified_data));
                    results.push(("message_stop".to_string(), "{\"type\": \"message_stop\"}".to_string()));
                }
            }
            return results;
        }
        return results;
    }

    pub fn anthropic_to_openai_stream_chunk(chunk: &Value) -> Value {
        let mut openai_chunk = json!({});

        // 处理基本字段
        if let Some(id) = chunk.get("id") {
            openai_chunk["id"] = id.clone();
        } else if let Some(message) = chunk.get("message") {
            if let Some(msg_id) = message.get("id") {
                openai_chunk["id"] = msg_id.clone();
            }
        }

        if openai_chunk.get("id").is_none() {
            openai_chunk["id"] = json!("chatcmpl-".to_owned() + &uuid::Uuid::new_v4().to_string());
        }

        openai_chunk["object"] = json!("chat.completion.chunk");
        openai_chunk["created"] = json!(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());

        let mut delta = json!({});
        let mut finish_reason = Value::Null;

        // 根据 chunk 类型处理
        if let Some(chunk_type) = chunk.get("type").and_then(|t| t.as_str()) {
            match chunk_type {
                "message_start" => {
                    // 消息开始，设置角色
                    delta["role"] = json!("assistant");
                    delta["content"] = json!("");
                    delta["reasoning_content"] = json!("");
                }
                "content_block_start" => {
                    // 内容块开始
                    if let Some(content_block) = chunk.get("content_block") {
                        if let Some(block_type) = content_block.get("type").and_then(|t| t.as_str()) {
                            match block_type {
                                "text" => {
                                    // 文本块开始，通常不需要特殊处理
                                    delta["content"] = json!("");
                                }
                                "tool_use" => {
                                    // 工具使用块开始
                                    if let Some(index) = chunk.get("index") {
                                        let tool_call = json!({
                                            "index": index,
                                            "id": content_block.get("id").unwrap_or(&json!("")),
                                            "type": "function",
                                            "function": {
                                                "name": content_block.get("name").unwrap_or(&json!("")),
                                                "arguments": ""
                                            }
                                        });
                                        delta["tool_calls"] = json!([tool_call]);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "content_block_delta" => {
                    // 处理内容块增量
                    if let Some(chunk_delta) = chunk.get("delta") {
                        if let Some(delta_type) = chunk_delta.get("type").and_then(|t| t.as_str()) {
                            match delta_type {
                                "thinking_delta" => {
                                    if let Some(text) = chunk_delta.get("thinking").and_then(|t| t.as_str()) {
                                        delta["reasoning_content"] = json!(text);
                                    }
                                }
                                "text_delta" => {
                                    if let Some(text) = chunk_delta.get("text").and_then(|t| t.as_str()) {
                                        delta["content"] = json!(text);
                                    }
                                }
                                "input_json_delta" => {
                                    // 工具参数增量
                                    if let Some(partial_json) = chunk_delta.get("partial_json") {
                                        if let Some(index) = chunk.get("index") {
                                            let tool_call = json!({
                                                "index": index,
                                                "function": {
                                                    "arguments": partial_json
                                                }
                                            });
                                            delta["tool_calls"] = json!([tool_call]);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "content_block_stop" => {
                    // 内容块结束，通常不需要特殊处理
                    delta = json!({});
                }
                "message_delta" => {
                    // 处理消息级增量，主要是停止原因
                    if let Some(chunk_delta) = chunk.get("delta") {
                        if let Some(stop_reason) = chunk_delta.get("stop_reason") {
                            finish_reason = Self::map_anthropic_stop_reason_to_openai(Some(stop_reason));
                        }
                    }
                }
                "message_stop" => {
                    // 消息结束
                    finish_reason = json!("stop");
                    delta = json!({});
                }
                "ping" => {
                    // 心跳包，返回空的增量
                    delta = json!({});
                }
                _ => {
                    delta = json!({});
                }
            }
        }

        let choice = json!({
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        });

        openai_chunk["choices"] = json!([choice]);

        openai_chunk
    }
    
    // 消息转换方法
    fn convert_openai_messages_to_anthropic(messages: &Vec<Value>) -> Value {
        let mut anthropic_messages = Vec::new();

        for message in messages {
            if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
                if role == "system" {
                    continue; // 系统消息单独处理
                }

                let mut anthropic_message = json!({
                    "role": if role == "assistant" { "assistant" } else { "user" }
                });

                let mut content = Vec::new();

                // 处理内容
                if let Some(msg_content) = message.get("content") {
                    if let Some(content_str) = msg_content.as_str() {
                        // 文本内容
                        content.push(json!({
                            "type": "text",
                            "text": content_str
                        }));
                    } else if let Some(content_array) = msg_content.as_array() {
                        // 多模态内容
                        for item in content_array {
                            if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
                                match item_type {
                                    "text" => {
                                        content.push(json!({
                                            "type": "text",
                                            "text": item.get("text").unwrap_or(&json!(""))
                                        }));
                                    }
                                    "image_url" => {
                                        if let Some(image_url) = item.get("image_url") {
                                            if let Some(url) = image_url.get("url").and_then(|u| u.as_str()) {
                                                if url.starts_with("data:") {
                                                    // 处理 base64 图片
                                                    let parts: Vec<&str> = url.split(',').collect();
                                                    if parts.len() == 2 {
                                                        let media_type = parts[0]
                                                            .replace("data:", "")
                                                            .replace(";base64", "");
                                                        content.push(json!({
                                                            "type": "image",
                                                            "source": {
                                                                "type": "base64",
                                                                "media_type": media_type,
                                                                "data": parts[1]
                                                            }
                                                        }));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                // 处理工具调用结果
                if let Some(tool_call_id) = message.get("tool_call_id") {
                    anthropic_message["role"] = json!("user");
                    content.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": message.get("content").unwrap_or(&json!(""))
                    }));
                }

                if !content.is_empty() {
                    anthropic_message["content"] = json!(content);
                    anthropic_messages.push(anthropic_message);
                }
            }
        }

        json!(anthropic_messages)
    }

    fn convert_anthropic_messages_to_openai(messages: &Vec<Value>) -> Vec<Value> {
        let mut openai_messages = Vec::new();

        for message in messages {
            if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
                let mut openai_message = json!({
                    "role": role
                });

                if let Some(content) = message.get("content") {
                    if let Some(content_array) = content.as_array() {
                        let mut text_content = String::new();
                        let mut tool_calls = Vec::new();
                        let mut multimodal_content = Vec::new();
                        let mut has_non_text = false;

                        for item in content_array {
                            if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
                                match item_type {
                                    "text" => {
                                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                            text_content.push_str(text);
                                            multimodal_content.push(json!({
                                                "type": "text",
                                                "text": text
                                            }));
                                        }
                                    }
                                    "image" => {
                                        has_non_text = true;
                                        if let Some(source) = item.get("source") {
                                            if let Some(media_type) = source.get("media_type").and_then(|m| m.as_str()) {
                                                if let Some(data) = source.get("data").and_then(|d| d.as_str()) {
                                                    multimodal_content.push(json!({
                                                        "type": "image_url",
                                                        "image_url": {
                                                            "url": format!("data:{};base64,{}", media_type, data)
                                                        }
                                                    }));
                                                }
                                            }
                                        }
                                    }
                                    "tool_use" => {
                                        let tool_call = json!({
                                            "id": item.get("id").unwrap_or(&json!("")),
                                            "type": "function",
                                            "function": {
                                                "name": item.get("name").unwrap_or(&json!("")),
                                                "arguments": serde_json::to_string(item.get("input").unwrap_or(&json!({})))
                                                    .unwrap_or("{}".to_string())
                                            }
                                        });
                                        tool_calls.push(tool_call);
                                    }
                                    "tool_result" => {
                                        text_content.push_str(item.get("content").and_then(|t| t.as_str()).unwrap_or(""));
                                        openai_message["role"] = json!("tool");
                                        openai_message["tool_call_id"] = item.get("tool_use_id").unwrap_or(&json!("")).clone();
                                    }
                                    _ => {}
                                }
                            }
                        }

                        if has_non_text {
                            openai_message["content"] = json!(multimodal_content);
                        } else {
                            openai_message["content"] = json!(text_content);
                        }

                        if !tool_calls.is_empty() {
                            openai_message["tool_calls"] = json!(tool_calls);
                        }
                    } else {
                        openai_message["content"] = content.clone();
                    }
                }

                openai_messages.push(openai_message);
            }
        }

        openai_messages
    }

    // 工具转换方法
    fn convert_openai_tools_to_anthropic(tools: &Value) -> Value {
        let mut anthropic_tools = Vec::new();

        if let Some(tools_array) = tools.as_array() {
            for tool in tools_array {
                if let Some(function) = tool.get("function") {
                    anthropic_tools.push(json!({
                        "name": function.get("name").unwrap_or(&json!("")),
                        "description": function.get("description").unwrap_or(&json!("")),
                        "input_schema": function.get("parameters").unwrap_or(&json!({}))
                    }));
                }
            }
        }

        json!(anthropic_tools)
    }

    fn convert_anthropic_tools_to_openai(tools: &Value) -> Value {
        let mut openai_tools = Vec::new();

        if let Some(tools_array) = tools.as_array() {
            for tool in tools_array {
                openai_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").unwrap_or(&json!("")),
                        "description": tool.get("description").unwrap_or(&json!("")),
                        "parameters": tool.get("input_schema").unwrap_or(&json!({}))
                    }
                }));
            }
        }

        json!(openai_tools)
    }

    // 系统消息提取
    fn extract_system_message(messages: &Vec<Value>) -> Option<Value> {
        for message in messages {
            if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
                if role == "system" {
                    return Some(message.get("content").unwrap_or(&json!("")).clone());
                }
            }
        }
        None
    }

    // 停止原因映射
    fn map_openai_finish_reason_to_anthropic(finish_reason: &Value) -> Value {
        match finish_reason.as_str().unwrap_or("") {
            "stop" => json!("end_turn"),
            "length" => json!("max_tokens"),
            "tool_calls" => json!("tool_use"),
            "content_filter" => json!("stop_sequence"),
            _ => json!("end_turn")
        }
    }

    fn map_anthropic_stop_reason_to_openai(stop_reason: Option<&Value>) -> Value {
        match stop_reason.and_then(|s| s.as_str()).unwrap_or("") {
            "end_turn" => json!("stop"),
            "max_tokens" => json!("length"),
            "tool_use" => json!("tool_calls"),
            "stop_sequence" => json!("stop"),
            _ => json!("stop")
        }
    }
}



#[cfg(test)]
mod tests {
    use regex::Regex;
    use super::*;

    #[test]
    fn test_openai_to_anthropic_request() {
        let openai_request = json!({
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello!"
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "tools": [
                {
                    "function": {
                        "description": "Adds two numbers together.",
                        "name": "add",
                        "parameters": {
                            "properties": {
                                "a": {
                                    "type": "integer"
                                },
                                "b": {
                                    "type": "integer"
                                }
                            },
                            "required": [
                                "a",
                                "b"
                            ],
                            "type": "object"
                        }
                    },
                    "type": "function"
                }
            ]
        });

        let anthropic_request = ApiConverter::openai_to_anthropic_request(&openai_request);
        
        assert_eq!(anthropic_request["model"], "gpt-4");
        assert_eq!(anthropic_request["max_tokens"], 100);
        assert_eq!(anthropic_request["system"], "You are a helpful assistant.");
        assert_eq!(anthropic_request["temperature"], 0.7);
        assert_eq!(anthropic_request["tools"][0]["description"], "Adds two numbers together.");
        assert_eq!(anthropic_request["tools"][0]["input_schema"]["properties"]["a"]["type"], "integer");
    }

    #[test]
    fn test_anthropic_to_openai_request() {
        let anthropic_request = json!({
            "model": "claude-opus-4-1-20250805",
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in San Francisco?"
                }
            ]
        });

        let openai_request = ApiConverter::anthropic_to_openai_request(&anthropic_request);
        
        assert_eq!(openai_request["model"], "claude-opus-4-1-20250805");
        assert_eq!(openai_request["max_tokens"], 1024);
        
        let messages = openai_request["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "What is the weather like in San Francisco?");
        assert_eq!(openai_request["tools"][0]["function"]["description"], "Get the current weather in a given location");
        assert_eq!(openai_request["tools"][0]["function"]["parameters"]["properties"]["location"]["type"], "string");
    }

    #[test]
    fn test_openai_to_anthropic_response() {
        // 测试基本的文本响应
        let openai_response = json!({
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

        let anthropic_response = ApiConverter::openai_to_anthropic_response(&openai_response);
        
        assert_eq!(anthropic_response["id"], "chatcmpl-123");
        assert_eq!(anthropic_response["type"], "message");
        assert_eq!(anthropic_response["role"], "assistant");
        assert_eq!(anthropic_response["content"][0]["type"], "text");
        assert_eq!(anthropic_response["content"][0]["text"], "Hello, how can I help you today?");
        assert_eq!(anthropic_response["stop_reason"], "end_turn");
        assert_eq!(anthropic_response["usage"]["input_tokens"], 9);
        assert_eq!(anthropic_response["usage"]["output_tokens"], 12);
    }

    #[test]
    fn test_openai_to_anthropic_response_with_reasoning() {
        // 测试包含推理内容的响应
        let openai_response = json!({
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

        let anthropic_response = ApiConverter::openai_to_anthropic_response(&openai_response);
        
        assert_eq!(anthropic_response["id"], "chatcmpl-456");
        assert_eq!(anthropic_response["type"], "message");
        assert_eq!(anthropic_response["role"], "assistant");
        assert_eq!(anthropic_response["content"][0]["type"], "thinking");
        assert_eq!(anthropic_response["content"][0]["thinking"], "I need to think about this step by step.");
        assert_eq!(anthropic_response["content"][1]["type"], "text");
        assert_eq!(anthropic_response["content"][1]["text"], "The answer is 42.");
        assert_eq!(anthropic_response["stop_reason"], "end_turn");
        assert_eq!(anthropic_response["usage"]["input_tokens"], 15);
        assert_eq!(anthropic_response["usage"]["output_tokens"], 20);
    }

    #[test]
    fn test_openai_to_anthropic_response_with_tool_calls() {
        // 测试包含工具调用的响应
        let openai_response = json!({
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

        let anthropic_response = ApiConverter::openai_to_anthropic_response(&openai_response);
        
        assert_eq!(anthropic_response["id"], "chatcmpl-789");
        assert_eq!(anthropic_response["type"], "message");
        assert_eq!(anthropic_response["role"], "assistant");
        assert_eq!(anthropic_response["content"][0]["type"], "text");
        assert_eq!(anthropic_response["content"][0]["text"], "I'll help you get the weather.");
        assert_eq!(anthropic_response["content"][1]["type"], "tool_use");
        assert_eq!(anthropic_response["content"][1]["id"], "call_abc123");
        assert_eq!(anthropic_response["content"][1]["name"], "get_weather");
        assert_eq!(anthropic_response["content"][1]["input"]["location"], "San Francisco, CA");
        assert_eq!(anthropic_response["stop_reason"], "tool_use");
        assert_eq!(anthropic_response["usage"]["input_tokens"], 25);
        assert_eq!(anthropic_response["usage"]["output_tokens"], 30);
    }

    #[test]
    fn test_openai_to_anthropic_response_empty_content() {
        // 测试空内容的响应
        let openai_response = json!({
            "id": "chatcmpl-empty",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ""
                    },
                    "finish_reason": "stop"
                }
            ]
        });

        let anthropic_response = ApiConverter::openai_to_anthropic_response(&openai_response);
        
        assert_eq!(anthropic_response["id"], "chatcmpl-empty");
        assert_eq!(anthropic_response["type"], "message");
        assert_eq!(anthropic_response["role"], "assistant");
        // 空内容不应该被添加到 content 数组中
        assert!(anthropic_response["content"].as_array().unwrap().is_empty());
        assert_eq!(anthropic_response["stop_reason"], "end_turn");
    }

    #[test]
    fn test_openai_to_anthropic_response_null_content() {
        // 测试 null 内容的响应
        let openai_response = json!({
            "id": "chatcmpl-null",
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

        let anthropic_response = ApiConverter::openai_to_anthropic_response(&openai_response);
        
        assert_eq!(anthropic_response["id"], "chatcmpl-null");
        assert_eq!(anthropic_response["type"], "message");
        assert_eq!(anthropic_response["role"], "assistant");
        // null 内容不应该被添加到 content 数组中
        assert!(anthropic_response["content"].as_array().unwrap().is_empty());
        assert_eq!(anthropic_response["stop_reason"], "end_turn");
    }

    #[test]
    fn test_openai_to_anthropic_response_max_tokens_finish_reason() {
        // 测试 max_tokens 停止原因
        let openai_response = json!({
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

        let anthropic_response = ApiConverter::openai_to_anthropic_response(&openai_response);
        
        assert_eq!(anthropic_response["id"], "chatcmpl-max");
        assert_eq!(anthropic_response["type"], "message");
        assert_eq!(anthropic_response["role"], "assistant");
        assert_eq!(anthropic_response["content"][0]["type"], "text");
        assert_eq!(anthropic_response["content"][0]["text"], "This is a truncated response because");
        assert_eq!(anthropic_response["stop_reason"], "max_tokens");
    }

    #[test]
    fn test_openai_to_anthropic_response_no_usage() {
        // 测试没有使用统计的响应
        let openai_response = json!({
            "id": "chatcmpl-nousage",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello world"
                    },
                    "finish_reason": "stop"
                }
            ]
        });

        let anthropic_response = ApiConverter::openai_to_anthropic_response(&openai_response);
        
        assert_eq!(anthropic_response["id"], "chatcmpl-nousage");
        assert_eq!(anthropic_response["type"], "message");
        assert_eq!(anthropic_response["role"], "assistant");
        assert_eq!(anthropic_response["content"][0]["type"], "text");
        assert_eq!(anthropic_response["content"][0]["text"], "Hello world");
        assert_eq!(anthropic_response["stop_reason"], "end_turn");
        // 默认使用统计应该为 0
        assert_eq!(anthropic_response["usage"]["input_tokens"], 0);
        assert_eq!(anthropic_response["usage"]["output_tokens"], 0);
    }


    #[test]
    fn test_anthropic_to_openai_response() {
        // 测试基本的文本响应
        let anthropic_response = json!({
            "id": "msg_123",
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

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "Hello, how can I help you today?");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "stop");
        assert_eq!(openai_response["usage"]["prompt_tokens"], 9);
        assert_eq!(openai_response["usage"]["completion_tokens"], 12);
        assert_eq!(openai_response["usage"]["total_tokens"], 21);
    }

    #[test]
    fn test_anthropic_to_openai_response_with_thinking() {
        // 测试包含推理内容的响应
        let anthropic_response = json!({
            "id": "msg_456",
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

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "The answer is 42.");
        assert_eq!(openai_response["choices"][0]["message"]["reasoning_content"], "I need to think about this step by step.");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "stop");
        assert_eq!(openai_response["usage"]["prompt_tokens"], 15);
        assert_eq!(openai_response["usage"]["completion_tokens"], 20);
        assert_eq!(openai_response["usage"]["total_tokens"], 35);
    }

    #[test]
    fn test_anthropic_to_openai_response_with_tool_calls() {
        // 测试包含工具调用的响应
        let anthropic_response = json!({
            "id": "msg_789",
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

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "I'll help you get the weather.");
        assert_eq!(openai_response["choices"][0]["message"]["tool_calls"][0]["id"], "tool_123");
        assert_eq!(openai_response["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(openai_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"location\":\"San Francisco, CA\"}");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(openai_response["usage"]["prompt_tokens"], 25);
        assert_eq!(openai_response["usage"]["completion_tokens"], 30);
        assert_eq!(openai_response["usage"]["total_tokens"], 55);
    }

    #[test]
    fn test_anthropic_to_openai_response_empty_content() {
        // 测试空内容的响应
        let anthropic_response = json!({
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn"
        });

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_anthropic_to_openai_response_no_id() {
        // 测试没有 ID 的响应
        let anthropic_response = json!({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello world"
                }
            ],
            "stop_reason": "end_turn"
        });

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "stop");
        // 应该生成一个 ID
        assert!(openai_response["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }

    #[test]
    fn test_anthropic_to_openai_response_with_redacted_thinking() {
        // 测试包含编辑后推理内容的响应
        let anthropic_response = json!({
            "id": "msg_redacted",
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

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "I can't share that information.");
        assert_eq!(openai_response["choices"][0]["message"]["reasoning_content"], "'<redacted_thinking>sensitive information</redacted_thinking>'");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_anthropic_to_openai_response_max_tokens_stop_reason() {
        // 测试 max_tokens 停止原因
        let anthropic_response = json!({
            "id": "msg_max",
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

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "This is a truncated response because");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn test_anthropic_to_openai_response_no_usage() {
        // 测试没有使用统计的响应
        let anthropic_response = json!({
            "id": "msg_nousage",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello world"
                }
            ],
            "stop_reason": "end_turn"
        });

        let openai_response = ApiConverter::anthropic_to_openai_response(&anthropic_response);
        
        assert_eq!(openai_response["object"], "chat.completion");
        assert_eq!(openai_response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(openai_response["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(openai_response["choices"][0]["finish_reason"], "stop");
        // 没有使用统计字段
        assert!(!openai_response.as_object().unwrap().contains_key("usage"));
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

        let anthropic_chunk = ApiConverter::openai_to_anthropic_stream_chunk(&openai_chunk);
        
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

        let anthropic_chunk = ApiConverter::openai_to_anthropic_stream_chunk(&openai_chunk);
        
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

        let anthropic_chunk = ApiConverter::openai_to_anthropic_stream_chunk(&openai_chunk);
        
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

        let anthropic_chunk = ApiConverter::openai_to_anthropic_stream_chunk(&openai_chunk);
        
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
                        "content": "Hello"
                    },
                    "finish_reason": null
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        });

        let anthropic_chunk = ApiConverter::openai_to_anthropic_stream_chunk(&openai_chunk);
        
        assert_eq!(anthropic_chunk["type"], "content_block_delta");
        assert_eq!(anthropic_chunk["delta"]["type"], "text_delta");
        assert_eq!(anthropic_chunk["delta"]["text"], "Hello");
        assert_eq!(anthropic_chunk["usage"]["input_tokens"], 10);
        assert_eq!(anthropic_chunk["usage"]["output_tokens"], 5);
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

        let anthropic_chunk = ApiConverter::openai_to_anthropic_stream_chunk(&openai_chunk);
        
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

        let anthropic_chunk = ApiConverter::openai_to_anthropic_stream_chunk(&openai_chunk);
        
        // 没有 choices 应该返回 ping 心跳包
        assert_eq!(anthropic_chunk["type"], "ping");
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_start() {
        // 测试 message_start 类型的 Anthropic 流式响应块
        let anthropic_chunk = json!({
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-opus-20240229"
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
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
        let anthropic_chunk = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"]["content"], "");
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_start_tool_use() {
        // 测试 content_block_start 类型的 Anthropic 流式响应块（工具调用）
        let anthropic_chunk = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "tool_123",
                "name": "get_weather",
                "input": {}
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(openai_chunk["choices"][0]["delta"]["tool_calls"][0]["id"], "tool_123");
        assert_eq!(openai_chunk["choices"][0]["delta"]["tool_calls"][0]["type"], "function");
        assert_eq!(openai_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(openai_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "");
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_delta_thinking() {
        // 测试 content_block_delta 类型的 Anthropic 流式响应块（推理内容）
        let anthropic_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "I need to think about this step by step."
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"]["reasoning_content"], "I need to think about this step by step.");
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_delta_text() {
        // 测试 content_block_delta 类型的 Anthropic 流式响应块（文本内容）
        let anthropic_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello, how can I help you today?"
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"]["content"], "Hello, how can I help you today?");
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_delta_tool_use() {
        // 测试 content_block_delta 类型的 Anthropic 流式响应块（工具调用参数）
        let anthropic_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": "{\"location\": \"San Francisco, CA\"}"
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(openai_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "{\"location\": \"San Francisco, CA\"}");
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_content_block_stop() {
        // 测试 content_block_stop 类型的 Anthropic 流式响应块
        let anthropic_chunk = json!({
            "type": "content_block_stop",
            "index": 0
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_delta() {
        // 测试 message_delta 类型的 Anthropic 流式响应块
        let anthropic_chunk = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn"
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_stop() {
        // 测试 message_stop 类型的 Anthropic 流式响应块
        let anthropic_chunk = json!({
            "type": "message_stop"
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_ping() {
        // 测试 ping 类型的 Anthropic 流式响应块
        let anthropic_chunk = json!({
            "type": "ping"
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_no_id() {
        // 测试没有 ID 的 Anthropic 流式响应块
        let anthropic_chunk = json!({
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-opus-20240229"
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        // 应该生成一个 ID
        assert!(openai_chunk["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_unknown_type() {
        // 测试未知类型的 Anthropic 流式响应块
        let anthropic_chunk = json!({
            "type": "unknown_type"
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_delta_max_tokens() {
        // 测试 message_delta 类型的 Anthropic 流式响应块（max_tokens 停止原因）
        let anthropic_chunk = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "max_tokens"
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn test_anthropic_to_openai_stream_chunk_message_delta_tool_use() {
        // 测试 message_delta 类型的 Anthropic 流式响应块（tool_use 停止原因）
        let anthropic_chunk = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use"
            }
        });

        let openai_chunk = ApiConverter::anthropic_to_openai_stream_chunk(&anthropic_chunk);
        
        assert_eq!(openai_chunk["object"], "chat.completion.chunk");
        assert_eq!(openai_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(openai_chunk["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn test_openai_to_anthropic_stream_chunks_message_start() {
        // 测试初始消息开始的情况
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
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = ApiConverter::openai_to_anthropic_stream_chunks(
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
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = ApiConverter::openai_to_anthropic_stream_chunks(
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
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = ApiConverter::openai_to_anthropic_stream_chunks(
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
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "content_block_delta".to_string();
        let mut previous_delta_type = "text_delta".to_string();
        let mut msg_index = 0;
        
        let results = ApiConverter::openai_to_anthropic_stream_chunks(
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
        let openai_chunk = json!({
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
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "content_block_delta".to_string();
        let mut previous_delta_type = "thinking_delta".to_string(); // 前一个是推理内容
        let mut msg_index = 0;
        
        let results = ApiConverter::openai_to_anthropic_stream_chunks(
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
        let openai_chunk = json!({
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
        
        let model = "claude-3-opus".to_string();
        let mut previous_event = "".to_string();
        let mut previous_delta_type = "".to_string();
        let mut msg_index = 0;
        
        let results = ApiConverter::openai_to_anthropic_stream_chunks(
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