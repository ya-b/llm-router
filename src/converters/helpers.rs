use serde_json::{Value, json};

// 停止原因映射
pub fn map_openai_finish_reason_to_anthropic(finish_reason: &Value) -> Value {
    match finish_reason.as_str() {
        Some("stop") => json!("end_turn"),
        Some("length") => json!("max_tokens"),
        Some("tool_calls") => json!("tool_use"),
        Some("content_filter") => json!("stop_sequence"),
        _ => json!("end_turn"),
    }
}

pub fn map_anthropic_stop_reason_to_openai(stop_reason: Option<&Value>) -> Value {
    match stop_reason.and_then(|s| s.as_str()) {
        Some("end_turn") => json!("stop"),
        Some("max_tokens") => json!("length"),
        Some("tool_use") => json!("tool_calls"),
        Some("stop_sequence") => json!("stop"),
        _ => json!("stop"),
    }
}
