use crate::converters::openai::openai_completion_tokens_details::OpenAICompletionTokensDetails;
use crate::converters::openai::openai_prompt_tokens_details::OpenAIPromptTokensDetails;
use crate::converters::openai::{OpenAIChoice, OpenAIResponse, OpenAIResponseMessage, OpenAIUsage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::responses_content::{ResponsesContentPart, ResponsesReasoningSummary};
use super::responses_output_item::ResponsesOutputItem;
use super::responses_reasoning_info::ResponsesReasoningInfo;
use super::responses_text_config::ResponsesTextConfig;
use super::responses_tool::ResponsesTool;
use super::responses_tool_choice::ResponsesToolChoice;
use super::responses_usage::ResponsesUsage;
use super::responses_usage_detail::ResponsesUsageDetail;

/// Response payload returned by the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(rename = "created_at")]
    pub created_at: u64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    pub model: String,
    #[serde(default)]
    pub output: Vec<ResponsesOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoningInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
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
    pub usage: Option<ResponsesUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
    #[serde(flatten)]
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

impl From<OpenAIResponse> for ResponsesResponse {
    fn from(openai: OpenAIResponse) -> Self {
        let OpenAIResponse {
            id,
            object,
            created,
            model,
            choices,
            usage,
            system_fingerprint,
            service_tier,
        } = openai;

        let mut output_items = Vec::new();
        for (idx, choice) in choices.into_iter().enumerate() {
            output_items.push(choice_to_output_item(idx, choice));
        }

        let status = output_items
            .iter()
            .find_map(|item| match item {
                ResponsesOutputItem::Message { status, .. } => Some(status.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "completed".to_string());

        let mut extra = HashMap::new();
        if let Some(fingerprint) = system_fingerprint {
            extra.insert("system_fingerprint".to_string(), Value::String(fingerprint));
        }
        if let Some(tier) = service_tier {
            extra.insert("service_tier".to_string(), Value::String(tier));
        }

        ResponsesResponse {
            id,
            object,
            created_at: created,
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model,
            output: output_items,
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
            usage: usage.map(convert_usage_from_openai),
            user: None,
            metadata: None,
            extra,
        }
    }
}

fn choice_to_output_item(index: usize, choice: OpenAIChoice) -> ResponsesOutputItem {
    let OpenAIChoice {
        message,
        finish_reason,
        ..
    } = choice;

    let OpenAIResponseMessage {
        role,
        content,
        reasoning_content,
        tool_calls,
    } = message;

    let mut parts: Vec<ResponsesContentPart> = Vec::new();

    if let Some(reasoning) = reasoning_content {
        parts.push(ResponsesContentPart::Reasoning {
            id: format!("reasoning-{}", index),
            summary: vec![ResponsesReasoningSummary::SummaryText {
                text: reasoning,
                extra: HashMap::new(),
            }],
            encrypted_content: None,
            status: None,
            extra: HashMap::new(),
        });
    }

    if let Some(text) = content {
        parts.push(ResponsesContentPart::OutputText {
            text,
            annotations: Vec::new(),
            extra: HashMap::new(),
        });
    }

    if let Some(calls) = tool_calls {
        for call in calls {
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

    ResponsesOutputItem::Message {
        id: format!("choice-{}", index),
        status: map_finish_reason_to_message_status(&finish_reason),
        role,
        content: parts,
        metadata: None,
        name: None,
        extra: HashMap::new(),
    }
}

fn map_finish_reason_to_message_status(finish_reason: &str) -> String {
    match finish_reason {
        "tool_calls" => "requires_action".to_string(),
        "length" => "incomplete".to_string(),
        "cancelled" => "cancelled".to_string(),
        "failed" => "failed".to_string(),
        _ => "completed".to_string(),
    }
}

fn convert_usage_from_openai(usage: OpenAIUsage) -> ResponsesUsage {
    ResponsesUsage {
        input_tokens: usage.prompt_tokens as u64,
        input_tokens_details: usage
            .prompt_tokens_details
            .map(convert_prompt_detail_from_openai),
        output_tokens: usage.completion_tokens as u64,
        output_tokens_details: usage
            .completion_tokens_details
            .map(convert_completion_detail_from_openai),
        total_tokens: usage.total_tokens as u64,
        extra: HashMap::new(),
    }
}

fn convert_prompt_detail_from_openai(detail: OpenAIPromptTokensDetails) -> ResponsesUsageDetail {
    let mut extra = HashMap::new();
    if let Some(audio) = detail.audio_tokens {
        extra.insert("audio_tokens".to_string(), Value::from(audio));
    }

    ResponsesUsageDetail {
        cached_tokens: detail.cached_tokens.map(|v| v as u64),
        reasoning_tokens: None,
        extra,
    }
}

fn convert_completion_detail_from_openai(
    detail: OpenAICompletionTokensDetails,
) -> ResponsesUsageDetail {
    let mut extra = HashMap::new();
    if let Some(audio) = detail.audio_tokens {
        extra.insert("audio_tokens".to_string(), Value::from(audio));
    }
    if let Some(accepted) = detail.accepted_prediction_tokens {
        extra.insert(
            "accepted_prediction_tokens".to_string(),
            Value::from(accepted),
        );
    }
    if let Some(rejected) = detail.rejected_prediction_tokens {
        extra.insert(
            "rejected_prediction_tokens".to_string(),
            Value::from(rejected),
        );
    }

    ResponsesUsageDetail {
        cached_tokens: None,
        reasoning_tokens: detail.reasoning_tokens.map(|v| v as u64),
        extra,
    }
}
