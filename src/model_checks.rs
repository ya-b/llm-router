use std::sync::Arc;
use crate::config::Config;

pub async fn perform_model_checks(
    config: &Arc<Config>,
    llm_client: &Arc<crate::llm_client::LlmClient>,
) -> anyhow::Result<()> {
    use crate::config::ApiType;
    use crate::converters::request_wrapper::RequestWrapper;
    use crate::converters::openai::{OpenAIRequest, OpenAIMessage, OpenAIContent};
    use crate::converters::anthropic::{AnthropicRequest, AnthropicMessage, AnthropicContent};
    use crate::converters::gemini::{GeminiRequest, gemini_content::GeminiContent, gemini_part::GeminiPart, gemini_generation_config::GeminiGenerationConfig};
    use futures::stream::{self, StreamExt};

    println!("Checking models ({} total):", config.model_list.len());
    let concurrency: usize = 20;
    let client = llm_client.clone();
    let tasks = stream::iter(config.model_list.iter().cloned()).map(|mc| {
        let client = client.clone();
        async move {
            let request = match mc.llm_params.api_type {
                ApiType::OpenAI => {
                    let req = OpenAIRequest {
                        model: mc.model_name.clone(),
                        messages: vec![OpenAIMessage {
                            role: "user".to_string(),
                            content: OpenAIContent::Text("ping".to_string()),
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        }],
                        max_tokens: Some(1),
                        temperature: Some(0.0),
                        response_format: None,
                        tools: None,
                        stream: Some(false),
                        extra_fields: std::collections::HashMap::new(),
                    };
                    RequestWrapper::OpenAI(req)
                }
                ApiType::Anthropic => {
                    let req = AnthropicRequest {
                        model: mc.model_name.clone(),
                        max_tokens: 1,
                        messages: Some(vec![AnthropicMessage { role: "user".to_string(), content: AnthropicContent::Text("ping".to_string()) }]),
                        system: None,
                        tools: None,
                        metadata: None,
                        stream: Some(false),
                        temperature: Some(0.0),
                        extra_fields: std::collections::HashMap::new(),
                    };
                    RequestWrapper::Anthropic(req)
                }
                ApiType::Gemini => {
                    let req = GeminiRequest {
                        model: mc.model_name.clone(),
                        contents: vec![GeminiContent { role: Some("user".to_string()), parts: vec![GeminiPart::Text { text: "ping".to_string(), thought: None, thought_signature: None }] }],
                        system_instruction: None,
                        tools: None,
                        generation_config: Some(GeminiGenerationConfig { response_mime_type: None, response_schema: None, temperature: Some(0.0), max_output_tokens: Some(1), ..Default::default() }),
                        stream: Some(false),
                        extra_fields: std::collections::HashMap::new(),
                    };
                    RequestWrapper::Gemini(req)
                }
            };

            let req_id = crate::request_id::RequestId(uuid::Uuid::new_v4().to_string());
            let result = client.forward_request(&request, &mc, &req_id).await;
            match result {
                Ok(resp) => {
                    if resp.status().is_success() {
                        println!(
                            "[OK] {} -> {} ({})",
                            mc.model_name,
                            mc.llm_params.model,
                            match mc.llm_params.api_type { ApiType::OpenAI => "openai", ApiType::Anthropic => "anthropic", ApiType::Gemini => "gemini" }
                        );
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
                        println!(
                            "[FAIL] {} -> {} (status: {})\n  {}",
                            mc.model_name, mc.llm_params.model, status, truncate(&body, 500)
                        );
                    }
                }
                Err(e) => {
                    println!(
                        "[ERROR] {} -> {}: {}",
                        mc.model_name, mc.llm_params.model, e
                    );
                }
            }
        }
    })
    .buffer_unordered(concurrency)
    .collect::<Vec<()>>();

    tasks.await;
    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        match s.char_indices().nth(max_len) {
            None => s.to_string(),
            Some((idx, _)) => format!("{}…", &s[..idx]),
        }
    }
}

