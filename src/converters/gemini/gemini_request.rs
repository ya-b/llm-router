use crate::converters::openai::{OpenAIContent, OpenAIRequest, OpenAITool};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// Import the structs from their new files
use crate::converters::gemini::{
    gemini_content::GeminiContent, gemini_function_declaration::GeminiFunctionDeclaration,
    gemini_generation_config::GeminiGenerationConfig, gemini_inline_data::GeminiInlineData,
    gemini_part::GeminiPart, gemini_tool::GeminiTool,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiRequest {
    // Not sent to Gemini API; used for routing consistency
    #[serde(skip_serializing)]
    #[serde(default)]
    pub model: String,
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(rename = "generationConfig")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    // Not sent to Gemini API; used only for routing
    #[serde(skip_serializing)]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra_fields: HashMap<String, Value>,
}

impl From<OpenAIRequest> for GeminiRequest {
    fn from(openai: OpenAIRequest) -> Self {
        let mut contents: Vec<GeminiContent> = Vec::new();
        let mut system_instruction: Option<GeminiContent> = None;

        for msg in openai.messages.into_iter() {
            if msg.role == "system" {
                // Map system to system_instruction as a text-only content
                match msg.content {
                    OpenAIContent::Text(t) => {
                        system_instruction = Some(GeminiContent {
                            role: Some("user".to_string()),
                            parts: vec![GeminiPart::Text {
                                text: t,
                                thought: None,
                                thought_signature: None,
                            }],
                        });
                    }
                    OpenAIContent::Array(items) => {
                        // Concatenate text items only for system
                        let text = items
                            .into_iter()
                            .filter_map(|i| if i.r#type == "text" { i.text } else { None })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !text.is_empty() {
                            system_instruction = Some(GeminiContent {
                                role: Some("user".to_string()),
                                parts: vec![GeminiPart::Text {
                                    text,
                                    thought: None,
                                    thought_signature: None,
                                }],
                            });
                        }
                    }
                }
                continue;
            }

            let role = Some(
                if msg.role == "assistant" {
                    "model"
                } else {
                    "user"
                }
                .to_string(),
            );
            let parts: Vec<GeminiPart> = match msg.content {
                OpenAIContent::Text(t) => vec![GeminiPart::Text {
                    text: t,
                    thought: None,
                    thought_signature: None,
                }],
                OpenAIContent::Array(items) => {
                    let mut parts: Vec<GeminiPart> = Vec::new();

                    // Aggregate text items into a single Text part
                    let text = items
                        .iter()
                        .filter_map(|i| {
                            if i.r#type == "text" {
                                i.text.clone()
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    if !text.is_empty() {
                        parts.push(GeminiPart::Text {
                            text,
                            thought: None,
                            thought_signature: None,
                        });
                    }

                    // Map image_url data URIs to inline_data parts
                    for i in items.into_iter() {
                        if i.r#type == "image_url" {
                            if let Some(image) = i.image_url {
                                if let Some((mime_type, data)) = parse_data_url(&image.url) {
                                    parts.push(GeminiPart::InlineData {
                                        inline_data: GeminiInlineData { mime_type, data },
                                    });
                                }
                            }
                        }
                    }

                    parts
                }
            };
            contents.push(GeminiContent { role, parts });
        }

        // Map Tools -> functionDeclarations (basic fields)
        let tools = openai.tools.map(|ts: Vec<OpenAITool>| {
            vec![GeminiTool {
                function_declarations: ts
                    .into_iter()
                    .map(|t| {
                        let function = t.function;
                        // Clean the parameters schema to remove unsupported fields
                        let mut params_value = function.parameters;
                        let parameters = if params_value.is_null() {
                            None
                        } else {
                            clean_json_schema_for_gemini(&mut params_value);
                            Some(params_value)
                        };
                        let name = function.name;
                        let description = Some(function.description);
                        GeminiFunctionDeclaration {
                            name,
                            description,
                            parameters,
                        }
                    })
                    .collect(),
            }]
        });

        // Build generation config and map structured output
        let mut generation_config = GeminiGenerationConfig {
            thinking_config: None,
            response_mime_type: None,
            response_schema: None,
            stop_sequences: None,
            temperature: openai.temperature,
            top_p: None,
            top_k: None,
            max_output_tokens: openai.max_tokens,
        };

        if let Some(rf) = &openai.response_format {
            match rf.r#type.as_str() {
                "json_schema" => {
                    generation_config.response_mime_type = Some("application/json".to_string());
                    if let Some(spec) = &rf.json_schema {
                        generation_config.response_schema = Some(spec.schema.clone());
                    }
                }
                "json_object" => {
                    generation_config.response_mime_type = Some("application/json".to_string());
                }
                _ => {}
            }
        }
        let generation_config = Some(generation_config);

        GeminiRequest {
            model: openai.model,
            contents,
            system_instruction,
            tools,
            generation_config,
            stream: openai.stream,
            extra_fields: openai.extra_fields,
        }
    }
}

fn clean_json_schema_for_gemini(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            map.remove("$schema");
            map.remove("additionalProperties");

            // Recursively clean nested objects
            for value in map.values_mut() {
                clean_json_schema_for_gemini(value);
            }
        }
        Value::Array(arr) => {
            // Recursively clean items in arrays
            for item in arr.iter_mut() {
                clean_json_schema_for_gemini(item);
            }
        }
        _ => {}
    }
}

fn parse_data_url(url: &str) -> Option<(String, String)> {
    // Expected format: data:<mime>;base64,<data>
    if let Some(rest) = url.strip_prefix("data:") {
        let mut iter = rest.splitn(2, ',');
        let header = iter.next()?;
        let data = iter.next()?.to_string();

        let mut header_parts = header.split(';');
        let mime_type = header_parts.next()?.to_string();
        // Ensure it's base64; if not, skip
        if header_parts.any(|p| p.eq_ignore_ascii_case("base64")) {
            return Some((mime_type, data));
        }
    }
    None
}
