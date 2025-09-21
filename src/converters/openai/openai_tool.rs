use serde::{de, Deserialize, Deserializer, Serialize};
use crate::converters::openai::openai_function::OpenAIFunction;

#[derive(Debug, Clone, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: OpenAIFunction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// Support both shapes:
// - Chat Completions style: { "type": "function", "function": { name, description, parameters } }
// - Responses API style:    { "type": "function", name, description?, parameters, strict? }
impl<'de> Deserialize<'de> for OpenAITool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Try deserializing from a serde_json::Value, then match shapes
        let v = serde_json::Value::deserialize(deserializer)?;

        #[derive(Deserialize)]
        struct LegacyTool {
            #[serde(rename = "type")]
            r#type: String,
            function: OpenAIFunction,
            #[serde(default)]
            strict: Option<bool>,
        }

        #[derive(Deserialize)]
        struct FlatTool {
            #[serde(rename = "type")]
            r#type: String,
            name: String,
            #[serde(default)]
            description: String,
            parameters: serde_json::Value,
            #[serde(default)]
            strict: Option<bool>,
        }

        if let Ok(l) = serde_json::from_value::<LegacyTool>(v.clone()) {
            return Ok(OpenAITool { r#type: l.r#type, function: l.function, strict: l.strict });
        }
        if let Ok(f) = serde_json::from_value::<FlatTool>(v) {
            return Ok(OpenAITool {
                r#type: f.r#type,
                function: OpenAIFunction { name: f.name, description: f.description, parameters: f.parameters },
                strict: f.strict,
            });
        }
        Err(de::Error::custom("invalid OpenAI tool format"))
    }
}
