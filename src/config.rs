use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use crate::utils::jq_util::check_jaq_filter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_list: Vec<ModelConfig>,
    pub router_settings: RouterSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    pub llm_params: LLMParams,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ApiType {
    OpenAI,
    Anthropic,
    Gemini,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMParams {
    pub api_type: ApiType,
    pub model: String,
    pub api_base: String,
    pub api_key: String,
    #[serde(default = "default_json_object")]
    pub rewrite_body: Value,
    #[serde(default = "default_json_object")]
    pub rewrite_header: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterSettings {
    pub strategy: RoutingStrategy,
    pub model_groups: Vec<ModelGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoutingStrategy {
    RoundRobin,
    LeastConn,
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGroup {
    pub name: String,
    
    pub models: Vec<ModelGroupEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGroupEntry {
    pub name: String,
    #[serde(default = "default_weight")]
    pub weight: u32,
    // Optional jq selector; when present and non-empty, request must satisfy it
    #[serde(default)]
    pub selector: Option<String>,
}

fn default_weight() -> u32 {
    100
}

fn default_json_object() -> Value { json!({}) }

impl Config {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Config = serde_yaml::from_str(&content)?;
        
        // Normalize rewrite_body/rewrite_header allowing stringified JSON in YAML
        for mc in &mut config.model_list {
            normalize_llm_params(&mut mc.llm_params);
        }
        
        Self::validate_model_names(&config)?;
        
        Self::validate_model_group_names(&config)?;
        
        Self::validate_model_group_model_names(&config)?;

        // Validate selectors in model groups (non-empty only)
        Self::validate_model_group_selectors(&config)?;
        
        Ok(config)
    }
    
    fn validate_model_names(config: &Config) -> anyhow::Result<()> {
        let mut seen_names = std::collections::HashSet::new();
        
        for model in &config.model_list {
            if seen_names.contains(&model.model_name) {
                return Err(anyhow::anyhow!(
                    "Duplicate model_name found: '{}'. Model names must be unique.",
                    model.model_name
                ));
            }
            seen_names.insert(model.model_name.clone());
        }
        
        Ok(())
    }
    
    fn validate_model_group_names(config: &Config) -> anyhow::Result<()> {
        let mut model_group_names = std::collections::HashSet::new();
        
        for model_group in &config.router_settings.model_groups {
            if model_group_names.contains(&model_group.name) {
                return Err(anyhow::anyhow!(
                    "Duplicate model_group names '{}' found in model_group", &model_group.name
                ));
            }
            model_group_names.insert(model_group.name.clone());
        }
        
        Ok(())
    }
    
    fn validate_model_group_model_names(config: &Config) -> anyhow::Result<()> {
        for model_group in &config.router_settings.model_groups {
            let mut seen_names = std::collections::HashSet::new();
            
            for entry in &model_group.models {
                if seen_names.contains(&entry.name) {
                    return Err(anyhow::anyhow!(
                        "Duplicate model name '{}' found in model_group '{}'. Model names in a group must be unique.",
                        entry.name, &model_group.name
                    ));
                }
                seen_names.insert(entry.name.clone());
            }
        }
        
        Ok(())
    }

    fn validate_model_group_selectors(config: &Config) -> anyhow::Result<()> {
        for group in &config.router_settings.model_groups {
            for entry in &group.models {
                if let Some(selector) = &entry.selector {
                    if !selector.trim().is_empty() {
                        if !check_jaq_filter(selector) {
                            return Err(anyhow::anyhow!(
                                "Invalid jq selector for model '{}' in group '{}': {}",
                                entry.name,
                                group.name,
                                selector
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn normalize_llm_params(params: &mut LLMParams) {
    // If the YAML provided a quoted JSON string, try to parse into JSON object/value
    if let Value::String(s) = &params.rewrite_body {
        if let Ok(v) = serde_json::from_str::<Value>(s) { params.rewrite_body = v; }
    }
    if let Value::String(s) = &params.rewrite_header {
        if let Ok(v) = serde_json::from_str::<Value>(s) { params.rewrite_header = v; }
    }
}
