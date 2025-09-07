use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMParams {
    pub model: String,
    pub api_base: String,
    pub api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterSettings {
    #[serde(deserialize_with = "deserialize_model_alias")]
    pub model_group_alias: HashMap<String, Vec<String>>,
}

fn deserialize_model_alias<'de, D>(deserializer: D) -> Result<HashMap<String, Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    let map: HashMap<String, Vec<String>> = serde_json::from_str(&s).map_err(serde::de::Error::custom)?;
    Ok(map)
}

impl Config {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn get_model_config(&self, model: &str) -> Option<&ModelConfig> {
        // First check if the model is in model_group_alias
        if let Some(aliases) = self.router_settings.model_group_alias.get(model) {
            if let Some(first_alias) = aliases.first() {
                return self.model_list.iter().find(|m| m.model_name == *first_alias);
            }
        }
        
        // If not found in alias, check direct model name
        self.model_list.iter().find(|m| m.model_name == model)
    }
}