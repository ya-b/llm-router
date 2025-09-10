use serde::{Deserialize, Serialize};

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
    pub api_type: String,
    pub model: String,
    pub api_base: String,
    pub api_key: String,
    #[serde(default = "default_body")]
    pub rewrite_body: String,
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
}

fn default_weight() -> u32 {
    100
}

fn default_body() -> String {
    "{}".to_string()
}

impl Config {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        
        Self::validate_model_names(&config)?;
        
        Self::validate_model_group_names(&config)?;
        
        Self::validate_model_group_model_names(&config)?;
        
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
}
