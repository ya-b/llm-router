use crate::config::{Config, ModelConfig, RoutingStrategy};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::fmt;
use rand::Rng;

pub struct ModelManager {
    config: Arc<Config>,
    proxy: Option<String>,
    // Key: (group_name, model_name), Value: connection count for the model in the group
    connection_counts: HashMap<(String, String), AtomicUsize>,
    // Key: (group_name, model_name), Value: current weight for smooth weighted round robin
    current_weights: HashMap<(String, String), AtomicUsize>,
}

impl fmt::Debug for ModelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelManager")
            .field("config", &self.config)
            .field("connection_counts", &self.connection_counts.keys().collect::<Vec<_>>())
            .field("current_weights", &self.current_weights.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ModelManager {
    pub fn new(config: Arc<Config>, proxy: Option<String>) -> Self {
        let mut connection_counts = HashMap::new();
        let mut current_weights = HashMap::new();
        
        // Initialize counters for all model groups
        for model_group in &config.router_settings.model_groups {
            // Initialize connection counts and current weights for each model in the group
            for model in &model_group.models {
                connection_counts.insert((model_group.name.clone(), model.name.clone()), AtomicUsize::new(0));
                // Initialize current weight with the model's configured weight
                current_weights.insert((model_group.name.clone(), model.name.clone()), AtomicUsize::new(model.weight as usize));
            }
        }
        
        Self {
            config,
            proxy,
            connection_counts,
            current_weights,
        }
    }
    
    pub fn update_config(&mut self, new_config: Arc<Config>) {
        // Create a new instance with the new config and reuse its fields
        let new_manager = Self::new(new_config.clone(), self.get_proxy().clone());
        self.config = new_config;
        self.connection_counts = new_manager.connection_counts;
        self.current_weights = new_manager.current_weights;
    }

    pub fn get_proxy(&self) -> Option<String> {
        self.proxy.clone()
    }
    
    pub fn get_model_config(&self, model: &str) -> Option<&ModelConfig> {
        // First check if the model is in model_group
        if let Some(model_group) = self.config.router_settings.model_groups.iter().find(|m| m.name == model) {
            if model_group.models.is_empty() {
                return None;
            }
            
            if model_group.models.len() == 1 {
                // If there's only one model in the group, return it directly
                return self.config.model_list.iter().find(|m| m.model_name == model_group.models[0].name);
            } else {
                // Multiple models in the group, use the configured strategy
                let selected_model_name = match self.config.router_settings.strategy {
                    RoutingStrategy::RoundRobin => self.select_round_robin(model, &model_group.models),
                    RoutingStrategy::LeastConn => self.select_least_conn(model, &model_group.models),
                    RoutingStrategy::Random => self.select_random(&model_group.models),
                };
                return self.config.model_list.iter().find(|m| m.model_name == selected_model_name);
            }
        }
        
        // If not found in group, check direct model name
        self.config.model_list.iter().find(|m| m.model_name == model)
    }

    fn select_round_robin(&self, group_name: &str, models: &[crate::config::ModelGroupEntry]) -> String {
        // Filter out non-existent models and calculate total weight
        let valid_models: Vec<&crate::config::ModelGroupEntry> = models
            .iter()
            .filter(|model| self.model_exists(&model.name))
            .collect();
        
        if valid_models.is_empty() {
            // No valid models found, fallback to first model in model_list
            return self.config.model_list[0].model_name.clone();
        }
        
        // Calculate total weight
        let total_weight: usize = valid_models.iter().map(|model| model.weight as usize).sum();
        
        // Find the model with the highest current weight
        let mut selected_model = &valid_models[0];
        let mut max_current_weight = 0;
        
        for model in &valid_models {
            if let Some(current_weight) = self.current_weights.get(&(group_name.to_string(), model.name.clone())) {
                let weight = current_weight.load(Ordering::SeqCst);
                if weight > max_current_weight {
                    max_current_weight = weight;
                    selected_model = model;
                }
            }
        }
        
        // Update weights: subtract total weight from selected model, add configured weight to all models
        for model in &valid_models {
            if let Some(current_weight) = self.current_weights.get(&(group_name.to_string(), model.name.clone())) {
                if model.name == selected_model.name {
                    // Subtract total weight from selected model
                    current_weight.fetch_sub(total_weight, Ordering::SeqCst);
                }
                // Add configured weight to all models
                current_weight.fetch_add(model.weight as usize, Ordering::SeqCst);
            }
        }
        
        selected_model.name.clone()
    }

    fn select_least_conn(&self, group_name: &str, models: &[crate::config::ModelGroupEntry]) -> String {
        let mut min_count = u32::MAX;
        let mut selected_model = None;
        
        for model_entry in models {
            // Check if model exists in model_list
            if !self.model_exists(&model_entry.name) {
                continue;
            }
            
            if let Some(count) = self.connection_counts.get(&(group_name.to_string(), model_entry.name.clone())) {
                let current_count = count.load(Ordering::SeqCst) as u32;
                // Adjust by weight (higher weight = lower effective count)
                let adjusted_count = current_count * 100 / model_entry.weight;
                if selected_model.is_none() || adjusted_count < min_count {
                    min_count = adjusted_count;
                    selected_model = Some(model_entry);
                }
            }
        }
        
        // If no valid model found, fallback to first valid model
        let selected_model = if let Some(model) = selected_model {
            model
        } else {
            // Try to find first valid model in the group
            let mut fallback_model = None;
            for model_entry in models {
                if self.model_exists(&model_entry.name) {
                    fallback_model = Some(model_entry);
                    break;
                }
            }
            
            if let Some(model) = fallback_model {
                model
            } else {
                // If still no valid model, create a fallback to first model in model_list
                return self.config.model_list[0].model_name.clone();
            }
        };
        
        // Increment connection count for selected model
        if let Some(count) = self.connection_counts.get(&(group_name.to_string(), selected_model.name.clone())) {
            count.fetch_add(1, Ordering::SeqCst);
        }
        
        selected_model.name.clone()
    }

    fn select_random(&self, models: &[crate::config::ModelGroupEntry]) -> String {
        // Create weighted model list for selection, filtering out non-existent models
        let weighted_models = self.create_valid_weighted_model_list(models);
        if weighted_models.is_empty() {
            // No valid models found, fallback to first model in model_list
            return self.config.model_list[0].model_name.clone();
        }
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..weighted_models.len());
        weighted_models[index].clone()
    }

    fn create_valid_weighted_model_list(&self, models: &[crate::config::ModelGroupEntry]) -> Vec<String> {
        let mut weighted_models = Vec::new();
        for model_entry in models {
            // Only include models that exist in model_list
            if self.model_exists(&model_entry.name) {
                for _ in 0..model_entry.weight {
                    weighted_models.push(model_entry.name.clone());
                }
            }
        }
        weighted_models
    }

    fn model_exists(&self, model_name: &str) -> bool {
        self.config.model_list.iter().any(|m| m.model_name == model_name)
    }

    fn get_first_valid_model_name(&self, models: &[crate::config::ModelGroupEntry]) -> String {
        for model_entry in models {
            if self.model_exists(&model_entry.name) {
                return model_entry.name.clone();
            }
        }
        // If no valid model found, return first model in model_list
        self.config.model_list[0].model_name.clone()
    }
    
    pub fn get_all_models_for_alias(&self, alias: &str) -> Vec<&ModelConfig> {
        let mut models = Vec::new();
        
        if let Some(model_group) = self.config.router_settings.model_groups.iter().find(|x| x.name == alias) {
            for model_entry in &model_group.models {
                if let Some(model_config) = self.config.model_list.iter().find(|m| m.model_name == model_entry.name) {
                    models.push(model_config);
                }
            }
        }
        
        models
    }
    
    pub fn is_alias(&self, model: &str) -> bool {
        self.config.router_settings.model_groups.iter().find(|x| x.name == model).is_some()
    }

    pub fn release_connection(&self, group_name: &str, model_name: &str) {
        if let Some(count) = self.connection_counts.get(&(group_name.to_string(), model_name.to_string())) {
            count.fetch_sub(1, Ordering::SeqCst);
        }
    }
    
    pub fn get_config(&self) -> &Arc<Config> {
        &self.config
    }
    
}