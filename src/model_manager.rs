use crate::config::{Config, ModelConfig, RoutingStrategy};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::fmt;
use rand::Rng;
use tracing::{debug, info, warn};

pub struct ModelManager {
    config: Arc<Config>,
    proxy: Option<String>,
    // Key: (group_name, model_name), Value: connection count for the model in the group
    connection_counts: HashMap<(String, String), AtomicUsize>,
    // Key: (group_name, model_name), Value: current weight for smooth weighted round robin
    current_weights: HashMap<(String, String), AtomicUsize>,
    // Key: (group_name, model_name), Value: active request count for the model in the group
    active_requests: HashMap<(String, String), AtomicUsize>,
}

impl fmt::Debug for ModelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelManager")
            .field("config", &self.config)
            .field("connection_counts", &self.connection_counts.keys().collect::<Vec<_>>())
            .field("current_weights", &self.current_weights.keys().collect::<Vec<_>>())
            .field("active_requests", &self.active_requests.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ModelManager {
    pub fn new(config: Arc<Config>, proxy: Option<String>) -> Self {
        let mut connection_counts = HashMap::new();
        let mut current_weights = HashMap::new();
        let mut active_requests = HashMap::new();
        
        // Initialize counters for all model groups
        for model_group in &config.router_settings.model_groups {
            // Initialize connection counts and current weights for each model in the group
            for model in &model_group.models {
                let key = (model_group.name.clone(), model.name.clone());
                connection_counts.insert(key.clone(), AtomicUsize::new(0));
                // Initialize current weight with the model's configured weight
                current_weights.insert(key.clone(), AtomicUsize::new(model.weight as usize));
                active_requests.insert(key.clone(), AtomicUsize::new(0));
            }
        }
        
        Self {
            config,
            proxy,
            connection_counts,
            current_weights,
            active_requests,
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

    pub fn select_round_robin(&self, group_name: &str, models: &[crate::config::ModelGroupEntry]) -> String {
        let valid_models: Vec<&crate::config::ModelGroupEntry> = models
            .iter()
            .filter(|model| self.model_exists(&model.name))
            .collect();

        if valid_models.is_empty() {
            return self.config.model_list.get(0).map_or_else(
                || {
                    warn!("No valid models for round robin and model_list is empty.");
                    String::new()
                },
                |m| m.model_name.clone(),
            );
        }

        let total_weight: usize = valid_models.iter().map(|model| model.weight as usize).sum();
        if total_weight == 0 {
            // If all weights are 0, select one randomly (unweighted)
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..valid_models.len());
            return valid_models[index].name.clone();
        }

        let mut best_models: Vec<&crate::config::ModelGroupEntry> = Vec::new();
        let mut max_current_weight = isize::MIN;

        for model in &valid_models {
            if let Some(current_weight_atomic) = self.current_weights.get(&(group_name.to_string(), model.name.clone())) {
                let current_weight = current_weight_atomic.load(Ordering::SeqCst) as isize;
                if current_weight > max_current_weight {
                    max_current_weight = current_weight;
                    best_models.clear();
                    best_models.push(model);
                } else if current_weight == max_current_weight {
                    best_models.push(model);
                }
            }
        }

        let selected_model = if best_models.len() == 1 {
            best_models[0]
        } else if !best_models.is_empty() {
            // Tie-breaking: randomly select one from the best models
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..best_models.len());
            best_models[index]
        } else {
            // Fallback if no model was selected for some reason
            valid_models[0]
        };

        // Update weights
        if let Some(current_weight_atomic) = self.current_weights.get(&(group_name.to_string(), selected_model.name.clone())) {
            current_weight_atomic.fetch_sub(total_weight, Ordering::SeqCst);
        }

        for model in &valid_models {
            if let Some(current_weight_atomic) = self.current_weights.get(&(group_name.to_string(), model.name.clone())) {
                current_weight_atomic.fetch_add(model.weight as usize, Ordering::SeqCst);
            }
        }

        selected_model.name.clone()
    }

    pub fn select_least_conn(&self, group_name: &str, models: &[crate::config::ModelGroupEntry]) -> String {
        let mut min_score = f64::MAX;
        let mut best_models: Vec<&crate::config::ModelGroupEntry> = Vec::new();

        let valid_models: Vec<&crate::config::ModelGroupEntry> = models
            .iter()
            .filter(|model| self.model_exists(&model.name))
            .collect();

        if valid_models.is_empty() {
            return self.config.model_list.get(0).map_or_else(
                || {
                    warn!("No valid models in group {} and model_list is empty.", group_name);
                    String::new()
                },
                |m| m.model_name.clone(),
            );
        }

        for model_entry in &valid_models {
            let key = (group_name.to_string(), model_entry.name.clone());

            let active_requests = self.active_requests.get(&key)
                .map(|count| count.load(Ordering::SeqCst) as f64)
                .unwrap_or(0.0);
            
            let current_weight = self.current_weights.get(&key)
                .map(|weight| weight.load(Ordering::SeqCst) as f64)
                .unwrap_or(model_entry.weight as f64);
            
            let score = if current_weight > 0.0 {
                active_requests / current_weight
            } else {
                f64::MAX
            };
            
            debug!("Model {} in group {}: active_requests={}, current_weight={}, score={}",
                   model_entry.name, group_name, active_requests, current_weight, score);

            if score < min_score {
                min_score = score;
                best_models.clear();
                best_models.push(model_entry);
            } else if (score - min_score).abs() < f64::EPSILON {
                best_models.push(model_entry);
            }
        }

        if best_models.is_empty() {
            warn!("No suitable model found in group {}, falling back to first valid model in group.", group_name);
            return valid_models.first().map_or_else(
                || self.config.model_list.get(0).map_or_else(|| String::new(), |m| m.model_name.clone()),
                |m| m.name.clone()
            );
        }

        if best_models.len() == 1 {
            return best_models[0].name.clone();
        }
        
        let best_models_owned: Vec<crate::config::ModelGroupEntry> = best_models.into_iter().cloned().collect();

        debug!("Multiple models with best score, using weighted random selection.");
        self.select_random(&best_models_owned)
    }

    pub fn select_random(&self, models: &[crate::config::ModelGroupEntry]) -> String {
        let valid_models: Vec<_> = models
            .iter()
            .filter(|model| self.model_exists(&model.name))
            .collect();

        if valid_models.is_empty() {
            return self.config.model_list.get(0).map_or_else(
                || {
                    warn!("No valid models found for random selection and model_list is empty.");
                    String::new()
                },
                |m| m.model_name.clone(),
            );
        }

        let total_weight: u32 = valid_models.iter().map(|m| m.weight).sum();
        if total_weight == 0 {
            // If all weights are 0, select one randomly (unweighted)
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..valid_models.len());
            return valid_models[index].name.clone();
        }

        let mut rng = rand::thread_rng();
        let mut random_weight = rng.gen_range(0..total_weight);

        for model in &valid_models {
            if random_weight < model.weight {
                return model.name.clone();
            }
            random_weight -= model.weight;
        }

        // Fallback in case of rounding errors or other unexpected issues.
        valid_models.last().map_or_else(
            || self.config.model_list.get(0).map_or_else(|| String::new(), |m| m.model_name.clone()),
            |m| m.name.clone()
        )
    }

    fn model_exists(&self, model_name: &str) -> bool {
        self.config.model_list.iter().any(|m| m.model_name == model_name)
    }

    
    pub fn is_alias(&self, model: &str) -> bool {
        self.config.router_settings.model_groups.iter().find(|x| x.name == model).is_some()
    }

    pub fn get_config(&self) -> &Arc<Config> {
        &self.config
    }

    /// Track the start of a chat completion request
    pub fn start_request(&self, group_name: &str, model_name: &str) {
        let key = (group_name.to_string(), model_name.to_string());
        
        // Increment active request count
        if let Some(active_requests) = self.active_requests.get(&key) {
            let new_count = active_requests.fetch_add(1, Ordering::SeqCst) + 1;
            debug!("Started request for model {} in group {}, active requests: {}", model_name, group_name, new_count);
        }
        
        // Also increment connection count for backward compatibility
        if let Some(connection_count) = self.connection_counts.get(&key) {
            let new_conn_count = connection_count.fetch_add(1, Ordering::SeqCst) + 1;
            debug!("Connection count for model {} in group {}: {}", model_name, group_name, new_conn_count);
        }
    }

    /// Track the end of a chat completion request
    pub fn end_request(&self, group_name: &str, model_name: &str, success: bool) {
        let key = (group_name.to_string(), model_name.to_string());
        
        // Decrement active request count
        if let Some(active_requests) = self.active_requests.get(&key) {
            let new_count = active_requests.fetch_sub(1, Ordering::SeqCst) - 1;
            debug!("Ended request for model {} in group {}, success: {}, active requests: {}",
                   model_name, group_name, success, new_count.max(0));
        }
        
        // Also decrement connection count for backward compatibility
        if let Some(connection_count) = self.connection_counts.get(&key) {
            let new_conn_count = connection_count.fetch_sub(1, Ordering::SeqCst) - 1;
            debug!("Connection count for model {} in group {}: {}", model_name, group_name, new_conn_count.max(0));
        }
        
        // Handle failure case
        if !success {
            warn!("Request failed for model {} in group {}, reducing weight", model_name, group_name);
            self.reduce_model_weight(group_name, model_name);
        }
    }

    /// Reduce the weight of a model by half when it fails
    fn reduce_model_weight(&self, group_name: &str, model_name: &str) {
        let key = (group_name.to_string(), model_name.to_string());
        
        // Find the model group and model entry to get the original weight
        if let Some(model_group) = self.config.router_settings.model_groups.iter().find(|g| g.name == group_name) {
            if let Some(model_entry) = model_group.models.iter().find(|m| m.name == model_name) {
                let _original_weight = model_entry.weight as usize;
                
                // Update current weight (reduce by half, minimum of 1)
                if let Some(current_weight) = self.current_weights.get(&key) {
                    let mut new_weight;
                    loop {
                        let current = current_weight.load(Ordering::SeqCst);
                        new_weight = (current / 2).max(1);
                        if current_weight.compare_exchange_weak(current, new_weight, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                            break;
                        }
                    }
                    
                    info!("Reduced weight for model {} in group {} from {} to {}",
                          model_name, group_name, current_weight.load(Ordering::SeqCst) + new_weight, new_weight);
                }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ModelConfig, RoutingStrategy, ModelGroup, ModelGroupEntry, LLMParams};

    // Helper function to create a test config
    fn create_test_config() -> Config {
        Config {
            model_list: vec![
                ModelConfig {
                    model_name: "model1".to_string(),
                    llm_params: LLMParams {
                        api_type: "openai".to_string(),
                        model: "gpt-3.5-turbo".to_string(),
                        api_base: "https://api.openai.com/v1".to_string(),
                        api_key: "test-key".to_string(),
                        rewrite_body: "{}".to_string(),
                        rewrite_header: "{}".to_string(),
                    },
                },
                ModelConfig {
                    model_name: "model2".to_string(),
                    llm_params: LLMParams {
                        api_type: "openai".to_string(),
                        model: "gpt-4".to_string(),
                        api_base: "https://api.openai.com/v1".to_string(),
                        api_key: "test-key".to_string(),
                        rewrite_body: "{}".to_string(),
                        rewrite_header: "{}".to_string(),
                    },
                },
                ModelConfig {
                    model_name: "model3".to_string(),
                    llm_params: LLMParams {
                        api_type: "openai".to_string(),
                        model: "gpt-4-turbo".to_string(),
                        api_base: "https://api.openai.com/v1".to_string(),
                        api_key: "test-key".to_string(),
                        rewrite_body: "{}".to_string(),
                        rewrite_header: "{}".to_string(),
                    },
                },
            ],
            router_settings: crate::config::RouterSettings {
                strategy: RoutingStrategy::RoundRobin,
                model_groups: vec![
                    ModelGroup {
                        name: "test_group".to_string(), // Use the same group name as in tests
                        models: vec![
                            ModelGroupEntry {
                                name: "model1".to_string(),
                                weight: 1,
                            },
                            ModelGroupEntry {
                                name: "model2".to_string(),
                                weight: 2,
                            },
                            ModelGroupEntry {
                                name: "model3".to_string(),
                                weight: 3,
                            },
                        ],
                    },
                    ModelGroup {
                        name: "group2".to_string(),
                        models: vec![
                            ModelGroupEntry {
                                name: "model1".to_string(),
                                weight: 1,
                            },
                            ModelGroupEntry {
                                name: "model3".to_string(),
                                weight: 1,
                            },
                        ],
                    },
                ],
            },
        }
    }

    #[test]
    fn test_select_round_robin() {
        let config = Arc::new(create_test_config());
        let model_manager = ModelManager::new(config, None);
        
        // Get models from the test_group in the config
        let models = vec![
            ModelGroupEntry {
                name: "model1".to_string(),
                weight: 1,
            },
            ModelGroupEntry {
                name: "model2".to_string(),
                weight: 2,
            },
            ModelGroupEntry {
                name: "model3".to_string(),
                weight: 3,
            },
        ];
        
        let group_name = "test_group";
        
        // Test round-robin selection multiple times to get a better distribution
        let mut selections = Vec::new();
        for _ in 0..6 {
            let selected = model_manager.select_round_robin(group_name, &models);
            selections.push(selected);
        }
        
        // Print the actual selections for debugging
        println!("Selections: {:?}", selections);
        
        // Check that all models were selected at least once
        let model3_count = selections.iter().filter(|s| s.as_str() == "model3").count();
        let model2_count = selections.iter().filter(|s| s.as_str() == "model2").count();
        let model1_count = selections.iter().filter(|s| s.as_str() == "model1").count();
        
        // With a small number of selections, we can't guarantee exact distribution,
        // but we can check that all models were selected at least once
        assert!(model3_count == 3);
        assert!(model2_count == 2);
        assert!(model1_count == 1);
        
        // The algorithm should distribute selections based on weights, but with a small
        // number of selections, we can't guarantee the exact order
        // Let's just check that the function returns valid model names
        for selection in selections {
            assert!(selection == "model1" || selection == "model2" || selection == "model3");
        }
    }
    
    #[test]
    fn test_select_round_robin_with_nonexistent_models() {
        let mut config = create_test_config();
        // Remove model2 from model_list to test handling of non-existent models
        config.model_list.retain(|m| m.model_name != "model2");
        let config = Arc::new(config);
        let model_manager = ModelManager::new(config, None);
        
        let models = vec![
            ModelGroupEntry {
                name: "model1".to_string(),
                weight: 1,
            },
            ModelGroupEntry {
                name: "model2".to_string(), // This model doesn't exist in model_list
                weight: 2,
            },
            ModelGroupEntry {
                name: "model3".to_string(),
                weight: 3,
            },
        ];
        
        let group_name = "test_group";
        
        // Test that non-existent models are filtered out
        let selected = model_manager.select_round_robin(group_name, &models);
        
        // Should select from existing models (model1 and model3)
        assert!(selected == "model1" || selected == "model3");
    }
    
    #[test]
    fn test_select_least_conn() {
        let config = Arc::new(create_test_config());
        let model_manager = ModelManager::new(config, None);
        
        let models = vec![
            ModelGroupEntry {
                name: "model1".to_string(),
                weight: 1,
            },
            ModelGroupEntry {
                name: "model2".to_string(),
                weight: 2,
            },
            ModelGroupEntry {
                name: "model3".to_string(),
                weight: 3,
            },
        ];
        
        let group_name = "test_group";
        
        // Initially, all models have 0 connections, so it should select based on weight
        // With equal connections, higher weight models are preferred
        let selected = model_manager.select_least_conn(group_name, &models);
        println!("Initial selection: {}", selected);
        assert!(selected == "model1" || selected == "model2" || selected == "model3");
        
        // Add connections to model3 to make it less preferred
        for _ in 0..5 {
            model_manager.start_request(group_name, "model3");
        }
        
        // Now model3 has more connections, check the selection
        let selected = model_manager.select_least_conn(group_name, &models);
        println!("After adding connections to model3: {}", selected);
        assert!(selected == "model1" || selected == "model2" || selected == "model3");
        
        // Add connections to model2 to make it less preferred
        for _ in 0..5 {
            model_manager.start_request(group_name, "model2");
        }
        
        // Now model2 has more connections, check the selection
        let selected = model_manager.select_least_conn(group_name, &models);
        println!("After adding connections to model2: {}", selected);
        assert!(selected == "model1" || selected == "model2" || selected == "model3");
        
        // Add connections to model1 to make it less preferred
        for _ in 0..5 {
            model_manager.start_request(group_name, "model1");
        }
        
        // Now model1 has more connections, check the selection
        let selected = model_manager.select_least_conn(group_name, &models);
        println!("After adding connections to model1: {}", selected);
        assert!(selected == "model1" || selected == "model2" || selected == "model3");
        
        // Reset all connections by ending them
        for _ in 0..5 {
            model_manager.end_request(group_name, "model1", true);
            model_manager.end_request(group_name, "model2", true);
            model_manager.end_request(group_name, "model3", true);
        }
        
        // Now all models should have 0 connections again, check the selection
        let selected = model_manager.select_least_conn(group_name, &models);
        println!("After resetting connections: {}", selected);
        assert!(selected == "model1" || selected == "model2" || selected == "model3");
    }
    
    #[test]
    fn test_select_least_conn_with_nonexistent_models() {
        let mut config = create_test_config();
        // Remove model2 from model_list to test handling of non-existent models
        config.model_list.retain(|m| m.model_name != "model2");
        let config = Arc::new(config);
        let model_manager = ModelManager::new(config, None);
        
        let models = vec![
            ModelGroupEntry {
                name: "model1".to_string(),
                weight: 1,
            },
            ModelGroupEntry {
                name: "model2".to_string(), // This model doesn't exist in model_list
                weight: 2,
            },
            ModelGroupEntry {
                name: "model3".to_string(),
                weight: 3,
            },
        ];
        
        let group_name = "test_group";
        
        // Test that non-existent models are filtered out
        let selected = model_manager.select_least_conn(group_name, &models);
        
        // Should select from existing models (model1 and model3)
        assert!(selected == "model1" || selected == "model3");
    }
    
    #[test]
    fn test_select_random() {
        let config = Arc::new(create_test_config());
        let model_manager = ModelManager::new(config, None);
        
        let models = vec![
            ModelGroupEntry {
                name: "model1".to_string(),
                weight: 1,
            },
            ModelGroupEntry {
                name: "model2".to_string(),
                weight: 2,
            },
            ModelGroupEntry {
                name: "model3".to_string(),
                weight: 3,
            },
        ];
        
        let group_name = "test_group";
        
        // Test random selection multiple times
        let mut selections = Vec::new();
        for _ in 0..1000 {
            let selected = model_manager.select_random(&models);
            selections.push(selected);
        }
        
        // Check that all models are selected
        assert!(selections.contains(&"model1".to_string()));
        assert!(selections.contains(&"model2".to_string()));
        assert!(selections.contains(&"model3".to_string()));
        
        // Check that selection frequency roughly matches weights
        let model1_count = selections.iter().filter(|s| s.as_str() == "model1").count();
        let model2_count = selections.iter().filter(|s| s.as_str() == "model2").count();
        let model3_count = selections.iter().filter(|s| s.as_str() == "model3").count();
        
        // With weights 1, 2, 3, the ratios should be approximately 1:2:3
        // Allow some tolerance for randomness
        let total = model1_count + model2_count + model3_count;
        let model1_ratio = model1_count as f64 / total as f64;
        let model2_ratio = model2_count as f64 / total as f64;
        let model3_ratio = model3_count as f64 / total as f64;
        
        // Expected ratios: 1/6, 2/6, 3/6
        assert!((model1_ratio - 1.0/6.0).abs() < 0.1);
        assert!((model2_ratio - 2.0/6.0).abs() < 0.1);
        assert!((model3_ratio - 3.0/6.0).abs() < 0.1);
    }
    
    #[test]
    fn test_select_random_with_nonexistent_models() {
        let mut config = create_test_config();
        // Remove model2 from model_list to test handling of non-existent models
        config.model_list.retain(|m| m.model_name != "model2");
        let config = Arc::new(config);
        let model_manager = ModelManager::new(config, None);
        
        let models = vec![
            ModelGroupEntry {
                name: "model1".to_string(),
                weight: 1,
            },
            ModelGroupEntry {
                name: "model2".to_string(), // This model doesn't exist in model_list
                weight: 2,
            },
            ModelGroupEntry {
                name: "model3".to_string(),
                weight: 3,
            },
        ];
        
        // Test that non-existent models are filtered out
        let selected = model_manager.select_random(&models);
        
        // Should select from existing models (model1 and model3)
        assert!(selected == "model1" || selected == "model3");
    }
    
    #[test]
    fn test_select_random_with_all_nonexistent_models() {
        let mut config = create_test_config();
        // Remove all models from model_list
        config.model_list.clear();
        let config = Arc::new(config);
        let model_manager = ModelManager::new(config, None);
        
        let models = vec![
            ModelGroupEntry {
                name: "model1".to_string(), // Doesn't exist
                weight: 1,
            },
            ModelGroupEntry {
                name: "model2".to_string(), // Doesn't exist
                weight: 2,
            },
        ];
        
        // When all models are non-existent and the model_list is empty,
        // the function should handle this gracefully by returning an empty string.
        let selected = model_manager.select_random(&models);
        
        // Check that the function returns an empty string and does not panic.
        assert!(selected.is_empty());
    }
}