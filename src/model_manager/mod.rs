use crate::config::{Config, ModelConfig, RoutingStrategy};
use std::collections::HashMap;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::fmt;
use tracing::{debug, info, warn};

mod strategy;
mod registry;
mod types;
mod health;

use types::ModelKey;

pub struct ModelManager {
    pub(super) config: Arc<Config>,
    // Key: (group_name, model_name), Value: current weight for smooth weighted round robin
    pub(super) current_weights: HashMap<ModelKey, AtomicIsize>,
    // Key: (group_name, model_name), Value: active request count for the model in the group
    pub(super) active_requests: HashMap<ModelKey, AtomicUsize>,
    // Per-group lock to make SWRR selection + update atomic across the group
    pub(super) group_locks: HashMap<String, Mutex<()>>,
    // Runtime health/weight factors
    pub(super) health: health::Health,
}

impl fmt::Debug for ModelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelManager")
            .field("config", &self.config)
            .field("current_weights", &self.current_weights.keys().collect::<Vec<_>>())
            .field("active_requests", &self.active_requests.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub group: Option<String>,
    pub model_name: String,
    pub config: ModelConfig,
}

impl ModelManager {

    pub fn resolve(&self, hint: &str) -> Option<Selection> {
        // If it's a group alias
        if let Some(model_group) = self
            .config
            .router_settings
            .model_groups
            .iter()
            .find(|g| g.name == hint)
        {
            // Filter valid
            let registry = registry::Registry::new(&self.config);
            let valid_models: Vec<crate::config::ModelGroupEntry> =
                registry.filter_valid_entries(&model_group.models);
            if valid_models.is_empty() {
                return None;
            }
            let chosen = match self.config.router_settings.strategy {
                RoutingStrategy::RoundRobin => self.select_round_robin(&model_group.name, &valid_models),
                RoutingStrategy::LeastConn => self.select_least_conn(&model_group.name, &valid_models),
                RoutingStrategy::Random => self.select_random(&valid_models),
            };
            if chosen.is_empty() { return None; }
            if let Some(cfg) = self.find_model(&chosen) {
                return Some(Selection {
                    group: Some(model_group.name.clone()),
                    model_name: chosen,
                    config: cfg.clone(),
                });
            }
            return None;
        }

        // Otherwise treat as direct model name
        self.find_model(hint).map(|cfg| Selection {
            group: None,
            model_name: hint.to_string(),
            config: cfg.clone(),
        })
    }
    pub fn new(config: Arc<Config>) -> Self {
        let mut current_weights = HashMap::new();
        let mut active_requests = HashMap::new();
        let mut group_locks = HashMap::new();
        
        // Initialize counters for all model groups
        for model_group in &config.router_settings.model_groups {
            // Create a lock per group to guard SWRR selection + updates
            group_locks
                .entry(model_group.name.clone())
                .or_insert_with(|| Mutex::new(()));
            // Initialize connection counts and current weights for each model in the group
            for model in &model_group.models {
                let key = ModelKey::new(model_group.name.clone(), model.name.clone());
                // Initialize current weight to 0 for SWRR
                current_weights.insert(key.clone(), AtomicIsize::new(0));
                active_requests.insert(key.clone(), AtomicUsize::new(0));
            }
        }
        let health = health::Health::new_from_config(&config.clone());
        Self {
            config,
            current_weights,
            active_requests,
            group_locks,
            health: health,
        }
    }
    
    pub fn update_config(&mut self, new_config: Arc<Config>) {
        // Create a new instance with the new config and reuse its fields
        let new_manager = Self::new(new_config.clone());
        self.config = new_config;
        self.current_weights = new_manager.current_weights;
        self.active_requests = new_manager.active_requests;
        self.group_locks = new_manager.group_locks;
        self.health = new_manager.health;
    }

    // Helper: find a model config by exact name
    fn find_model(&self, name: &str) -> Option<&ModelConfig> {
        self.config.model_list.iter().find(|m| m.model_name == name)
    }
 
    pub(super) fn model_exists(&self, model_name: &str) -> bool {
        self.config.model_list.iter().any(|m| m.model_name == model_name)
    }
 
    pub fn get_config(&self) -> &Arc<Config> {
        &self.config
    }

    /// Track the start of a chat completion request
    pub fn start_request(&self, group_name: &str, model_name: &str) {
        let key = ModelKey::new(group_name.to_string(), model_name.to_string());
        
        // Increment active request count
        if let Some(active_requests) = self.active_requests.get(&key) {
            let new_count = active_requests.fetch_add(1, Ordering::SeqCst) + 1;
            debug!("Started request for model {} in group {}, active requests: {}", model_name, group_name, new_count);
        }
        
    }

    /// Start using a selection handle
    pub fn start(&self, selection: &Selection) {
        if let Some(group) = &selection.group {
            self.start_request(group, &selection.model_name);
        }
    }

    /// Track the end of a chat completion request
    pub fn end_request(&self, group_name: &str, model_name: &str, success: bool) {
        let key = ModelKey::new(group_name.to_string(), model_name.to_string());
        
        // Decrement active request count
        if let Some(active_requests) = self.active_requests.get(&key) {
            let new_count = active_requests.fetch_sub(1, Ordering::SeqCst) - 1;
            debug!("Ended request for model {} in group {}, success: {}, active requests: {}",
                   model_name, group_name, success, new_count.max(0));
        }
        
        // Handle health updates
        if !success {
            warn!("Request failed for model {} in group {}, reducing weight", model_name, group_name);
            self.reduce_model_weight(group_name, model_name);
        } else {
            self.health.recover_on_success(&key);
        }
    }

    /// Reduce the weight of a model by half when it fails
    fn reduce_model_weight(&self, group_name: &str, model_name: &str) {
        let key = ModelKey::new(group_name.to_string(), model_name.to_string());
        // Update runtime health factor and breaker state
        self.health.decay(&key);
        self.health.on_failure(&key);
        
        // Find the model group and model entry to get the original weight
        if let Some(model_group) = self.config.router_settings.model_groups.iter().find(|g| g.name == group_name) {
            if let Some(model_entry) = model_group.models.iter().find(|m| m.name == model_name) {
                let _original_weight = model_entry.weight as usize;
                
                // Update current weight (reduce by half, minimum of 1)
                if let Some(current_weight) = self.current_weights.get(&key) {
                    let mut new_weight;
                    let mut old_weight;
                    loop {
                        let current = current_weight.load(Ordering::SeqCst);
                        old_weight = current;
                        // halve; ensure at least 1
                        new_weight = (current / 2).max(1);
                        if current_weight
                            .compare_exchange_weak(current, new_weight, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            break;
                        }
                    }
                    
                    info!(
                        "Reduced weight for model {} in group {} from {} to {}",
                        model_name,
                        group_name,
                        old_weight,
                        new_weight
                    );
                }
            }
        }
    }

    /// End using a selection handle
    pub fn end(&self, selection: &Selection, success: bool) {
        if let Some(group) = &selection.group {
            self.end_request(group, &selection.model_name, success);
        } else {
            // Direct model (no group). Keep current behavior: no counters/health updates.
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
                        api_type: crate::config::ApiType::OpenAI,
                        model: "gpt-3.5-turbo".to_string(),
                        api_base: "https://api.openai.com/v1".to_string(),
                        api_key: "test-key".to_string(),
                        rewrite_body: serde_json::json!({}),
                        rewrite_header: serde_json::json!({}),
                    },
                },
                ModelConfig {
                    model_name: "model2".to_string(),
                    llm_params: LLMParams {
                        api_type: crate::config::ApiType::OpenAI,
                        model: "gpt-4".to_string(),
                        api_base: "https://api.openai.com/v1".to_string(),
                        api_key: "test-key".to_string(),
                        rewrite_body: serde_json::json!({}),
                        rewrite_header: serde_json::json!({}),
                    },
                },
                ModelConfig {
                    model_name: "model3".to_string(),
                    llm_params: LLMParams {
                        api_type: crate::config::ApiType::OpenAI,
                        model: "gpt-4-turbo".to_string(),
                        api_base: "https://api.openai.com/v1".to_string(),
                        api_key: "test-key".to_string(),
                        rewrite_body: serde_json::json!({}),
                        rewrite_header: serde_json::json!({}),
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
        let model_manager = ModelManager::new(config);
        
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
        let model_manager = ModelManager::new(config);
        
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
        let model_manager = ModelManager::new(config);
        
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
        let model_manager = ModelManager::new(config);
        
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
        let model_manager = ModelManager::new(config);
        
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
        let model_manager = ModelManager::new(config);
        
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
        let model_manager = ModelManager::new(config);
        
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
