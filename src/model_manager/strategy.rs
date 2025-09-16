use rand::Rng;
use tracing::{debug, warn};

use super::ModelManager;
use super::types::ModelKey;

impl ModelManager {
    pub fn select_round_robin(&self, group_name: &str, models: &[crate::config::ModelGroupEntry]) -> String {
        let base_models: Vec<&crate::config::ModelGroupEntry> = models
            .iter()
            .filter(|model| self.model_exists(&model.name))
            .collect();

        // Apply circuit breaker permit; fallback to base list if all filtered out
        let mut valid_models: Vec<&crate::config::ModelGroupEntry> = base_models
            .iter()
            .copied()
            .filter(|m| self.health.permit(group_name, m))
            .collect();
        if valid_models.is_empty() { valid_models = base_models; }

        if valid_models.is_empty() {
            return self.config.model_list.get(0).map_or_else(
                || {
                    warn!("No valid models for round robin and model_list is empty.");
                    String::new()
                },
                |m| m.model_name.clone(),
            );
        }

        // Guard the whole SWRR step for this group to ensure atomicity
        let _guard = self
            .group_locks
            .get(group_name)
            .map(|m| m.lock().unwrap());

        // Sum of effective weights (as isize)
        let total_weight: isize = valid_models
            .iter()
            .map(|model| self.health.effective_weight(group_name, model) as isize)
            .sum();
        if total_weight == 0 {
            // If all weights are 0, select one randomly (unweighted)
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..valid_models.len());
            return valid_models[index].name.clone();
        }

        // 1) Add configured weight to each model's current weight
        for model in &valid_models {
            if let Some(current) = self
                .current_weights
                .get(&ModelKey::new(group_name.to_string(), model.name.clone()))
            {
                let w = self.health.effective_weight(group_name, model) as isize;
                current.fetch_add(w, std::sync::atomic::Ordering::SeqCst);
            }
        }

        // 2) Select model with maximum current weight (random tie-break)
        let mut max_current = isize::MIN;
        let mut best_models: Vec<&crate::config::ModelGroupEntry> = Vec::new();
        for model in &valid_models {
            if let Some(current) = self
                .current_weights
                .get(&ModelKey::new(group_name.to_string(), model.name.clone()))
            {
                let val = current.load(std::sync::atomic::Ordering::SeqCst);
                if val > max_current {
                    max_current = val;
                    best_models.clear();
                    best_models.push(model);
                } else if val == max_current {
                    best_models.push(model);
                }
            }
        }

        let selected_model = if best_models.len() == 1 {
            best_models[0]
        } else if !best_models.is_empty() {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..best_models.len());
            best_models[index]
        } else {
            // Fallback should not happen; choose first valid
            valid_models[0]
        };

        // 3) Subtract total weight from the selected model's current weight
        if let Some(curr) = self
            .current_weights
            .get(&ModelKey::new(group_name.to_string(), selected_model.name.clone()))
        {
            curr.fetch_sub(total_weight, std::sync::atomic::Ordering::SeqCst);
        }

        selected_model.name.clone()
    }

    pub fn select_least_conn(&self, group_name: &str, models: &[crate::config::ModelGroupEntry]) -> String {
        let mut min_score = f64::MAX;
        let mut best_models: Vec<&crate::config::ModelGroupEntry> = Vec::new();

        let base_models: Vec<&crate::config::ModelGroupEntry> = models
            .iter()
            .filter(|model| self.model_exists(&model.name))
            .collect();

        let mut valid_models: Vec<&crate::config::ModelGroupEntry> = base_models
            .iter()
            .copied()
            .filter(|m| self.health.permit(group_name, m))
            .collect();
        if valid_models.is_empty() { valid_models = base_models; }

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
            let key = ModelKey::new(group_name.to_string(), model_entry.name.clone());

            let active_requests = self
                .active_requests
                .get(&key)
                .map(|count| count.load(std::sync::atomic::Ordering::SeqCst) as f64)
                .unwrap_or(0.0);
            
            let current_weight = self.health.effective_weight(group_name, model_entry) as f64;
            
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
        self.select_random_with_group(group_name, &best_models_owned)
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

        let total_weight: u32 = valid_models
            .iter()
            .map(|m| m.weight)
            .sum();
        if total_weight == 0 {
            // If all weights are 0, select one randomly (unweighted)
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..valid_models.len());
            return valid_models[index].name.clone();
        }

        let mut rng = rand::thread_rng();
        let mut random_weight = rng.gen_range(0..total_weight);

        for model in &valid_models {
            let w = model.weight;
            if random_weight < w {
                return model.name.clone();
            }
            random_weight -= w;
        }

        // Fallback in case of rounding errors or other unexpected issues.
        valid_models.last().map_or_else(
            || self.config.model_list.get(0).map_or_else(|| String::new(), |m| m.model_name.clone()),
            |m| m.name.clone()
        )
    }

    pub fn select_random_with_group(&self, group_name: &str, models: &[crate::config::ModelGroupEntry]) -> String {
        let base_models: Vec<_> = models
            .iter()
            .filter(|model| self.model_exists(&model.name))
            .collect();

        if base_models.is_empty() {
            return self.config.model_list.get(0).map_or_else(
                || {
                    warn!("No valid models found for random selection and model_list is empty.");
                    String::new()
                },
                |m| m.model_name.clone(),
            );
        }

        let mut valid_models: Vec<_> = base_models
            .iter()
            .copied()
            .filter(|m| self.health.permit(group_name, m))
            .collect();
        if valid_models.is_empty() { valid_models = base_models; }

        let total_weight: u32 = valid_models
            .iter()
            .map(|m| self.health.effective_weight(group_name, m))
            .sum();
        if total_weight == 0 {
            // If all weights are 0, select one randomly (unweighted)
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..valid_models.len());
            return valid_models[index].name.clone();
        }

        let mut rng = rand::thread_rng();
        let mut random_weight = rng.gen_range(0..total_weight);

        for model in &valid_models {
            let w = self.health.effective_weight(group_name, model);
            if random_weight < w {
                return model.name.clone();
            }
            random_weight -= w;
        }

        valid_models.last().map_or_else(
            || self.config.model_list.get(0).map_or_else(|| String::new(), |m| m.model_name.clone()),
            |m| m.name.clone()
        )
    }
}
