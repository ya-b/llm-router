use crate::config::{Config, ModelGroupEntry};

pub struct Registry<'a> {
    cfg: &'a Config,
}

impl<'a> Registry<'a> {
    pub fn new(cfg: &'a Config) -> Self { Self { cfg } }

    pub fn model_exists(&self, model_name: &str) -> bool {
        self.cfg.model_list.iter().any(|m| m.model_name == model_name)
    }

    pub fn filter_valid_entries(&self, entries: &[ModelGroupEntry]) -> Vec<ModelGroupEntry> {
        entries
            .iter()
            .filter(|e| self.model_exists(&e.name))
            .cloned()
            .collect()
    }
}

