use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct AnthropicMetadata {
    pub user_id: Option<String>
}