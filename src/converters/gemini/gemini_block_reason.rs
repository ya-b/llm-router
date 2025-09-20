use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GeminiBlockReason {
    BlockReasonUnspecified,
    Safety,
    Other,
    Blocklist,
    ProhibitedContent,
    ImageSafety,
}