use serde::{Deserialize, Serialize};

use super::responses_function_tool_choice::ResponsesFunctionToolChoice;
use super::responses_hosted_tool_choice::ResponsesHostedToolChoice;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Mode(String),
    Hosted(ResponsesHostedToolChoice),
    Function(ResponsesFunctionToolChoice),
}
