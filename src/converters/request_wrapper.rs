use super::anthropic::AnthropicRequest;
use super::gemini::GeminiRequest;
use super::openai::OpenAIRequest;
use super::responses::ResponsesRequest;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestWrapper {
    OpenAI(OpenAIRequest),
    Anthropic(AnthropicRequest),
    Gemini(GeminiRequest),
    Responses(ResponsesRequest),
}

impl RequestWrapper {
    pub fn get_openai(&self) -> OpenAIRequest {
        match self {
            RequestWrapper::OpenAI(req) => req.clone(),
            RequestWrapper::Anthropic(req) => req.clone().into(),
            RequestWrapper::Gemini(req) => req.clone().into(),
            RequestWrapper::Responses(req) => req.clone().into(),
        }
    }

    pub fn get_anthropic(&self) -> AnthropicRequest {
        match self {
            RequestWrapper::Anthropic(req) => req.clone(),
            RequestWrapper::OpenAI(req) => req.clone().into(),
            RequestWrapper::Gemini(req) => {
                let oai: OpenAIRequest = req.clone().into();
                oai.into()
            }
            RequestWrapper::Responses(req) => {
                let oai: OpenAIRequest = req.clone().into();
                oai.into()
            }
        }
    }

    pub fn get_gemini(&self) -> GeminiRequest {
        match self {
            RequestWrapper::Gemini(req) => req.clone(),
            RequestWrapper::OpenAI(req) => req.clone().into(),
            RequestWrapper::Anthropic(req) => {
                let oai: OpenAIRequest = req.clone().into();
                oai.into()
            }
            RequestWrapper::Responses(req) => {
                let oai: OpenAIRequest = req.clone().into();
                oai.into()
            }
        }
    }

    pub fn get_responses(&self) -> ResponsesRequest {
        match self {
            RequestWrapper::Responses(req) => req.clone(),
            RequestWrapper::OpenAI(req) => req.clone().into(),
            RequestWrapper::Anthropic(req) => {
                let oai: OpenAIRequest = req.clone().into();
                oai.into()
            }
            RequestWrapper::Gemini(req) => {
                let oai: OpenAIRequest = req.clone().into();
                oai.into()
            }
        }
    }

    pub fn get_model(&self) -> &String {
        match self {
            RequestWrapper::OpenAI(req) => &req.model,
            RequestWrapper::Anthropic(req) => &req.model,
            RequestWrapper::Gemini(req) => &req.model,
            RequestWrapper::Responses(req) => &req.model,
        }
    }

    pub fn is_stream(&self) -> &Option<bool> {
        match self {
            RequestWrapper::OpenAI(req) => &req.stream,
            RequestWrapper::Anthropic(req) => &req.stream,
            RequestWrapper::Gemini(req) => &req.stream,
            RequestWrapper::Responses(req) => &req.stream,
        }
    }
}
