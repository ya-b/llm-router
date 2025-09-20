use crate::llm_client::LlmClient;
use crate::model_manager::ModelManager;
use crate::models::{ErrorDetail, ErrorResponse};
use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct AppState {
    pub model_manager: Arc<RwLock<ModelManager>>,
    pub token: Option<String>,
    pub llm_client: Arc<LlmClient>,
}

pub async fn require_authorization(
    State(app_state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    // Skip authorization for health check endpoint
    if request.uri().path() == "/health" || request.uri().path() == "/v1/models" {
        return next.run(request).await;
    }

    // If no token is configured, skip authorization
    if app_state.token.is_none() {
        return next.run(request).await;
    }

    let path = request.uri().path();
    let provided_token = if path.starts_with("/v1/chat/completions") {
        request
            .headers()
            .get("Authorization")
            .and_then(|hv| hv.to_str().ok())
            .map(|s| s.trim())
            .and_then(|s| s.strip_prefix("Bearer ").map(|t| t.trim()))
            .map(|s| s)
    } else if path.starts_with("/v1/messages") {
        request
            .headers()
            .get("x-api-key")
            .and_then(|hv| hv.to_str().ok())
            .map(|s| s.trim())
            .map(|s| s)
    } else if path.starts_with("/models/") {
        request
            .uri()
            .query()
            .and_then(|q| {
                // simple parse without external deps
                for pair in q.split('&') {
                    let mut it = pair.splitn(2, '=');
                    if let (Some(k), Some(v)) = (it.next(), it.next()) {
                        if k == "key" {
                            return Some(v);
                        }
                    }
                }
                None
            })
            .or_else(|| {
                request
                    .headers()
                    .get("x-goog-api-key")
                    .and_then(|hv| hv.to_str().ok())
                    .map(|s| s.trim())
                    .map(|s| s)
            })
    } else {
        None
    };

    if provided_token.is_none() {
        info!("Missing authentication token for path: {}", path);
        let error_response = ErrorResponse {
            error: ErrorDetail {
                message: format!("Authentication token is required"),
                r#type: "invalid_request_error".to_string(),
                code: Some("missing_auth_token".to_string()),
            },
        };
        return (StatusCode::UNAUTHORIZED, Json(error_response)).into_response();
    }

    // Validate token
    if provided_token.as_deref() != app_state.token.as_deref() {
        info!("Invalid token provided");
        let error_response = ErrorResponse {
            error: ErrorDetail {
                message: "Invalid authentication token".to_string(),
                r#type: "invalid_request_error".to_string(),
                code: Some("invalid_token".to_string()),
            },
        };
        return (StatusCode::UNAUTHORIZED, Json(error_response)).into_response();
    }

    debug!("Token validation successful");
    next.run(request).await
}
