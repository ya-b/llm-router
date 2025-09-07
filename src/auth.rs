use crate::model_manager::ModelManager;
use crate::models::{ErrorResponse, ErrorDetail};
use axum::{
    body::Body,
    extract::State,
    http::{StatusCode, Request},
    response::{IntoResponse, Response},
    Json,
    middleware::Next,
};
use tracing::{info, debug};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct AppState {
    pub model_manager: Arc<RwLock<ModelManager>>,
    pub token: Option<String>,
}

pub async fn require_authorization(
    State(app_state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, Response> {
    // Skip authorization for health check endpoint
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    // If no token is configured, skip authorization
    if app_state.token.is_none() {
        return Ok(next.run(request).await);
    }

    // Extract token from Authorization header
    let token = request.headers()
        .get("Authorization")
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer ").map(|t| t.trim()))
        .or_else(|| request.headers().get("x-api-key").and_then(|hv| hv.to_str().ok()));
    let provided_token = match token {
        Some(header_str) => {
            Some(header_str)
        }
        None => {
            info!("Missing Authorization header");
            let error_response = ErrorResponse {
                error: ErrorDetail {
                    message: "Authorization header is required".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    code: Some("missing_auth_header".to_string()),
                },
            };
            return Err((StatusCode::UNAUTHORIZED, Json(error_response)).into_response());
        }
    };
    
    // Validate token
    if provided_token != Some(app_state.token.as_ref().unwrap().as_str()) {
        info!("Invalid token provided");
        let error_response = ErrorResponse {
            error: ErrorDetail {
                message: "Invalid authentication token".to_string(),
                r#type: "invalid_request_error".to_string(),
                code: Some("invalid_token".to_string()),
            },
        };
        return Err((StatusCode::UNAUTHORIZED, Json(error_response)).into_response());
    }
    
    debug!("Token validation successful");
    Ok(next.run(request).await)
}