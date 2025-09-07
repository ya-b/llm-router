use crate::config::Config;
use crate::models::{ErrorResponse, ErrorDetail};
use axum::{
    body::Body,
    extract::State,
    http::{StatusCode, Request},
    response::{IntoResponse, Response},
    Json,
    middleware::Next,
};
use tracing::{warn, info};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct AppState {
    pub config: Arc<RwLock<Config>>,
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
    let auth_header = request.headers().get("Authorization");
    let provided_token = match auth_header {
        Some(header_value) => {
            let header_str = match header_value.to_str() {
                Ok(s) => s,
                Err(_) => {
                    warn!("Invalid Authorization header format");
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: "Invalid Authorization header format".to_string(),
                            r#type: "invalid_request_error".to_string(),
                            code: Some("invalid_auth_header".to_string()),
                        },
                    };
                    return Err((StatusCode::UNAUTHORIZED, Json(error_response)).into_response());
                }
            };
            
            // Check if header starts with "Bearer "
            if let Some(bearer_token) = header_str.strip_prefix("Bearer ") {
                Some(bearer_token)
            } else {
                warn!("Authorization header does not use Bearer format");
                let error_response = ErrorResponse {
                    error: ErrorDetail {
                        message: "Authorization header must use Bearer format".to_string(),
                        r#type: "invalid_request_error".to_string(),
                        code: Some("invalid_auth_format".to_string()),
                    },
                };
                return Err((StatusCode::UNAUTHORIZED, Json(error_response)).into_response());
            }
        }
        None => {
            warn!("Missing Authorization header");
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
        warn!("Invalid token provided");
        let error_response = ErrorResponse {
            error: ErrorDetail {
                message: "Invalid authentication token".to_string(),
                r#type: "invalid_request_error".to_string(),
                code: Some("invalid_token".to_string()),
            },
        };
        return Err((StatusCode::UNAUTHORIZED, Json(error_response)).into_response());
    }
    
    info!("Token validation successful");
    Ok(next.run(request).await)
}