use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};
use tracing::{Instrument, info_span};
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct RequestId(pub String);

pub async fn inject_request_id(mut req: Request, next: Next) -> Response {
    // Use incoming x-request-id if provided, else generate a new one
    let id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    // Ensure header is present for downstream handlers
    if !req.headers().contains_key("x-request-id") {
        if let Ok(val) = HeaderValue::from_str(&id) {
            req.headers_mut().insert("x-request-id", val);
        }
    }

    // Also store in request extensions for easy extraction
    req.extensions_mut().insert(RequestId(id.clone()));

    // Create a span carrying trace_id for log correlation
    let span = info_span!(
        "http_request",
        trace_id = %id,
        method = %req.method(),
        path = %req.uri().path()
    );

    let mut resp = next.run(req).instrument(span).await;

    // Reflect x-request-id back to the client
    if let Ok(val) = HeaderValue::from_str(&id) {
        resp.headers_mut().insert("x-request-id", val);
    }

    resp
}
