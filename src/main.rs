mod auth;
mod config;
mod converters;
mod llm_client;
mod logging;
mod model_checks;
mod model_manager;
mod models;
mod request_id;
mod router;
mod utils;

use axum::{
    Router,
    routing::{get, post},
};
use clap::Parser;
use config::Config;
use router::{anthropic_chat, gemini_chat, list_models, openai_chat, responses_chat};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{Level, info};

#[derive(Parser, Debug)]
#[command(name = "llm-router")]
#[command(about = "A router for LLM API requests")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0")]
    ip: String,

    #[arg(short, long, default_value = "8000")]
    port: u16,

    /// Path to config file
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(short, long)]
    token: Option<String>,

    /// trace, debug, info, warn, error
    #[arg(short, long, default_value = "warn")]
    log_level: String,

    /// Also write logs to this file (max 10MB)
    #[arg(long)]
    log_file: Option<String>,

    /// socks and http proxy, example: socks5://192.168.0.2:10080
    #[arg(long)]
    proxy: Option<String>,

    /// Check availability of all models in config and exit
    #[arg(long)]
    check: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    let ip = args.ip;
    let port = args.port;

    // Parse log level
    let log_level = Level::from_str(&args.log_level).unwrap_or_else(|_| {
        eprintln!("Invalid log level: {}. Using INFO level.", args.log_level);
        Level::INFO
    });

    // Initialize logging: always log to stdout, optionally also to file (capped at 10MB)
    logging::init_logging(log_level, args.log_file.as_deref());

    // Load configuration
    let config_path = args.config.clone();
    let config = Arc::new(Config::from_file(&config_path)?);
    info!("Configuration loaded successfully from: {}", config_path);

    // Create a reqwest client
    let client_builder = reqwest::Client::builder();
    let client_builder = if let Some(proxy) = &args.proxy {
        let proxy = reqwest::Proxy::all(proxy).expect("Failed to create proxy");
        client_builder.proxy(proxy)
    } else {
        client_builder
    };
    let http_client = Arc::new(client_builder.build().expect("Failed to build HTTP client"));

    // Create LlmClient
    let llm_client = Arc::new(llm_client::LlmClient::new(http_client));

    // If --check is provided, verify all models and exit
    if args.check {
        model_checks::perform_model_checks(&config, &llm_client).await?;
        return Ok(());
    }

    // Create model manager with RwLock for dynamic updates
    let model_manager = Arc::new(RwLock::new(model_manager::ModelManager::new(
        config.clone(),
    )));

    // Create app state with model manager and token
    let app_state = auth::AppState {
        model_manager: model_manager.clone(),
        token: args.token,
        llm_client,
    };

    // Create router
    let app = Router::new()
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/messages", post(anthropic_chat))
        .route("/v1/responses", post(responses_chat))
        .route("/v1beta/models/{*tail}", post(gemini_chat))
        .route("/v1/models", get(list_models))
        .route("/health", get(|| async { "OK" }))
        .layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            auth::require_authorization,
        ))
        .layer(CorsLayer::permissive())
        .layer(axum::middleware::from_fn(request_id::inject_request_id))
        .with_state(app_state);

    // Start server
    let bind_address = format!("{}:{}", ip, port);
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;
    info!("Server started on http://{}", bind_address);

    // Graceful shutdown: stop accepting new connections on Ctrl+C/SIGTERM
    // and wait for in-flight requests to complete.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

// Waits for Ctrl+C (all platforms) or SIGTERM (unix) and returns.
async fn shutdown_signal() {
    // Listen for Ctrl+C
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("Failed to listen for Ctrl+C: {}", e);
        }
    };

    // Also listen for SIGTERM on Unix
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => tracing::error!("Failed to install SIGTERM handler: {}", e),
        }
    };

    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tracing::info!("Shutdown signal handler armed. Press Ctrl+C to stop.");
    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Ctrl+C received. Starting graceful shutdown...");
        }
        _ = term => {
            tracing::info!("SIGTERM received. Starting graceful shutdown...");
        }
    }
}

// (moved perform_model_checks and logging helpers to separate modules)
