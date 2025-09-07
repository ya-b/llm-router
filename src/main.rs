mod auth;
mod config;
mod models;
mod router;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use config::Config;
use router::{chat_completion, health_check};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, Level};
use tracing_subscriber;
use notify::{RecursiveMode, Watcher, EventKind};
use std::path::Path;
use tokio::sync::mpsc;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "llm-router")]
#[command(about = "A router for LLM API requests")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0")]
    ip: String,

    #[arg(short, long, default_value = "4000")]
    port: u16,

    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(short, long)]
    token: Option<String>,

    #[arg(short, long, default_value = "warn")]
    log_level: String,
}

async fn watch_config_file(config_path: &str, config: Arc<RwLock<Config>>) -> anyhow::Result<()> {
    let (tx, mut rx) = mpsc::channel(100);

    let mut watcher = notify::recommended_watcher(move |res| {
        if let Ok(event) = res {
            if let Err(e) = tx.blocking_send(event) {
                eprintln!("Failed to send event: {}", e);
            }
        }
    })?;

    watcher.watch(Path::new(config_path), RecursiveMode::NonRecursive)?;

    while let Some(event) = rx.recv().await {
        if let EventKind::Modify(_) = event.kind {
            info!("Config file modified, attempting to reload");
            match Config::from_file(config_path) {
                Ok(new_config) => {
                    let mut config_write = config.write().await;
                    *config_write = new_config;
                    info!("Configuration reloaded successfully");
                }
                Err(e) => {
                    warn!("Failed to reload configuration: {}", e);
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    let ip = args.ip;
    let port = args.port;

    // Parse log level
    let log_level = match args.log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => {
            eprintln!("Invalid log level: {}. Using INFO level.", args.log_level);
            Level::INFO
        }
    };

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .init();

    // Load configuration
    let config_path = args.config.clone();
    let config = Config::from_file(&config_path)?;
    info!("Configuration loaded successfully from: {}", config_path);

    // Create shared state with RwLock for dynamic updates
    let config_state = Arc::new(RwLock::new(config));

    // Clone the config state for the config watcher
    let config_watcher_state = config_state.clone();

    // Start config file watcher in a separate task
    tokio::spawn(async move {
        if let Err(e) = watch_config_file(&config_path, config_watcher_state).await {
            warn!("Config file watcher error: {}", e);
        }
    });

    // Create app state with config and token
    let app_state = auth::AppState {
        config: config_state,
        token: args.token,
    };

    // Create router
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completion))
        .route("/health", get(health_check))
        .layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            auth::require_authorization,
        ))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    // Start server
    let bind_address = format!("{}:{}", ip, port);
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;
    info!("Server started on http://{}", bind_address);

    axum::serve(listener, app).await?;
    Ok(())
}
