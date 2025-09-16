mod auth;
mod config;
mod converters;
mod models;
mod model_manager;
mod router;
mod llm_client;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use config::Config;
use router::{anthropic_chat, openai_chat, list_models};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn, Level};
use std::str::FromStr;
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

    /// socks and http proxy, example: socks5://192.168.0.2:10080
    #[arg(long)]
    proxy: Option<String>,
}

async fn watch_config_file(
    config_path: &str,
    model_manager: &Arc<tokio::sync::RwLock<model_manager::ModelManager>>
) -> anyhow::Result<()> {
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
                    // Update model manager with new config and reset counters
                    let mut model_manager_write = model_manager.write().await;
                    model_manager_write.update_config(Arc::new(new_config));
                    info!("Configuration reloaded successfully");
                }
                Err(e) => {
                    error!("Failed to reload configuration: {}", e);
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
    let log_level = Level::from_str(&args.log_level).unwrap_or_else(|_| {
        eprintln!("Invalid log level: {}. Using INFO level.", args.log_level);
        Level::INFO
    });

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .init();

    // Load configuration
    let config_path = args.config.clone();
    // Create model manager with RwLock for dynamic updates
    let model_manager = Arc::new(RwLock::new(model_manager::ModelManager::new(Arc::new(Config::from_file(&config_path)?))));
    info!("Configuration loaded successfully from: {}", config_path);

    // Start config file watcher in a separate task
    let config_path_for_watcher = config_path.clone();
    let model_manager_for_watcher = model_manager.clone();
    tokio::spawn(async move {
        if let Err(e) = watch_config_file(&config_path_for_watcher, &model_manager_for_watcher).await {
            warn!("Config file watcher error: {}", e);
        }
    });

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
        .route("/v1/models", get(list_models))
        .route("/health", get(|| async { "OK" }))
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
