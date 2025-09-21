mod auth;
mod config;
mod converters;
mod models;
mod model_manager;
mod router;
mod llm_client;
mod utils;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use config::Config;
use router::{anthropic_chat, openai_chat, gemini_chat, list_models};
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

    /// Check availability of all models in config and exit
    #[arg(long)]
    check: bool,
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
        perform_model_checks(&config, &llm_client).await?;
        return Ok(());
    }

    // Create model manager with RwLock for dynamic updates
    let model_manager = Arc::new(RwLock::new(model_manager::ModelManager::new(config.clone())));

    // Start config file watcher in a separate task
    // let config_path_for_watcher = config_path.clone();
    // let model_manager_for_watcher = model_manager.clone();
    // tokio::spawn(async move {
    //     if let Err(e) = watch_config_file(&config_path_for_watcher, &model_manager_for_watcher).await {
    //         warn!("Config file watcher error: {}", e);
    //     }
    // });

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
        .route("/v1beta/models/{*tail}", post(gemini_chat))
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

async fn perform_model_checks(
    config: &Arc<Config>,
    llm_client: &Arc<llm_client::LlmClient>,
) -> anyhow::Result<()> {
    use crate::config::ApiType;
    use crate::converters::request_wrapper::RequestWrapper;
    use crate::converters::openai::{OpenAIRequest, OpenAIMessage, OpenAIContent};
    use crate::converters::anthropic::{AnthropicRequest, AnthropicMessage, AnthropicContent};
    use crate::converters::gemini::{GeminiRequest, gemini_content::GeminiContent, gemini_part::GeminiPart, gemini_generation_config::GeminiGenerationConfig};
    use futures::stream::{self, StreamExt};

    println!("Checking models ({} total):", config.model_list.len());
    let concurrency: usize = 20;
    let client = llm_client.clone();
    let tasks = stream::iter(config.model_list.iter().cloned()).map(|mc| {
        let client = client.clone();
        async move {
            let request = match mc.llm_params.api_type {
                ApiType::OpenAI => {
                    let req = OpenAIRequest {
                        model: mc.model_name.clone(),
                        messages: vec![OpenAIMessage {
                            role: "user".to_string(),
                            content: OpenAIContent::Text("ping".to_string()),
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        }],
                        max_tokens: Some(1),
                        temperature: Some(0.0),
                        response_format: None,
                        tools: None,
                        stream: Some(false),
                        extra_fields: std::collections::HashMap::new(),
                    };
                    RequestWrapper::OpenAI(req)
                }
                ApiType::Anthropic => {
                    let req = AnthropicRequest {
                        model: mc.model_name.clone(),
                        max_tokens: 1,
                        messages: Some(vec![AnthropicMessage { role: "user".to_string(), content: AnthropicContent::Text("ping".to_string()) }]),
                        system: None,
                        tools: None,
                        metadata: None,
                        stream: Some(false),
                        temperature: Some(0.0),
                        extra_fields: std::collections::HashMap::new(),
                    };
                    RequestWrapper::Anthropic(req)
                }
                ApiType::Gemini => {
                    let req = GeminiRequest {
                        model: mc.model_name.clone(),
                        contents: vec![GeminiContent { role: Some("user".to_string()), parts: vec![GeminiPart::Text { text: "ping".to_string(), thought: None, thought_signature: None }] }],
                        system_instruction: None,
                        tools: None,
                        generation_config: Some(GeminiGenerationConfig { response_mime_type: None, response_schema: None, temperature: Some(0.0), max_output_tokens: Some(1), ..Default::default() }),
                        stream: Some(false),
                        extra_fields: std::collections::HashMap::new(),
                    };
                    RequestWrapper::Gemini(req)
                }
            };

            let result = client.forward_request(&request, &mc).await;
            match result {
                Ok(resp) => {
                    if resp.status().is_success() {
                        println!(
                            "[OK] {} -> {} ({})",
                            mc.model_name,
                            mc.llm_params.model,
                            match mc.llm_params.api_type { ApiType::OpenAI => "openai", ApiType::Anthropic => "anthropic", ApiType::Gemini => "gemini" }
                        );
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
                        println!(
                            "[FAIL] {} -> {} (status: {})\n  {}",
                            mc.model_name, mc.llm_params.model, status, truncate(&body, 500)
                        );
                    }
                }
                Err(e) => {
                    println!(
                        "[ERROR] {} -> {}: {}",
                        mc.model_name, mc.llm_params.model, e
                    );
                }
            }
        }
    })
    .buffer_unordered(concurrency)
    .collect::<Vec<()>>();

    tasks.await;
    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len { s.to_string() } else { format!("{}…", &s[..max_len]) }
}
