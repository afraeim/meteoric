use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::{ipc::Channel, State};
use tokio_util::sync::CancellationToken;

use crate::{database, history::Database};

/// Default configuration constants as the application currently lacks a Settings UI.
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
pub const DEFAULT_MODEL_NAME: &str = "gemma4:e2b";
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-3-7-sonnet-latest";
pub const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";
pub const DEFAULT_PERPLEXITY_MODEL: &str = "sonar";
pub const DEFAULT_OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_OPENROUTER_API_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_GEMINI_API_BASE_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/openai";
pub const DEFAULT_PERPLEXITY_API_BASE_URL: &str = "https://api.perplexity.ai";
const DEFAULT_SYSTEM_PROMPT: &str = include_str!("../prompts/system_prompt.txt");

const CONFIG_PROVIDER_KEY: &str = "ai_provider";
const CONFIG_OPENAI_API_KEY: &str = "openai_api_key";
const CONFIG_ANTHROPIC_API_KEY: &str = "anthropic_api_key";
const CONFIG_GEMINI_API_KEY: &str = "gemini_api_key";
const CONFIG_PERPLEXITY_API_KEY: &str = "perplexity_api_key";
const CONFIG_OPENAI_MODEL: &str = "openai_model";
const CONFIG_ANTHROPIC_MODEL: &str = "anthropic_model";
const CONFIG_GEMINI_MODEL: &str = "gemini_model";
const CONFIG_PERPLEXITY_MODEL: &str = "perplexity_model";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AiProvider {
    Ollama,
    OpenAI,
    Anthropic,
    Gemini,
    Perplexity,
}

#[derive(Clone)]
pub struct ProviderConfig {
    pub active: AiProvider,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub perplexity_api_key: Option<String>,
    pub openai_base_url: String,
    pub gemini_base_url: String,
    pub perplexity_base_url: String,
    pub openrouter_http_referer: Option<String>,
    pub openrouter_x_title: Option<String>,
}

pub struct ProviderConfigState(pub Mutex<ProviderConfig>);

/// Classifies the kind of error returned from the Ollama backend.
/// Used by the frontend to pick accent bar color and display copy.
#[derive(Clone, Serialize, PartialEq, Debug)]
#[serde(rename_all = "PascalCase")]
pub enum OllamaErrorKind {
    /// Ollama process is not running (connection refused / timeout).
    NotRunning,
    /// The requested model has not been pulled yet (HTTP 404).
    ModelNotFound,
    /// Any other unexpected error.
    Other,
}

/// Structured error emitted over the streaming channel.
/// Rust owns all user-facing copy; the frontend only uses `kind` for styling.
#[derive(Clone, Serialize, Debug)]
pub struct OllamaError {
    pub kind: OllamaErrorKind,
    /// Final user-facing string. First line is the title, remainder is the subtitle.
    pub message: String,
}

/// Maps an HTTP status code to a user-friendly `OllamaError`.
pub fn classify_http_error(status: u16) -> OllamaError {
    match status {
        404 => OllamaError {
            kind: OllamaErrorKind::ModelNotFound,
            message: format!(
                "Model not found\nRun: ollama pull {DEFAULT_MODEL_NAME} in a terminal."
            ),
        },
        _ => OllamaError {
            kind: OllamaErrorKind::Other,
            message: format!("Something went wrong\nHTTP {status}"),
        },
    }
}

/// Maps a reqwest connection/transport error to a user-friendly `OllamaError`.
pub fn classify_stream_error(e: &reqwest::Error) -> OllamaError {
    if e.is_connect() || e.is_timeout() {
        OllamaError {
            kind: OllamaErrorKind::NotRunning,
            message: "Ollama isn't running\nStart Ollama and try again.".to_string(),
        }
    } else {
        OllamaError {
            kind: OllamaErrorKind::Other,
            message: "Something went wrong\nCould not reach Ollama.".to_string(),
        }
    }
}

/// Payload emitted back to the frontend per token chunk.
#[derive(Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamChunk {
    /// A single token chunk string.
    Token(String),
    /// A single thinking/reasoning token chunk string.
    ThinkingToken(String),
    /// Indicates the stream has fully completed.
    Done,
    /// The user explicitly cancelled generation.
    Cancelled,
    /// A structured, user-friendly error occurred during processing.
    Error(OllamaError),
}

/// A single message in the Ollama `/api/chat` conversation format.
///
/// The optional `images` field carries base64-encoded image data for
/// multimodal models. When absent or empty, the message is text-only.
#[derive(Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

/// Sampling parameters for Ollama `/api/chat`, following Google's recommended
/// configuration for Gemma4 models.
#[derive(Serialize)]
struct OllamaOptions {
    temperature: f64,
    top_p: f64,
    top_k: u32,
}

/// Request payload for Ollama `/api/chat` endpoint.
#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    think: bool,
    options: OllamaOptions,
}

/// Nested message object in Ollama `/api/chat` response chunks.
#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    content: Option<String>,
    thinking: Option<String>,
}

/// Expected structured response chunk from Ollama `/api/chat`.
#[derive(Deserialize)]
struct OllamaChatResponse {
    message: Option<OllamaChatResponseMessage>,
    done: Option<bool>,
}

/// Holds the active cancellation token for the current generation request.
///
/// Only one generation runs at a time — starting a new request replaces the
/// previous token. `cancel_generation` cancels whatever is currently active.
#[derive(Default)]
pub struct GenerationState {
    token: Mutex<Option<CancellationToken>>,
}

impl GenerationState {
    /// Creates a new empty generation state with no active token.
    pub fn new() -> Self {
        Self {
            token: Mutex::new(None),
        }
    }

    /// Stores a new cancellation token, replacing any previous one.
    fn set(&self, token: CancellationToken) {
        *self.token.lock().unwrap() = Some(token);
    }

    /// Cancels the active generation, if any, and clears the stored token.
    pub fn cancel(&self) {
        if let Some(token) = self.token.lock().unwrap().take() {
            token.cancel();
        }
    }

    /// Clears the stored token without cancelling it (used on natural completion).
    fn clear(&self) {
        *self.token.lock().unwrap() = None;
    }
}

/// Backend-managed conversation history with an epoch counter to prevent
/// stale writes after a reset. The Rust side is the source of truth; the
/// frontend sends only new user messages and receives streamed tokens.
pub struct ConversationHistory {
    pub messages: Mutex<Vec<ChatMessage>>,
    pub epoch: AtomicU64,
}

impl Default for ConversationHistory {
    fn default() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            epoch: AtomicU64::new(0),
        }
    }
}

impl ConversationHistory {
    /// Creates a new empty conversation history at epoch 0.
    pub fn new() -> Self {
        Self::default()
    }
}

/// System prompt loaded once at startup from `METEORIC_SYSTEM_PROMPT`
/// (falls back to `METEORIC_SYSTEM_PROMPT` for migration compatibility).
/// environment variable, falling back to a built-in default.
pub struct SystemPrompt(pub String);

fn env_non_empty(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        std::env::var(key)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    })
}

/// Reads `METEORIC_SYSTEM_PROMPT` (or legacy `METERIC_SYSTEM_PROMPT`) from
/// the environment, falling back to the
/// built-in default when unset or empty.
pub fn load_system_prompt() -> String {
    env_non_empty(&["METEORIC_SYSTEM_PROMPT", "METERIC_SYSTEM_PROMPT"])
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string())
}

pub fn load_provider_config_with_db(conn: Option<&rusqlite::Connection>) -> ProviderConfig {
    let active = config_or_env(
        conn,
        CONFIG_PROVIDER_KEY,
        &["METEORIC_AI_PROVIDER", "METERIC_AI_PROVIDER"],
    )
    .map(|raw| provider_from_str(&raw))
    .unwrap_or(AiProvider::OpenAI);

    let openai_api_key = config_or_env(conn, CONFIG_OPENAI_API_KEY, &["OPENAI_API_KEY"]);
    let openai_base_url = env_non_empty(&["METEORIC_OPENAI_BASE_URL", "METERIC_OPENAI_BASE_URL"])
        .unwrap_or_else(|| {
            if openai_api_key
                .as_deref()
                .is_some_and(|k| k.starts_with("sk-or-v1-"))
            {
                DEFAULT_OPENROUTER_API_BASE_URL.to_string()
            } else {
                DEFAULT_OPENAI_API_BASE_URL.to_string()
            }
        });

    let anthropic_api_key =
        config_or_env(conn, CONFIG_ANTHROPIC_API_KEY, &["ANTHROPIC_API_KEY"]);
    let gemini_api_key = config_or_env(conn, CONFIG_GEMINI_API_KEY, &["GEMINI_API_KEY"]);
    let perplexity_api_key =
        config_or_env(conn, CONFIG_PERPLEXITY_API_KEY, &["PERPLEXITY_API_KEY"]);
    let gemini_base_url =
        env_non_empty(&["METEORIC_GEMINI_BASE_URL", "METERIC_GEMINI_BASE_URL"])
            .unwrap_or_else(|| DEFAULT_GEMINI_API_BASE_URL.to_string());
    let perplexity_base_url = env_non_empty(&[
        "METEORIC_PERPLEXITY_BASE_URL",
        "METERIC_PERPLEXITY_BASE_URL",
    ])
    .unwrap_or_else(|| DEFAULT_PERPLEXITY_API_BASE_URL.to_string());

    ProviderConfig {
        active,
        openai_api_key,
        anthropic_api_key,
        gemini_api_key,
        perplexity_api_key,
        openai_base_url,
        gemini_base_url,
        perplexity_base_url,
        openrouter_http_referer: env_non_empty(&[
            "METEORIC_OPENROUTER_HTTP_REFERER",
            "METERIC_OPENROUTER_HTTP_REFERER",
        ]),
        openrouter_x_title: env_non_empty(&[
            "METEORIC_OPENROUTER_X_TITLE",
            "METERIC_OPENROUTER_X_TITLE",
        ]),
    }
}

pub fn load_provider_config() -> ProviderConfig {
    load_provider_config_with_db(None)
}

/// Model configuration loaded once at startup from `METEORIC_SUPPORTED_AI_MODELS`
/// (falls back to `METERIC_SUPPORTED_AI_MODELS` for migration compatibility).
/// environment variable (comma-separated list). The first entry is the active model
/// used for inference. Falls back to `DEFAULT_MODEL_NAME` when unset or empty.
#[derive(Clone)]
pub struct ModelConfig {
    pub active: String,
    pub all: Vec<String>,
}

pub struct ModelConfigState(pub Mutex<ModelConfig>);

fn provider_from_str(raw: &str) -> AiProvider {
    match raw.to_ascii_lowercase().as_str() {
        "openai" => AiProvider::OpenAI,
        "anthropic" => AiProvider::Anthropic,
        "gemini" => AiProvider::Gemini,
        "perplexity" => AiProvider::Perplexity,
        _ => AiProvider::Ollama,
    }
}

fn provider_to_str(provider: AiProvider) -> &'static str {
    match provider {
        AiProvider::Ollama => "ollama",
        AiProvider::OpenAI => "openai",
        AiProvider::Anthropic => "anthropic",
        AiProvider::Gemini => "gemini",
        AiProvider::Perplexity => "perplexity",
    }
}

fn config_or_env(
    conn: Option<&rusqlite::Connection>,
    config_key: &str,
    env_keys: &[&str],
) -> Option<String> {
    if let Some(conn) = conn {
        if let Ok(Some(v)) = database::get_config(conn, config_key) {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    env_non_empty(env_keys)
}

/// Reads `METEORIC_SUPPORTED_AI_MODELS` (or legacy `METERIC_SUPPORTED_AI_MODELS`)
/// from the environment and returns a
/// `ModelConfig`. Trims whitespace around each entry and filters empty entries.
/// Defaults to `[DEFAULT_MODEL_NAME]` when the variable is unset or empty.
pub fn load_model_config_with_provider(
    conn: Option<&rusqlite::Connection>,
    provider: AiProvider,
) -> ModelConfig {
    let ollama_models: Vec<String> =
        env_non_empty(&["METEORIC_SUPPORTED_AI_MODELS", "METERIC_SUPPORTED_AI_MODELS"])
            .map(|s| {
                s.split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect()
            })
            .unwrap_or_else(|| vec![DEFAULT_MODEL_NAME.to_string()]);

    let models: Vec<String> = match provider {
        AiProvider::Anthropic => vec![
            config_or_env(
                conn,
                CONFIG_ANTHROPIC_MODEL,
                &["METEORIC_ANTHROPIC_MODEL", "METERIC_ANTHROPIC_MODEL"],
            )
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_MODEL.to_string()),
        ],
        AiProvider::OpenAI => vec![
            config_or_env(
                conn,
                CONFIG_OPENAI_MODEL,
                &["METEORIC_OPENAI_MODEL", "METERIC_OPENAI_MODEL"],
            )
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_string()),
        ],
        AiProvider::Gemini => vec![
            config_or_env(
                conn,
                CONFIG_GEMINI_MODEL,
                &["METEORIC_GEMINI_MODEL", "METERIC_GEMINI_MODEL"],
            )
            .unwrap_or_else(|| DEFAULT_GEMINI_MODEL.to_string()),
        ],
        AiProvider::Perplexity => vec![
            config_or_env(
                conn,
                CONFIG_PERPLEXITY_MODEL,
                &[
                    "METEORIC_PERPLEXITY_MODEL",
                    "METERIC_PERPLEXITY_MODEL",
                ],
            )
            .unwrap_or_else(|| DEFAULT_PERPLEXITY_MODEL.to_string()),
        ],
        AiProvider::Ollama => ollama_models,
    };

    let active = models
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());
    ModelConfig {
        active,
        all: models,
    }
}

pub fn load_model_config() -> ModelConfig {
    let provider = load_provider_config().active;
    load_model_config_with_provider(None, provider)
}

/// Returns the active model and full supported list to the frontend.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg_attr(not(coverage), tauri::command)]
pub fn get_model_config(model_config: tauri::State<'_, ModelConfigState>) -> serde_json::Value {
    let cfg = model_config.0.lock().unwrap();
    serde_json::json!({ "active": cfg.active, "all": cfg.all })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderSetupStatus {
    pub active_provider: String,
    pub has_api_key: bool,
    pub needs_setup: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveProviderSetupPayload {
    pub provider: String,
    pub api_key: String,
    pub model: Option<String>,
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg_attr(not(coverage), tauri::command)]
pub fn get_provider_setup_status(
    provider_config: tauri::State<'_, ProviderConfigState>,
) -> ProviderSetupStatus {
    let cfg = provider_config.0.lock().unwrap();
    let has_api_key = match cfg.active {
        AiProvider::OpenAI => cfg.openai_api_key.as_deref().is_some_and(|k| !k.trim().is_empty()),
        AiProvider::Anthropic => cfg
            .anthropic_api_key
            .as_deref()
            .is_some_and(|k| !k.trim().is_empty()),
        AiProvider::Gemini => cfg.gemini_api_key.as_deref().is_some_and(|k| !k.trim().is_empty()),
        AiProvider::Perplexity => cfg
            .perplexity_api_key
            .as_deref()
            .is_some_and(|k| !k.trim().is_empty()),
        AiProvider::Ollama => true,
    };

    ProviderSetupStatus {
        active_provider: provider_to_str(cfg.active).to_string(),
        has_api_key,
        needs_setup: !has_api_key && !matches!(cfg.active, AiProvider::Ollama),
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg_attr(not(coverage), tauri::command)]
pub fn save_provider_setup(
    payload: SaveProviderSetupPayload,
    db: State<'_, Database>,
    provider_config: State<'_, ProviderConfigState>,
    model_config: State<'_, ModelConfigState>,
) -> Result<(), String> {
    let provider = provider_from_str(payload.provider.trim());
    let api_key = payload.api_key.trim();
    if api_key.is_empty() && !matches!(provider, AiProvider::Ollama) {
        return Err("API key is required for this provider".to_string());
    }

    let conn = db.0.lock().map_err(|e| e.to_string())?;
    database::set_config(&conn, CONFIG_PROVIDER_KEY, provider_to_str(provider))
        .map_err(|e| e.to_string())?;

    if !api_key.is_empty() {
        let key_name = match provider {
            AiProvider::OpenAI => Some(CONFIG_OPENAI_API_KEY),
            AiProvider::Anthropic => Some(CONFIG_ANTHROPIC_API_KEY),
            AiProvider::Gemini => Some(CONFIG_GEMINI_API_KEY),
            AiProvider::Perplexity => Some(CONFIG_PERPLEXITY_API_KEY),
            AiProvider::Ollama => None,
        };
        if let Some(key_name) = key_name {
            database::set_config(&conn, key_name, api_key).map_err(|e| e.to_string())?;
        }
    }

    if let Some(model) = payload.model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        let model_key = match provider {
            AiProvider::OpenAI => Some(CONFIG_OPENAI_MODEL),
            AiProvider::Anthropic => Some(CONFIG_ANTHROPIC_MODEL),
            AiProvider::Gemini => Some(CONFIG_GEMINI_MODEL),
            AiProvider::Perplexity => Some(CONFIG_PERPLEXITY_MODEL),
            AiProvider::Ollama => None,
        };
        if let Some(model_key) = model_key {
            database::set_config(&conn, model_key, model).map_err(|e| e.to_string())?;
        }
    }

    let next_provider_cfg = load_provider_config_with_db(Some(&conn));
    let next_model_cfg = load_model_config_with_provider(Some(&conn), next_provider_cfg.active);
    drop(conn);

    *provider_config.0.lock().unwrap() = next_provider_cfg;
    *model_config.0.lock().unwrap() = next_model_cfg;

    Ok(())
}

fn content_for_provider(msg: &ChatMessage) -> String {
    match &msg.images {
        Some(images) if !images.is_empty() => format!(
            "{}\n\n[Note: {} attached image(s) were provided in the original message.]",
            msg.content,
            images.len()
        ),
        _ => msg.content.clone(),
    }
}

fn classify_provider_http_error(status: u16, provider: &str) -> OllamaError {
    if status == 401 {
        return OllamaError {
            kind: OllamaErrorKind::Other,
            message: format!("Authentication failed\nCheck your {provider} API key and try again."),
        };
    }

    OllamaError {
        kind: OllamaErrorKind::Other,
        message: format!("Something went wrong\n{provider} API returned HTTP {status}"),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn stream_openai_chat(
    model: &str,
    messages: Vec<ChatMessage>,
    system_prompt: &str,
    api_key: &str,
    openai_base_url: &str,
    provider_name_override: Option<&str>,
    openrouter_http_referer: Option<&str>,
    openrouter_x_title: Option<&str>,
    client: &reqwest::Client,
    cancel_token: CancellationToken,
    on_chunk: impl Fn(StreamChunk),
) -> String {
    let api_base = openai_base_url.trim_end_matches('/');
    let endpoint = format!("{api_base}/chat/completions");
    let is_openrouter = api_base.contains("openrouter.ai");
    let provider_name = provider_name_override.unwrap_or(if is_openrouter {
        "OpenRouter"
    } else {
        "OpenAI"
    });

    let mut payload_messages = vec![serde_json::json!({
        "role": "system",
        "content": system_prompt,
    })];

    for msg in messages {
        payload_messages.push(serde_json::json!({
            "role": msg.role,
            "content": content_for_provider(&msg),
        }));
    }

    let mut req = client
        .post(&endpoint)
        .bearer_auth(api_key)
        .json(&serde_json::json!({
            "model": model,
            "messages": payload_messages,
            "stream": false,
            "temperature": 0.7,
        }));

    if is_openrouter && provider_name_override.is_none() {
        if let Some(referer) = openrouter_http_referer {
            req = req.header("HTTP-Referer", referer);
        }
        if let Some(title) = openrouter_x_title {
            req = req.header("X-Title", title);
        }
    }

    let response = tokio::select! {
        _ = cancel_token.cancelled() => {
            on_chunk(StreamChunk::Cancelled);
            return String::new();
        }
        result = req.send() => result
    };

    let response = match response {
        Ok(r) => r,
        Err(_) => {
            on_chunk(StreamChunk::Error(OllamaError {
                kind: OllamaErrorKind::Other,
                message: format!("Something went wrong\nCould not reach {provider_name}."),
            }));
            return String::new();
        }
    };

    if !response.status().is_success() {
        on_chunk(StreamChunk::Error(classify_provider_http_error(
            response.status().as_u16(),
            provider_name,
        )));
        return String::new();
    }

    let body = match response.json::<serde_json::Value>().await {
        Ok(v) => v,
        Err(_) => {
            on_chunk(StreamChunk::Error(OllamaError {
                kind: OllamaErrorKind::Other,
                message: format!(
                    "Something went wrong\n{provider_name} returned an invalid response."
                ),
            }));
            return String::new();
        }
    };

    let content = body
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string();

    if !content.is_empty() {
        on_chunk(StreamChunk::Token(content.clone()));
    }
    on_chunk(StreamChunk::Done);
    content
}

pub async fn stream_anthropic_chat(
    model: &str,
    messages: Vec<ChatMessage>,
    system_prompt: &str,
    api_key: &str,
    client: &reqwest::Client,
    cancel_token: CancellationToken,
    on_chunk: impl Fn(StreamChunk),
) -> String {
    let endpoint = "https://api.anthropic.com/v1/messages";
    let payload_messages: Vec<serde_json::Value> = messages
        .into_iter()
        .map(|msg| {
            serde_json::json!({
                "role": msg.role,
                "content": content_for_provider(&msg),
            })
        })
        .collect();

    let req = client
        .post(endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": model,
            "system": system_prompt,
            "max_tokens": 2048,
            "messages": payload_messages,
        }));

    let response = tokio::select! {
        _ = cancel_token.cancelled() => {
            on_chunk(StreamChunk::Cancelled);
            return String::new();
        }
        result = req.send() => result
    };

    let response = match response {
        Ok(r) => r,
        Err(_) => {
            on_chunk(StreamChunk::Error(OllamaError {
                kind: OllamaErrorKind::Other,
                message: "Something went wrong\nCould not reach Anthropic.".to_string(),
            }));
            return String::new();
        }
    };

    if !response.status().is_success() {
        on_chunk(StreamChunk::Error(classify_provider_http_error(
            response.status().as_u16(),
            "Anthropic",
        )));
        return String::new();
    }

    let body = match response.json::<serde_json::Value>().await {
        Ok(v) => v,
        Err(_) => {
            on_chunk(StreamChunk::Error(OllamaError {
                kind: OllamaErrorKind::Other,
                message: "Something went wrong\nAnthropic returned an invalid response."
                    .to_string(),
            }));
            return String::new();
        }
    };

    let content = body
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|parts| {
            parts
                .iter()
                .find(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
        })
        .and_then(|part| part.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .to_string();

    if !content.is_empty() {
        on_chunk(StreamChunk::Token(content.clone()));
    }
    on_chunk(StreamChunk::Done);
    content
}

/// Core streaming logic for Ollama `/api/chat`, separated from the Tauri
/// command for testability. Uses `tokio::select!` to race each chunk read
/// against the cancellation token, ensuring the HTTP connection is dropped
/// immediately when the user cancels — which signals Ollama to stop inference.
/// Returns the accumulated assistant response so the caller can persist it.
pub async fn stream_ollama_chat(
    endpoint: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    think: bool,
    client: &reqwest::Client,
    cancel_token: CancellationToken,
    on_chunk: impl Fn(StreamChunk),
) -> String {
    let request_payload = OllamaChatRequest {
        model: model.to_string(),
        messages,
        stream: true,
        think,
        options: OllamaOptions {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 64,
        },
    };

    let mut accumulated = String::new();

    let res = client.post(endpoint).json(&request_payload).send().await;

    match res {
        Ok(response) => {
            if !response.status().is_success() {
                let status = response.status().as_u16();
                on_chunk(StreamChunk::Error(classify_http_error(status)));
                return accumulated;
            }

            let mut stream = response.bytes_stream();
            let mut buffer: Vec<u8> = Vec::new();

            loop {
                tokio::select! {
                    biased;
                    _ = cancel_token.cancelled() => {
                        // Drop the stream — closes the HTTP connection,
                        // which signals Ollama to stop inference.
                        drop(stream);
                        on_chunk(StreamChunk::Cancelled);
                        return accumulated;
                    }
                    chunk_opt = stream.next() => {
                        match chunk_opt {
                            Some(Ok(bytes)) => {
                                buffer.extend_from_slice(&bytes);

                                while let Some(idx) = buffer.iter().position(|&b| b == b'\n') {
                                    let line_bytes = buffer.drain(..=idx).collect::<Vec<u8>>();
                                    if let Ok(line_text) = String::from_utf8(line_bytes) {
                                        let trimmed = line_text.trim();
                                        if trimmed.is_empty() {
                                            continue;
                                        }

                                        if let Ok(json) =
                                            serde_json::from_str::<OllamaChatResponse>(trimmed)
                                        {
                                            if let Some(ref msg) = json.message {
                                                if let Some(ref thinking) = msg.thinking {
                                                    if !thinking.is_empty() {
                                                        on_chunk(StreamChunk::ThinkingToken(
                                                            thinking.clone(),
                                                        ));
                                                    }
                                                }
                                                if let Some(ref token) = msg.content {
                                                    if !token.is_empty() {
                                                        accumulated.push_str(token);
                                                        on_chunk(StreamChunk::Token(
                                                            token.clone(),
                                                        ));
                                                    }
                                                }
                                            }
                                            if let Some(true) = json.done {
                                                on_chunk(StreamChunk::Done);
                                            }
                                        }
                                    }
                                }
                            }
                            Some(Err(e)) => {
                                on_chunk(StreamChunk::Error(classify_stream_error(&e)));
                                return accumulated;
                            }
                            None => return accumulated,
                        }
                    }
                }
            }
        }
        Err(e) => {
            on_chunk(StreamChunk::Error(classify_stream_error(&e)));
        }
    }

    accumulated
}

/// Streams a chat response from the local Ollama backend. Appends the user
/// message and assistant response to conversation history after completion
/// or cancellation (retaining context for follow-up requests). Uses an epoch
/// counter to prevent stale writes after a reset.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg_attr(not(coverage), tauri::command)]
#[allow(clippy::too_many_arguments)]
pub async fn ask_ollama(
    message: String,
    quoted_text: Option<String>,
    image_paths: Option<Vec<String>>,
    think: bool,
    on_event: Channel<StreamChunk>,
    client: State<'_, reqwest::Client>,
    generation: State<'_, GenerationState>,
    history: State<'_, ConversationHistory>,
    system_prompt: State<'_, SystemPrompt>,
    model_config: State<'_, ModelConfigState>,
    provider_config: State<'_, ProviderConfigState>,
) -> Result<(), String> {
    let cancel_token = CancellationToken::new();
    generation.set(cancel_token.clone());

    // Build user message content.  When quoted text is present, label it
    // explicitly so the model knows the highlighted text is the primary
    // subject and any attached images provide surrounding context.
    let content = match quoted_text {
        Some(ref qt) if !qt.trim().is_empty() => {
            format!("[Highlighted Text]\n\"{qt}\"\n\n[Request]\n{message}")
        }
        _ => message,
    };

    // Base64-encode attached images for the Ollama multimodal API.
    let images = match image_paths {
        Some(ref paths) if !paths.is_empty() => {
            Some(crate::images::encode_images_as_base64(paths)?)
        }
        _ => None,
    };

    let user_msg = ChatMessage {
        role: "user".to_string(),
        content,
        images,
    };

    // Snapshot the current epoch and build the messages array for Ollama.
    // The user message is NOT yet committed to history — it is only added
    // after a response (including partial/cancelled) to prevent orphaned
    // messages on errors.
    let (epoch_at_start, messages) = {
        let conv = history.messages.lock().unwrap();
        let epoch = history.epoch.load(Ordering::SeqCst);
        let mut msgs = vec![ChatMessage {
            role: "system".to_string(),
            content: system_prompt.0.clone(),
            images: None,
        }];
        msgs.extend(conv.clone());
        msgs.push(user_msg.clone());
        (epoch, msgs)
    };

    let provider_cfg = provider_config.0.lock().unwrap().clone();
    let model_cfg = model_config.0.lock().unwrap().clone();

    let accumulated = match &provider_cfg.active {
        AiProvider::Ollama => {
            let endpoint = format!("{}/api/chat", DEFAULT_OLLAMA_URL.trim_end_matches('/'));
            stream_ollama_chat(
                &endpoint,
                &model_cfg.active,
                messages,
                think,
                &client,
                cancel_token.clone(),
                |chunk| {
                    let _ = on_event.send(chunk);
                },
            )
            .await
        }
        AiProvider::OpenAI => {
            let Some(api_key) = provider_cfg.openai_api_key.as_deref() else {
                let _ = on_event.send(StreamChunk::Error(OllamaError {
                    kind: OllamaErrorKind::Other,
                    message: "Missing OpenAI API key\nSet it in Meteoric settings and try again."
                        .to_string(),
                }));
                generation.clear();
                return Ok(());
            };
            stream_openai_chat(
                &model_cfg.active,
                messages,
                &system_prompt.0,
                api_key,
                &provider_cfg.openai_base_url,
                None,
                provider_cfg.openrouter_http_referer.as_deref(),
                provider_cfg.openrouter_x_title.as_deref(),
                &client,
                cancel_token.clone(),
                |chunk| {
                    let _ = on_event.send(chunk);
                },
            )
            .await
        }
        AiProvider::Anthropic => {
            let Some(api_key) = provider_cfg.anthropic_api_key.as_deref() else {
                let _ = on_event.send(StreamChunk::Error(OllamaError {
                    kind: OllamaErrorKind::Other,
                    message: "Missing Anthropic API key\nSet it in Meteoric settings and try again."
                        .to_string(),
                }));
                generation.clear();
                return Ok(());
            };
            stream_anthropic_chat(
                &model_cfg.active,
                messages,
                &system_prompt.0,
                api_key,
                &client,
                cancel_token.clone(),
                |chunk| {
                    let _ = on_event.send(chunk);
                },
            )
            .await
        }
        AiProvider::Gemini => {
            let Some(api_key) = provider_cfg.gemini_api_key.as_deref() else {
                let _ = on_event.send(StreamChunk::Error(OllamaError {
                    kind: OllamaErrorKind::Other,
                    message:
                        "Missing Gemini API key\nSet it in Meteoric settings and try again."
                            .to_string(),
                }));
                generation.clear();
                return Ok(());
            };
            stream_openai_chat(
                &model_cfg.active,
                messages,
                &system_prompt.0,
                api_key,
                &provider_cfg.gemini_base_url,
                Some("Gemini"),
                None,
                None,
                &client,
                cancel_token.clone(),
                |chunk| {
                    let _ = on_event.send(chunk);
                },
            )
            .await
        }
        AiProvider::Perplexity => {
            let Some(api_key) = provider_cfg.perplexity_api_key.as_deref() else {
                let _ = on_event.send(StreamChunk::Error(OllamaError {
                    kind: OllamaErrorKind::Other,
                    message: "Missing Perplexity API key\nSet it in Meteoric settings and try again."
                        .to_string(),
                }));
                generation.clear();
                return Ok(());
            };
            stream_openai_chat(
                &model_cfg.active,
                messages,
                &system_prompt.0,
                api_key,
                &provider_cfg.perplexity_base_url,
                Some("Perplexity"),
                None,
                None,
                &client,
                cancel_token.clone(),
                |chunk| {
                    let _ = on_event.send(chunk);
                },
            )
            .await
        }
    };

    // Persist user + assistant messages to in-memory history when the epoch
    // has not changed (no reset during streaming) and we received content.
    // This includes cancelled generations so that subsequent requests retain
    // the conversational context (the user message and any partial response).
    let current_epoch = history.epoch.load(Ordering::SeqCst);
    if current_epoch == epoch_at_start && !accumulated.is_empty() {
        let mut conv = history.messages.lock().unwrap();
        // Preserve images in history so that follow-up messages can still
        // reference earlier screenshots or attachments.  The full conversation
        // (including base64 blobs) is replayed to Ollama on every turn, which
        // is fine for a localhost-only setup.
        conv.push(user_msg);
        conv.push(ChatMessage {
            role: "assistant".to_string(),
            content: accumulated,
            images: None,
        });
    }

    generation.clear();
    Ok(())
}

/// Cancels the currently active generation, if any.
///
/// Signals the `CancellationToken` stored in `GenerationState`, which causes the
/// `stream_ollama_chat` loop to exit immediately and drop the HTTP connection.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg_attr(not(coverage), tauri::command)]
pub async fn cancel_generation(generation: State<'_, GenerationState>) -> Result<(), String> {
    generation.cancel();
    Ok(())
}

/// Clears the backend conversation history and increments the epoch counter.
/// The epoch increment prevents any in-flight `ask_ollama` from writing stale
/// messages into the freshly cleared history.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg_attr(not(coverage), tauri::command)]
pub fn reset_conversation(history: State<'_, ConversationHistory>) {
    history.epoch.fetch_add(1, Ordering::SeqCst);
    history.messages.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    fn collect_chunks() -> (Arc<StdMutex<Vec<StreamChunk>>>, impl Fn(StreamChunk)) {
        let chunks: Arc<StdMutex<Vec<StreamChunk>>> = Arc::new(StdMutex::new(Vec::new()));
        let chunks_clone = chunks.clone();
        let callback = move |chunk: StreamChunk| {
            chunks_clone.lock().unwrap().push(chunk);
        };
        (chunks, callback)
    }

    /// Helper: builds a `/api/chat` response line from content + done flag.
    fn chat_line(content: &str, done: bool) -> String {
        format!(
            "{{\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"done\":{}}}\n",
            content, done
        )
    }

    #[tokio::test]
    async fn streams_tokens_from_valid_response() {
        let mut server = mockito::Server::new_async().await;
        let body = format!(
            "{}{}{}",
            chat_line("Hello", false),
            chat_line(" world", false),
            chat_line("", true),
        );
        let mock = server
            .mock("POST", "/api/chat")
            .with_body(body)
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
            images: None,
        }];

        let accumulated = stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            messages,
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert!(matches!(&chunks[0], StreamChunk::Token(t) if t == "Hello"));
        assert!(matches!(&chunks[1], StreamChunk::Token(t) if t == " world"));
        assert!(matches!(&chunks[2], StreamChunk::Done));
        assert_eq!(accumulated, "Hello world");
    }

    #[tokio::test]
    async fn handles_http_500() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_status(500)
            .with_body("Internal Server Error")
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        let accumulated = stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::Error(e) if e.kind == OllamaErrorKind::Other));
        assert!(accumulated.is_empty());
    }

    #[tokio::test]
    async fn handles_connection_refused() {
        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        let accumulated = stream_ollama_chat(
            "http://127.0.0.1:1/api/chat",
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        let chunks = chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::Error(_)));
        assert!(accumulated.is_empty());
    }

    #[tokio::test]
    async fn handles_malformed_json() {
        let mut server = mockito::Server::new_async().await;
        let body = format!("not json at all\n{}", chat_line("ok", true));
        let mock = server
            .mock("POST", "/api/chat")
            .with_body(body)
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Done)));
    }

    #[tokio::test]
    async fn handles_empty_response_body() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_body("")
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        let accumulated = stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert!(chunks.is_empty());
        assert!(accumulated.is_empty());
    }

    #[tokio::test]
    async fn tokens_arrive_in_order() {
        let mut server = mockito::Server::new_async().await;
        let body = format!(
            "{}{}{}{}",
            chat_line("A", false),
            chat_line("B", false),
            chat_line("C", false),
            chat_line("", true),
        );
        let mock = server
            .mock("POST", "/api/chat")
            .with_body(body)
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        let accumulated = stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        let tokens: Vec<&str> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, vec!["A", "B", "C"]);
        assert_eq!(accumulated, "ABC");
    }

    #[tokio::test]
    async fn handles_invalid_utf8_in_stream() {
        let mut server = mockito::Server::new_async().await;
        let mut body = b"\xFF\xFE\n".to_vec();
        body.extend_from_slice(chat_line("ok", true).as_bytes());
        let mock = server
            .mock("POST", "/api/chat")
            .with_body(body)
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Done)));
    }

    #[tokio::test]
    async fn handles_mid_stream_network_error() {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: application/x-ndjson\r\n\
                      Transfer-Encoding: chunked\r\n\r\n\
                      4\r\ntest",
                )
                .await;
        });

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("http://127.0.0.1:{}/api/chat", port),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        let chunks = chunks.lock().unwrap();
        let has_no_tokens = chunks.iter().all(|c| !matches!(c, StreamChunk::Token(_)));
        assert!(has_no_tokens);
    }

    #[tokio::test]
    async fn http_500_with_empty_body() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_status(500)
            .with_body("")
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(
            matches!(&chunks[0], StreamChunk::Error(e) if e.kind == OllamaErrorKind::Other && e.message.contains("500"))
        );
    }

    #[tokio::test]
    async fn whitespace_only_lines_are_skipped() {
        let mut server = mockito::Server::new_async().await;
        let body = format!("   \n{}", chat_line("hi", true));
        let mock = server
            .mock("POST", "/api/chat")
            .with_body(body)
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Done)));
    }

    #[tokio::test]
    async fn message_field_absent_emits_only_done() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_body("{\"done\":true}\n")
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert!(chunks.iter().all(|c| !matches!(c, StreamChunk::Token(_))));
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Done)));
    }

    #[tokio::test]
    async fn cancellation_stops_stream_and_emits_cancelled() {
        use std::sync::Arc;
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server_done = Arc::new(tokio::sync::Notify::new());
        let server_done_clone = server_done.clone();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let first_line = chat_line("A", false);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\r\n{}",
                first_line
            );
            let _ = stream.write_all(header.as_bytes()).await;
            server_done_clone.notified().await;
        });

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let token_clone = token.clone();
        let (chunks, callback) = collect_chunks();

        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            token_clone.cancel();
        });

        stream_ollama_chat(
            &format!("http://127.0.0.1:{}/api/chat", port),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        let chunks = chunks.lock().unwrap();
        assert!(chunks
            .iter()
            .any(|c| matches!(c, StreamChunk::Token(t) if t == "A")));
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Cancelled)));
        assert!(chunks.iter().all(|c| !matches!(c, StreamChunk::Done)));

        server_done.notify_one();
        tokio::task::yield_now().await;
    }

    #[tokio::test]
    async fn pre_cancelled_token_emits_cancelled_immediately() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/api/chat")
            .with_body(chat_line("Hello", true))
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        token.cancel();

        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        let chunks = chunks.lock().unwrap();
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Cancelled)));
    }

    #[tokio::test]
    async fn sends_messages_array_in_request() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"hi"}]}"#.to_string(),
            ))
            .with_body(chat_line("", true))
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (_, callback) = collect_chunks();
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Be helpful".to_string(),
                images: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                images: None,
            },
        ];

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            messages,
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn message_content_absent_emits_only_done() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_body("{\"message\":{\"role\":\"assistant\"},\"done\":true}\n")
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert!(chunks.iter().all(|c| !matches!(c, StreamChunk::Token(_))));
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Done)));
    }

    #[test]
    fn generation_state_set_and_cancel() {
        let state = GenerationState::new();
        let token = CancellationToken::new();
        let token_clone = token.clone();

        state.set(token);
        assert!(!token_clone.is_cancelled());

        state.cancel();
        assert!(token_clone.is_cancelled());
    }

    #[test]
    fn generation_state_cancel_when_empty() {
        let state = GenerationState::new();
        state.cancel();
    }

    #[test]
    fn generation_state_clear_does_not_cancel() {
        let state = GenerationState::new();
        let token = CancellationToken::new();
        let token_clone = token.clone();

        state.set(token);
        state.clear();
        assert!(!token_clone.is_cancelled());
    }

    #[test]
    fn generation_state_set_replaces_previous() {
        let state = GenerationState::new();
        let first = CancellationToken::new();
        let first_clone = first.clone();
        let second = CancellationToken::new();
        let second_clone = second.clone();

        state.set(first);
        state.set(second);

        state.cancel();
        assert!(!first_clone.is_cancelled());
        assert!(second_clone.is_cancelled());
    }

    /// Guard to serialize tests that mutate environment variables.
    /// Rust runs tests in parallel by default; without serialization these
    /// tests race on shared environment variables.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    fn set_ollama_provider_for_model_tests() {
        std::env::remove_var("METEORIC_AI_PROVIDER");
        std::env::set_var("METERIC_AI_PROVIDER", "ollama");
    }

    fn clear_provider_env_after_model_tests() {
        std::env::remove_var("METEORIC_AI_PROVIDER");
        std::env::remove_var("METERIC_AI_PROVIDER");
    }

    // ── load_model_config tests ──────────────────────────────────────────────

    #[test]
    fn load_model_config_returns_default_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_ollama_provider_for_model_tests();
        std::env::remove_var("METEORIC_SUPPORTED_AI_MODELS");
        std::env::remove_var("METERIC_SUPPORTED_AI_MODELS");
        let config = load_model_config();
        assert_eq!(config.active, DEFAULT_MODEL_NAME);
        assert_eq!(config.all, vec![DEFAULT_MODEL_NAME.to_string()]);
        clear_provider_env_after_model_tests();
    }

    #[test]
    fn load_model_config_reads_single_model() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_ollama_provider_for_model_tests();
        std::env::remove_var("METEORIC_SUPPORTED_AI_MODELS");
        std::env::set_var("METERIC_SUPPORTED_AI_MODELS", "gemma4:e4b");
        let config = load_model_config();
        assert_eq!(config.active, "gemma4:e4b");
        assert_eq!(config.all, vec!["gemma4:e4b".to_string()]);
        std::env::remove_var("METERIC_SUPPORTED_AI_MODELS");
        clear_provider_env_after_model_tests();
    }

    #[test]
    fn load_model_config_reads_multiple_models_first_is_active() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_ollama_provider_for_model_tests();
        std::env::remove_var("METEORIC_SUPPORTED_AI_MODELS");
        std::env::set_var("METERIC_SUPPORTED_AI_MODELS", "gemma4:e2b,gemma4:e4b");
        let config = load_model_config();
        assert_eq!(config.active, "gemma4:e2b");
        assert_eq!(
            config.all,
            vec!["gemma4:e2b".to_string(), "gemma4:e4b".to_string()]
        );
        std::env::remove_var("METERIC_SUPPORTED_AI_MODELS");
        clear_provider_env_after_model_tests();
    }

    #[test]
    fn load_model_config_trims_whitespace_around_entries() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_ollama_provider_for_model_tests();
        std::env::remove_var("METEORIC_SUPPORTED_AI_MODELS");
        std::env::set_var("METERIC_SUPPORTED_AI_MODELS", " gemma4:e2b , gemma4:e4b ");
        let config = load_model_config();
        assert_eq!(config.active, "gemma4:e2b");
        assert_eq!(
            config.all,
            vec!["gemma4:e2b".to_string(), "gemma4:e4b".to_string()]
        );
        std::env::remove_var("METERIC_SUPPORTED_AI_MODELS");
        clear_provider_env_after_model_tests();
    }

    #[test]
    fn load_model_config_falls_back_to_default_when_whitespace_only() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_ollama_provider_for_model_tests();
        std::env::remove_var("METEORIC_SUPPORTED_AI_MODELS");
        std::env::set_var("METERIC_SUPPORTED_AI_MODELS", "   ");
        let config = load_model_config();
        assert_eq!(config.active, DEFAULT_MODEL_NAME);
        assert_eq!(config.all, vec![DEFAULT_MODEL_NAME.to_string()]);
        std::env::remove_var("METERIC_SUPPORTED_AI_MODELS");
        clear_provider_env_after_model_tests();
    }

    #[test]
    fn load_model_config_filters_empty_entries_from_list() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_ollama_provider_for_model_tests();
        std::env::remove_var("METEORIC_SUPPORTED_AI_MODELS");
        std::env::set_var("METERIC_SUPPORTED_AI_MODELS", "gemma4:e2b,,gemma4:e4b");
        let config = load_model_config();
        assert_eq!(
            config.all,
            vec!["gemma4:e2b".to_string(), "gemma4:e4b".to_string()]
        );
        std::env::remove_var("METERIC_SUPPORTED_AI_MODELS");
        clear_provider_env_after_model_tests();
    }

    #[test]
    fn load_model_config_falls_back_when_all_entries_are_empty_commas() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_ollama_provider_for_model_tests();
        std::env::remove_var("METEORIC_SUPPORTED_AI_MODELS");
        // All entries filter to empty strings, leaving an empty list.
        // The active model must still fall back to DEFAULT_MODEL_NAME.
        std::env::set_var("METERIC_SUPPORTED_AI_MODELS", ",");
        let config = load_model_config();
        assert_eq!(config.active, DEFAULT_MODEL_NAME);
        assert_eq!(config.all, Vec::<String>::new());
        std::env::remove_var("METERIC_SUPPORTED_AI_MODELS");
        clear_provider_env_after_model_tests();
    }

    // ── sampling options test ────────────────────────────────────────────────

    #[tokio::test]
    async fn sends_sampling_options_in_request() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"options":{"temperature":1.0,"top_p":0.95,"top_k":64}}"#.to_string(),
            ))
            .with_body(chat_line("", true))
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (_, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
    }

    #[test]
    fn load_system_prompt_returns_default_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("METEORIC_SYSTEM_PROMPT");
        std::env::remove_var("METERIC_SYSTEM_PROMPT");

        let prompt = load_system_prompt();
        assert_eq!(prompt, DEFAULT_SYSTEM_PROMPT);
    }

    #[test]
    fn load_system_prompt_reads_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("METEORIC_SYSTEM_PROMPT");
        std::env::set_var("METERIC_SYSTEM_PROMPT", "Custom prompt");

        let prompt = load_system_prompt();
        assert_eq!(prompt, "Custom prompt");

        std::env::remove_var("METERIC_SYSTEM_PROMPT");
    }

    #[test]
    fn load_system_prompt_ignores_empty_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("METEORIC_SYSTEM_PROMPT");
        std::env::set_var("METERIC_SYSTEM_PROMPT", "   ");

        let prompt = load_system_prompt();
        assert_eq!(prompt, DEFAULT_SYSTEM_PROMPT);

        std::env::remove_var("METERIC_SYSTEM_PROMPT");
    }

    #[test]
    fn load_provider_config_auto_uses_openrouter_base_for_sk_or_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("METEORIC_OPENAI_BASE_URL");
        std::env::remove_var("METERIC_OPENAI_BASE_URL");
        std::env::set_var("OPENAI_API_KEY", "sk-or-v1-test-key");

        let cfg = load_provider_config();
        assert_eq!(cfg.openai_base_url, DEFAULT_OPENROUTER_API_BASE_URL);

        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn load_provider_config_prefers_explicit_openai_base_url_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("OPENAI_API_KEY", "sk-or-v1-test-key");
        std::env::set_var("METEORIC_OPENAI_BASE_URL", "https://example.com/v1");

        let cfg = load_provider_config();
        assert_eq!(cfg.openai_base_url, "https://example.com/v1");

        std::env::remove_var("METEORIC_OPENAI_BASE_URL");
        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn load_provider_config_reads_gemini_key_and_base() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("METEORIC_AI_PROVIDER", "gemini");
        std::env::set_var("GEMINI_API_KEY", "test-gemini-key");
        std::env::set_var("METEORIC_GEMINI_BASE_URL", "https://example.com/gemini");

        let cfg = load_provider_config();
        assert_eq!(cfg.active, AiProvider::Gemini);
        assert_eq!(cfg.gemini_api_key.as_deref(), Some("test-gemini-key"));
        assert_eq!(cfg.gemini_base_url, "https://example.com/gemini");

        std::env::remove_var("METEORIC_AI_PROVIDER");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("METEORIC_GEMINI_BASE_URL");
    }

    #[test]
    fn load_model_config_uses_gemini_default_model() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("METEORIC_AI_PROVIDER", "gemini");
        std::env::remove_var("METEORIC_GEMINI_MODEL");
        std::env::remove_var("METERIC_GEMINI_MODEL");

        let cfg = load_model_config();
        assert_eq!(cfg.active, DEFAULT_GEMINI_MODEL);
        assert_eq!(cfg.all, vec![DEFAULT_GEMINI_MODEL.to_string()]);

        std::env::remove_var("METEORIC_AI_PROVIDER");
    }

    #[test]
    fn load_provider_config_reads_perplexity_key_and_base() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("METEORIC_AI_PROVIDER", "perplexity");
        std::env::set_var("PERPLEXITY_API_KEY", "pplx-test-key");
        std::env::set_var("METEORIC_PERPLEXITY_BASE_URL", "https://example.com/pplx");

        let cfg = load_provider_config();
        assert_eq!(cfg.active, AiProvider::Perplexity);
        assert_eq!(cfg.perplexity_api_key.as_deref(), Some("pplx-test-key"));
        assert_eq!(cfg.perplexity_base_url, "https://example.com/pplx");

        std::env::remove_var("METEORIC_AI_PROVIDER");
        std::env::remove_var("PERPLEXITY_API_KEY");
        std::env::remove_var("METEORIC_PERPLEXITY_BASE_URL");
    }

    #[test]
    fn load_model_config_uses_perplexity_default_model() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("METEORIC_AI_PROVIDER", "perplexity");
        std::env::remove_var("METEORIC_PERPLEXITY_MODEL");
        std::env::remove_var("METERIC_PERPLEXITY_MODEL");

        let cfg = load_model_config();
        assert_eq!(cfg.active, DEFAULT_PERPLEXITY_MODEL);
        assert_eq!(cfg.all, vec![DEFAULT_PERPLEXITY_MODEL.to_string()]);

        std::env::remove_var("METEORIC_AI_PROVIDER");
    }

    #[test]
    fn conversation_history_new_starts_at_epoch_zero() {
        let h = ConversationHistory::new();
        assert_eq!(h.epoch.load(Ordering::SeqCst), 0);
        assert!(h.messages.lock().unwrap().is_empty());
    }

    #[test]
    fn conversation_history_epoch_increments_on_clear() {
        let h = ConversationHistory::new();
        h.messages.lock().unwrap().push(ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
            images: None,
        });

        h.epoch.fetch_add(1, Ordering::SeqCst);
        h.messages.lock().unwrap().clear();

        assert_eq!(h.epoch.load(Ordering::SeqCst), 1);
        assert!(h.messages.lock().unwrap().is_empty());
    }

    // ─── OllamaError classification ───────────────────────────────────────────

    #[test]
    fn classify_http_404_returns_model_not_found() {
        let err = classify_http_error(404);
        assert_eq!(err.kind, OllamaErrorKind::ModelNotFound);
        assert!(err.message.contains("gemma4:e2b"));
    }

    #[test]
    fn classify_http_500_returns_other_with_status() {
        let err = classify_http_error(500);
        assert_eq!(err.kind, OllamaErrorKind::Other);
        assert!(err.message.contains("500"));
    }

    #[test]
    fn classify_http_401_returns_other_with_status() {
        let err = classify_http_error(401);
        assert_eq!(err.kind, OllamaErrorKind::Other);
        assert!(err.message.contains("401"));
    }

    #[tokio::test]
    async fn connection_refused_emits_not_running_error() {
        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            "http://127.0.0.1:1/api/chat",
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        let chunks = chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(
            matches!(&chunks[0], StreamChunk::Error(e) if e.kind == OllamaErrorKind::NotRunning)
        );
    }

    #[tokio::test]
    async fn http_404_emits_model_not_found_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_status(404)
            .with_body("")
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(
            matches!(&chunks[0], StreamChunk::Error(e) if e.kind == OllamaErrorKind::ModelNotFound)
        );
    }

    #[test]
    fn thinking_token_serializes_correctly() {
        let chunk = StreamChunk::ThinkingToken("reasoning step".to_string());
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["type"], "ThinkingToken");
        assert_eq!(json["data"], "reasoning step");
    }

    #[test]
    fn ollama_chat_request_sends_think_false_explicitly() {
        let req = OllamaChatRequest {
            model: "test".to_string(),
            messages: vec![],
            stream: true,
            think: false,
            options: OllamaOptions {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 64,
            },
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["think"], false);
    }

    #[test]
    fn ollama_chat_request_includes_think_when_true() {
        let req = OllamaChatRequest {
            model: "test".to_string(),
            messages: vec![],
            stream: true,
            think: true,
            options: OllamaOptions {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 64,
            },
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["think"], true);
    }

    #[test]
    fn ollama_response_message_deserializes_thinking_field() {
        let json = r#"{"content":"hello","thinking":"let me think"}"#;
        let msg: OllamaChatResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.unwrap(), "hello");
        assert_eq!(msg.thinking.unwrap(), "let me think");
    }

    #[test]
    fn ollama_response_message_thinking_absent() {
        let json = r#"{"content":"hello"}"#;
        let msg: OllamaChatResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.unwrap(), "hello");
        assert!(msg.thinking.is_none());
    }

    #[tokio::test]
    async fn http_500_emits_other_error_with_status() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_status(500)
            .with_body("Internal Server Error")
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            false,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(
            matches!(&chunks[0], StreamChunk::Error(e) if e.kind == OllamaErrorKind::Other && e.message.contains("500"))
        );
    }

    /// Helper: builds a `/api/chat` response line with both thinking and content fields.
    fn chat_line_with_thinking(thinking: &str, content: &str, done: bool) -> String {
        format!(
            "{{\"message\":{{\"role\":\"assistant\",\"content\":\"{}\",\"thinking\":\"{}\"}},\"done\":{}}}\n",
            content, thinking, done
        )
    }

    #[tokio::test]
    async fn stream_ollama_chat_emits_thinking_tokens() {
        let mut server = mockito::Server::new_async().await;
        let body = format!(
            "{}{}{}",
            chat_line_with_thinking("step 1", "", false),
            chat_line_with_thinking("", "Hello", false),
            chat_line_with_thinking("", "", true),
        );
        let mock = server
            .mock("POST", "/api/chat")
            .with_body(body)
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        let accumulated = stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            true,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();

        // ThinkingToken emitted for thinking field
        assert!(matches!(&chunks[0], StreamChunk::ThinkingToken(t) if t == "step 1"));
        // Token emitted for content field
        assert!(matches!(&chunks[1], StreamChunk::Token(t) if t == "Hello"));
        // Done emitted
        assert!(matches!(&chunks[2], StreamChunk::Done));

        // Accumulated return value contains only content, not thinking
        assert_eq!(accumulated, "Hello");
    }

    #[tokio::test]
    async fn stream_ollama_chat_sends_think_true_in_request() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"think":true}"#.to_string(),
            ))
            .with_body(chat_line("", true))
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (_, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            true,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn stream_ollama_chat_empty_thinking_not_emitted() {
        let mut server = mockito::Server::new_async().await;
        let body = format!(
            "{}{}",
            chat_line_with_thinking("", "Hello", false),
            chat_line_with_thinking("", "", true),
        );
        let mock = server
            .mock("POST", "/api/chat")
            .with_body(body)
            .create_async()
            .await;

        let client = reqwest::Client::new();
        let token = CancellationToken::new();
        let (chunks, callback) = collect_chunks();

        stream_ollama_chat(
            &format!("{}/api/chat", server.url()),
            "test-model",
            vec![],
            true,
            &client,
            token,
            callback,
        )
        .await;

        mock.assert_async().await;
        let chunks = chunks.lock().unwrap();

        // No ThinkingToken emitted for empty thinking field
        assert!(chunks
            .iter()
            .all(|c| !matches!(c, StreamChunk::ThinkingToken(_))));
        // Content token still emitted
        assert!(chunks
            .iter()
            .any(|c| matches!(c, StreamChunk::Token(t) if t == "Hello")));
    }
}
