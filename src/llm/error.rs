use thiserror::Error;

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("HTTP request error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Ollama error: {0}")]
    OllamaError(String),

    #[error("DeepSeek error: {0}")]
    DeepSeekError(String),

    #[error("Gemini error: {0}")]
    GeminiError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Provider not configured: {0}")]
    ProviderNotConfigured(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}
