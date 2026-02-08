//! LLM client module for Orichalcum
//!
//! This module provides a unified client for multiple LLM providers using a typestate pattern.
//! Each provider is enabled/disabled at compile time, ensuring type-safe API usage.

pub mod deepseek;
pub mod error;
pub mod gemini;
pub mod ollama;

use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

pub use deepseek::{DeepSeek, DeepSeekConfig, DeepSeekMessage, DeepSeekResponse};
pub use error::LLMError;
pub use gemini::{Gemini, GeminiConfig, GeminiContent, GeminiGenerationConfig, GeminiResponse};
pub use ollama::{Ollama, OllamaConfig};

/// LLM client wrapper around reqwest::Client
/// Uses typestate pattern to track which providers are configured
#[derive(Clone)]
pub struct Client<S> {
    /// The underlying HTTP client
    pub(crate) client: reqwest::Client,
    /// Marker for the current state (which providers are enabled)
    pub(crate) state: PhantomData<S>,
    /// Ollama configuration
    pub(crate) ollama_config: Option<OllamaConfig>,
    /// DeepSeek configuration
    pub(crate) deepseek_config: Option<DeepSeekConfig>,
    /// Gemini configuration
    pub(crate) gemini_config: Option<GeminiConfig>,
    /// Cache for available models to support implicit validation
    pub(crate) model_cache: ModelCache,
}

/// Thread-safe cache for provider model lists
#[derive(Clone, Default)]
pub struct ModelCache {
    pub(crate) ollama: Arc<RwLock<Option<Vec<String>>>>,
    pub(crate) deepseek: Arc<RwLock<Option<Vec<String>>>>,
    pub(crate) gemini: Arc<RwLock<Option<Vec<String>>>>,
}

// ============================================================================
// Type States
// ============================================================================

/// Marker indicating a provider is enabled
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Enabled;

/// Marker indicating a provider is disabled
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Disabled;

/// Provider state container
/// Each type parameter tracks whether a specific provider is configured
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Providers<OllamaState, DeepSeekState, GeminiState> {
    _ollama: PhantomData<OllamaState>,
    _deepseek: PhantomData<DeepSeekState>,
    _gemini: PhantomData<GeminiState>,
}

// ============================================================================
// HasProvider trait implementations
// ============================================================================

/// Trait to check if a provider is available on this client
pub trait HasProvider<Provider> {}

/// Ollama is available when the first type param is Enabled
impl<D, G> HasProvider<Ollama> for Providers<Enabled, D, G> {}

/// DeepSeek is available when the second type param is Enabled
impl<O, G> HasProvider<DeepSeek> for Providers<O, Enabled, G> {}

/// Gemini is available when the third type param is Enabled
impl<O, D> HasProvider<Gemini> for Providers<O, D, Enabled> {}

// ============================================================================
// Client constructors and builders
// ============================================================================

impl Client<Providers<Disabled, Disabled, Disabled>> {
    /// Create a new LLM client with no providers configured
    pub fn new() -> Self {
        Client {
            client: reqwest::Client::new(),
            state: PhantomData,
            ollama_config: None,
            deepseek_config: None,
            gemini_config: None,
            model_cache: ModelCache::default(),
        }
    }
}

impl Default for Client<Providers<Disabled, Disabled, Disabled>> {
    fn default() -> Self {
        Self::new()
    }
}

// Builder methods that enable providers

impl<D, G> Client<Providers<Disabled, D, G>> {
    /// Enable Ollama provider with the default host (http://localhost:11434) and default model (phi4)
    pub fn with_ollama(self) -> Client<Providers<Enabled, D, G>> {
        self.with_ollama_at("http://localhost:11434")
    }

    /// Enable Ollama provider with a custom host URL
    pub fn with_ollama_at(self, host: impl Into<String>) -> Client<Providers<Enabled, D, G>> {
        Client {
            client: self.client,
            state: PhantomData,
            ollama_config: Some(OllamaConfig {
                host: host.into(),
                ..Default::default()
            }),
            deepseek_config: self.deepseek_config,
            gemini_config: self.gemini_config,
            model_cache: self.model_cache,
        }
    }
}

impl<O, G> Client<Providers<O, Disabled, G>> {
    /// Enable DeepSeek provider with API key and default base URL
    pub fn with_deepseek(self, api_key: impl Into<String>) -> Client<Providers<O, Enabled, G>> {
        self.with_deepseek_at(api_key, "https://api.deepseek.com")
    }

    /// Enable DeepSeek provider with API key and custom base URL
    pub fn with_deepseek_at(
        self,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Client<Providers<O, Enabled, G>> {
        Client {
            client: self.client,
            state: PhantomData,
            ollama_config: self.ollama_config,
            deepseek_config: Some(DeepSeekConfig {
                api_key: api_key.into(),
                base_url: base_url.into(),
                ..Default::default()
            }),
            gemini_config: self.gemini_config,
            model_cache: self.model_cache,
        }
    }
}

impl<O, D> Client<Providers<O, D, Disabled>> {
    /// Enable Gemini provider with API key and default base URL
    pub fn with_gemini(self, api_key: impl Into<String>) -> Client<Providers<O, D, Enabled>> {
        self.with_gemini_at(api_key, "https://generativelanguage.googleapis.com")
    }

    /// Enable Gemini provider with API key and custom base URL
    pub fn with_gemini_at(
        self,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Client<Providers<O, D, Enabled>> {
        Client {
            client: self.client,
            state: PhantomData,
            ollama_config: self.ollama_config,
            deepseek_config: self.deepseek_config,
            gemini_config: Some(GeminiConfig {
                api_key: api_key.into(),
                base_url: base_url.into(),
                ..Default::default()
            }),
            model_cache: self.model_cache,
        }
    }
}

// Edit methods for enabled providers

impl<D, G> Client<Providers<Enabled, D, G>> {
    /// Update the Ollama host URL
    pub fn edit_ollama_host(&mut self, host: impl Into<String>) {
        if let Some(ref mut config) = self.ollama_config {
            let new_host = host.into();
            assert!(!new_host.is_empty(), "Ollama host cannot be empty");
            config.host = new_host;
        }
    }

    /// Update the Ollama default model
    pub fn edit_ollama_default_model(&mut self, model: impl Into<String>) {
        if let Some(ref mut config) = self.ollama_config {
            config.default_model = model.into();
        }
    }
}

impl<O, G> Client<Providers<O, Enabled, G>> {
    /// Update the DeepSeek API key
    pub fn edit_deepseek_api_key(&mut self, api_key: impl Into<String>) {
        if let Some(ref mut config) = self.deepseek_config {
            config.api_key = api_key.into();
        }
    }

    /// Update the DeepSeek base URL
    pub fn edit_deepseek_base_url(&mut self, base_url: impl Into<String>) {
        if let Some(ref mut config) = self.deepseek_config {
            config.base_url = base_url.into();
        }
    }

    /// Update the DeepSeek default model
    pub fn edit_deepseek_default_model(&mut self, model: impl Into<String>) {
        if let Some(ref mut config) = self.deepseek_config {
            config.default_model = model.into();
        }
    }
}

impl<O, D> Client<Providers<O, D, Enabled>> {
    /// Update the Gemini API key
    pub fn edit_gemini_api_key(&mut self, api_key: impl Into<String>) {
        if let Some(ref mut config) = self.gemini_config {
            config.api_key = api_key.into();
        }
    }

    /// Update the Gemini base URL
    pub fn edit_gemini_base_url(&mut self, base_url: impl Into<String>) {
        if let Some(ref mut config) = self.gemini_config {
            config.base_url = base_url.into();
        }
    }

    /// Update the Gemini default model
    pub fn edit_gemini_default_model(&mut self, model: impl Into<String>) {
        if let Some(ref mut config) = self.gemini_config {
            config.default_model = model.into();
        }
    }
}

impl<S: Clone + Send + Sync + 'static> Client<S> {
    /// Internal dispatch method to call the first available provider.
    /// Used by semantic nodes where the provider typestate is erased.
    pub(crate) async fn dispatch_complete(&self, prompt: &str, model: Option<String>) -> Result<String, LLMError> {
        if self.deepseek_config.is_some() {
            return self.execute_deepseek(prompt, model).await;
        }

        if self.gemini_config.is_some() {
            return self.execute_gemini(prompt, model).await;
        }

        if self.ollama_config.is_some() {
            return self.execute_ollama(prompt, model).await;
        }

        Err(LLMError::ProviderNotConfigured("No LLM provider available".to_string()))
    }

    async fn execute_deepseek(&self, prompt: &str, model: Option<String>) -> Result<String, LLMError> {
        let mut builder = deepseek::DeepSeekCompletionBuilder::new(self).user(prompt).json_mode(true);
        if let Some(m) = model { builder = builder.model(m); }
        builder.execute().await
    }

    async fn execute_gemini(&self, prompt: &str, model: Option<String>) -> Result<String, LLMError> {
        let mut builder = gemini::GeminiCompletionBuilder::new(self).user(prompt).json_mode(true);
        if let Some(m) = model { builder = builder.model(m); }
        builder.execute().await
    }

    async fn execute_ollama(&self, prompt: &str, model: Option<String>) -> Result<String, LLMError> {
        let mut builder = ollama::OllamaCompletionBuilder::new(self).user(prompt).json_mode(true);
        if let Some(m) = model { builder = builder.model(m); }
        builder.execute().await
    }
}

// ============================================================================
// Deref to reqwest::Client for direct HTTP usage
// ============================================================================

impl<S: Clone + Send + Sync + 'static> std::ops::Deref for Client<S> {
    type Target = reqwest::Client;
    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl<S: Clone + Send + Sync + 'static> std::ops::DerefMut for Client<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = Client::new();
        assert!(client.ollama_config.is_none());
        assert!(client.deepseek_config.is_none());
        assert!(client.gemini_config.is_none());
    }

    #[test]
    fn test_with_ollama() {
        let client = Client::new().with_ollama();
        assert!(client.ollama_config.is_some());
        let config = client.ollama_config.unwrap();
        assert_eq!(config.host, "http://localhost:11434");
        assert_eq!(config.default_model, "phi4");

        let client_custom = Client::new().with_ollama_at("http://192.168.1.10:11434");
        assert_eq!(
            client_custom.ollama_config.unwrap().host,
            "http://192.168.1.10:11434"
        );
    }

    #[test]
    fn test_with_deepseek() {
        let client = Client::new().with_deepseek("test-key");
        assert!(client.deepseek_config.is_some());
        let config = client.deepseek_config.unwrap();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.deepseek.com");
        assert_eq!(config.default_model, "deepseek-reasoner");

        let client_custom = Client::new().with_deepseek_at("test-key", "https://custom.deepseek.com");
        let config_custom = client_custom.deepseek_config.unwrap();
        assert_eq!(config_custom.base_url, "https://custom.deepseek.com");
    }

    #[test]
    fn test_with_gemini() {
        let client = Client::new().with_gemini("test-key");
        assert!(client.gemini_config.is_some());
        let config = client.gemini_config.unwrap();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://generativelanguage.googleapis.com");
        assert_eq!(config.default_model, "gemini-3-flash-preview");
    }

    #[test]
    fn test_multiple_providers() {
        let client = Client::new()
            .with_ollama()
            .with_deepseek("deepseek-key")
            .with_gemini("gemini-key");

        assert!(client.ollama_config.is_some());
        assert!(client.deepseek_config.is_some());
        assert!(client.gemini_config.is_some());
    }
}
