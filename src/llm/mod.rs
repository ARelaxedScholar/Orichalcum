//! LLM client module for Orichalcum
//!
//! This module provides a unified client for multiple LLM providers using a typestate pattern.
//! Each provider is enabled/disabled at compile time, ensuring type-safe API usage.
//!
//! # Supported Providers
//! - **Ollama** - Local LLM inference
//! - **DeepSeek** - OpenAI-compatible cloud API
//! - **Gemini** - Google's Gemini API
//!
//! # Example
//! ```ignore
//! use orichalcum::llm::{Client, deepseek::DeepSeekMessage};
//!
//! // Create a client with DeepSeek enabled
//! let client = Client::new()
//!     .with_deepseek("your-api-key", None);
//!
//! // Use DeepSeek
//! let response = client.deepseek_complete(
//!     "deepseek-chat",
//!     "You are a helpful assistant.",
//!     "Hello!",
//!     Some(0.7),
//!     None
//! ).await?;
//! ```

pub mod deepseek;
pub mod error;
pub mod gemini;
pub mod ollama;

use std::marker::PhantomData;

pub use deepseek::{DeepSeek, DeepSeekConfig, DeepSeekMessage, DeepSeekResponse};
pub use gemini::{Gemini, GeminiConfig, GeminiContent, GeminiGenerationConfig, GeminiResponse};
pub use ollama::Ollama;

/// LLM client wrapper around reqwest::Client
/// Uses typestate pattern to track which providers are configured
#[derive(Clone)]
pub struct Client<S> {
    /// The underlying HTTP client
    pub(crate) client: reqwest::Client,
    /// Marker for the current state (which providers are enabled)
    state: PhantomData<S>,
    /// Ollama configuration (host URL)
    pub(crate) ollama_host: Option<String>,
    /// DeepSeek configuration
    pub(crate) deepseek_config: Option<DeepSeekConfig>,
    /// Gemini configuration
    pub(crate) gemini_config: Option<GeminiConfig>,
}

// ============================================================================
// Type States
// ============================================================================

/// Marker indicating a provider is enabled
pub struct Enabled;

/// Marker indicating a provider is disabled
pub struct Disabled;

/// Provider state container
/// Each type parameter tracks whether a specific provider is configured
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
            ollama_host: None,
            deepseek_config: None,
            gemini_config: None,
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
    /// Enable Ollama provider with the given host URL
    ///
    /// # Arguments
    /// * `host` - Ollama server URL (e.g., "http://localhost:11434")
    pub fn with_ollama(self, host: impl Into<String>) -> Client<Providers<Enabled, D, G>> {
        Client {
            client: self.client,
            state: PhantomData,
            ollama_host: Some(host.into()),
            deepseek_config: self.deepseek_config,
            gemini_config: self.gemini_config,
        }
    }
}

impl<O, G> Client<Providers<O, Disabled, G>> {
    /// Enable DeepSeek provider with API key and optional custom base URL
    ///
    /// # Arguments
    /// * `api_key` - DeepSeek API key
    /// * `base_url` - Optional custom base URL (defaults to https://api.deepseek.com)
    pub fn with_deepseek(
        self,
        api_key: impl Into<String>,
        base_url: Option<String>,
    ) -> Client<Providers<O, Enabled, G>> {
        Client {
            client: self.client,
            state: PhantomData,
            ollama_host: self.ollama_host,
            deepseek_config: Some(DeepSeekConfig {
                api_key: api_key.into(),
                base_url: base_url.unwrap_or_else(|| "https://api.deepseek.com".to_string()),
            }),
            gemini_config: self.gemini_config,
        }
    }
}

impl<O, D> Client<Providers<O, D, Disabled>> {
    /// Enable Gemini provider with API key and optional custom base URL
    ///
    /// # Arguments
    /// * `api_key` - Google Gemini API key
    /// * `base_url` - Optional custom base URL
    pub fn with_gemini(
        self,
        api_key: impl Into<String>,
        base_url: Option<String>,
    ) -> Client<Providers<O, D, Enabled>> {
        Client {
            client: self.client,
            state: PhantomData,
            ollama_host: self.ollama_host,
            deepseek_config: self.deepseek_config,
            gemini_config: Some(GeminiConfig {
                api_key: api_key.into(),
                base_url: base_url
                    .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string()),
            }),
        }
    }
}

// Edit methods for enabled providers

impl<D, G> Client<Providers<Enabled, D, G>> {
    /// Update the Ollama host URL
    pub fn edit_ollama_host(&mut self, host: impl Into<String>) {
        let new_host = host.into();
        assert!(!new_host.is_empty(), "Ollama host cannot be empty");
        self.ollama_host = Some(new_host);
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
}

impl<O, D> Client<Providers<O, D, Enabled>> {
    /// Update the Gemini API key
    pub fn edit_gemini_api_key(&mut self, api_key: impl Into<String>) {
        if let Some(ref mut config) = self.gemini_config {
            config.api_key = api_key.into();
        }
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
        assert!(client.ollama_host.is_none());
        assert!(client.deepseek_config.is_none());
        assert!(client.gemini_config.is_none());
    }

    #[test]
    fn test_with_ollama() {
        let client = Client::new().with_ollama("http://localhost:11434");
        assert_eq!(
            client.ollama_host,
            Some("http://localhost:11434".to_string())
        );
    }

    #[test]
    fn test_with_deepseek() {
        let client = Client::new().with_deepseek("test-key", None);
        assert!(client.deepseek_config.is_some());
        let config = client.deepseek_config.unwrap();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.deepseek.com");
    }

    #[test]
    fn test_with_gemini() {
        let client = Client::new().with_gemini("test-key", None);
        assert!(client.gemini_config.is_some());
        let config = client.gemini_config.unwrap();
        assert_eq!(config.api_key, "test-key");
    }

    #[test]
    fn test_multiple_providers() {
        let client = Client::new()
            .with_ollama("http://localhost:11434")
            .with_deepseek("deepseek-key", None)
            .with_gemini("gemini-key", None);

        assert!(client.ollama_host.is_some());
        assert!(client.deepseek_config.is_some());
        assert!(client.gemini_config.is_some());
    }
}
