//! Ollama LLM client for local inference

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::llm::{error::LLMError, Client, HasProvider};

/// Marker type for Ollama provider
pub struct Ollama;

/// Configuration for Ollama client
#[derive(Clone, Debug)]
pub struct OllamaConfig {
    /// Ollama server URL (default: http://localhost:11434)
    pub host: String,
    /// Default model to use (default: phi4)
    pub default_model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            default_model: "phi4".to_string(),
        }
    }
}

/// Response from Ollama's generate endpoint
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub response: String,
    pub done: bool,
    #[serde(default)]
    pub done_reason: String,
    #[serde(default)]
    pub context: Vec<u32>,
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub load_duration: u64,
    #[serde(default)]
    pub prompt_eval_count: u32,
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_count: u32,
    #[serde(default)]
    pub eval_duration: u64,
}

/// Request structure for Ollama chat completions
#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// A message in Ollama's chat format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

impl OllamaMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Options for Ollama generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Response from Ollama's chat endpoint
#[derive(Debug, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub message: OllamaMessage,
    pub done: bool,
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub load_duration: u64,
    #[serde(default)]
    pub prompt_eval_count: u32,
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_count: u32,
    #[serde(default)]
    pub eval_duration: u64,
}

use std::future::IntoFuture;
use std::pin::Pin;

/// Builder for Ollama chat completions
pub struct OllamaCompletionBuilder<'a, S> {
    pub(crate) client: &'a Client<S>,
    pub(crate) model: Option<String>,
    pub(crate) messages: Vec<OllamaMessage>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) stop_sequences: Option<Vec<String>>,
    pub(crate) json_mode: bool,
}

impl<'a, S> OllamaCompletionBuilder<'a, S> {
    pub fn new(client: &'a Client<S>) -> Self {
        Self {
            client,
            model: None,
            messages: Vec::new(),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stop_sequences: None,
            json_mode: false,
        }
    }
}

impl<'a, S> OllamaCompletionBuilder<'a, S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Set the model for this completion (overrides default)
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add a system message
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(OllamaMessage::system(content));
        self
    }

    /// Add a user message
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(OllamaMessage::user(content));
        self
    }

    /// Add an assistant message
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(OllamaMessage::assistant(content));
        self
    }

    /// Seed the conversation with existing messages
    pub fn messages(mut self, messages: Vec<OllamaMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Set the sampling temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the top-p sampling value
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the top-k sampling value
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the maximum number of tokens to generate
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Enable or disable JSON mode
    pub fn json_mode(mut self, enabled: bool) -> Self {
        self.json_mode = enabled;
        self
    }

    pub(crate) async fn execute(self) -> Result<String, LLMError> {
        let config = self.client.ollama_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("Ollama not configured".to_string())
        })?;

        let model_to_use = self.model.unwrap_or_else(|| config.default_model.clone());

        // Implicit validation
        let mut cache = self.client.model_cache.ollama.read().unwrap().clone();
        if cache.is_none() {
            if let Ok(models) = self.client.ollama_list_models().await {
                let names: Vec<String> = models.into_iter().map(|m| m.name).collect();
                *self.client.model_cache.ollama.write().unwrap() = Some(names.clone());
                cache = Some(names);
            }
        }

        if let Some(valid_models) = cache {
            if !valid_models.contains(&model_to_use) {
                return Err(LLMError::InvalidModel(format!(
                    "Model '{}' not found in Ollama available models",
                    model_to_use
                )));
            }
        }

        let options = Some(OllamaOptions {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            num_predict: self.max_tokens,
            stop: self.stop_sequences,
        });

        let response = self
            .client
            .call_ollama_chat(model_to_use, self.messages, options, self.json_mode)
            .await?;
        Ok(response.message.content)
    }
}

impl<'a, S> IntoFuture for OllamaCompletionBuilder<'a, S>
where
    S: HasProvider<Ollama> + Send + Sync + Clone + 'static,
{
    type Output = Result<String, LLMError>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Model information from Ollama
#[derive(Debug, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub modified_at: DateTime<Utc>,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
struct OllamaModelsResponse {
    pub models: Vec<OllamaModel>,
}

impl<S> Client<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// List available models from Ollama
    pub async fn ollama_list_models(&self) -> Result<Vec<OllamaModel>, LLMError> {
        let config = self.ollama_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("Ollama not configured".to_string())
        })?;

        let response = self
            .client
            .get(format!("{}/api/tags", config.host))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LLMError::OllamaError(format!(
                "Failed to list models: HTTP {}",
                response.status()
            )));
        }

        let res: OllamaModelsResponse = response.json().await?;
        Ok(res.models)
    }

    /// Call Ollama's generate endpoint (legacy)
    pub async fn call_ollama(
        &self,
        model: impl Into<String>,
        prompt: impl Into<String>,
        stream: bool,
    ) -> Result<OllamaResponse, LLMError> {
        let config = self.ollama_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("Ollama not configured".to_string())
        })?;

        let payload = json!({
            "model": model.into(),
            "prompt": prompt.into(),
            "stream": stream
        });

        let response = self
            .client
            .post(format!("{}/api/generate", config.host))
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::OllamaError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let ollama_response: OllamaResponse = response.json().await?;
        Ok(ollama_response)
    }

    pub async fn call_ollama_chat(
        &self,
        model: impl Into<String>,
        messages: Vec<OllamaMessage>,
        options: Option<OllamaOptions>,
        json_mode: bool,
    ) -> Result<OllamaChatResponse, LLMError> {
        let config = self.ollama_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("Ollama not configured".to_string())
        })?;

        let format = if json_mode {
            Some("json".to_string())
        } else {
            None
        };

        let request = OllamaChatRequest {
            model: model.into(),
            messages,
            stream: false,
            options,
            format,
        };

        let response = self
            .client
            .post(format!("{}/api/chat", config.host))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::OllamaError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let chat_response: OllamaChatResponse = response.json().await?;
        Ok(chat_response)
    }

    pub fn ollama_complete(&self) -> OllamaCompletionBuilder<'_, S> 
    where S: HasProvider<Ollama>
    {
        self.ollama_complete_internal()
    }

    pub(crate) fn ollama_complete_internal(&self) -> OllamaCompletionBuilder<'_, S> {
        OllamaCompletionBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_message_constructors() {
        let system = OllamaMessage::system("You are helpful");
        assert_eq!(system.role, "system");

        let user = OllamaMessage::user("Hello");
        assert_eq!(user.role, "user");

        let assistant = OllamaMessage::assistant("Hi there!");
        assert_eq!(assistant.role, "assistant");
    }
}
