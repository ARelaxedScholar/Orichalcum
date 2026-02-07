//! Ollama LLM client for local inference

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::llm::{error::LLMError, Client, HasProvider};

/// Marker type for Ollama provider
pub struct Ollama;

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
    client: &'a Client<S>,
    model: String,
    system_prompt: String,
    user_prompt: String,
    temperature: Option<f32>,
}

impl<'a, S> OllamaCompletionBuilder<'a, S>
where
    S: HasProvider<Ollama> + Send + Sync + 'static,
{
    pub fn new(
        client: &'a Client<S>,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        user_prompt: impl Into<String>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            system_prompt: system_prompt.into(),
            user_prompt: user_prompt.into(),
            temperature: None,
        }
    }

    /// Set the sampling temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

impl<'a, S> IntoFuture for OllamaCompletionBuilder<'a, S>
where
    S: HasProvider<Ollama> + Send + Sync + Clone + 'static,
{
    type Output = Result<String, LLMError>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let messages = vec![
                OllamaMessage::system(self.system_prompt),
                OllamaMessage::user(self.user_prompt),
            ];

            let options = self.temperature.map(|t| OllamaOptions {
                temperature: Some(t),
                top_p: None,
                top_k: None,
                num_predict: None,
            });

            let response = self
                .client
                .call_ollama_chat(self.model, messages, options)
                .await?;
            Ok(response.message.content)
        })
    }
}

impl<S> Client<S>
where
    S: HasProvider<Ollama> + Clone + Send + Sync + 'static,
{
    /// Call Ollama's generate endpoint (legacy)
    pub async fn call_ollama(
        &self,
        model: impl Into<String>,
        prompt: impl Into<String>,
        stream: bool,
    ) -> Result<OllamaResponse, LLMError> {
        let ollama_host: &str = self
            .ollama_host
            .as_ref()
            .ok_or_else(|| LLMError::ProviderNotConfigured("Ollama not configured".to_string()))?;

        let payload = json!({
            "model": model.into(),
            "prompt": prompt.into(),
            "stream": stream
        });

        let response = self
            .client
            .post(format!("{}/api/generate", ollama_host))
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

    /// Call Ollama's chat endpoint (recommended)
    pub async fn call_ollama_chat(
        &self,
        model: impl Into<String>,
        messages: Vec<OllamaMessage>,
        options: Option<OllamaOptions>,
    ) -> Result<OllamaChatResponse, LLMError> {
        let ollama_host: &str = self
            .ollama_host
            .as_ref()
            .ok_or_else(|| LLMError::ProviderNotConfigured("Ollama not configured".to_string()))?;

        let request = OllamaChatRequest {
            model: model.into(),
            messages,
            stream: false,
            options,
        };

        let response = self
            .client
            .post(format!("{}/api/chat", ollama_host))
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

    /// Convenience method for simple completions using a builder pattern
    ///
    /// # Example
    /// ```ignore
    /// let result = client.ollama_complete("llama3", "You are a helpful assistant.", "Hello!")
    ///     .temperature(0.7)
    ///     .await?;
    /// ```
    pub fn ollama_complete(
        &self,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        user_prompt: impl Into<String>,
    ) -> OllamaCompletionBuilder<'_, S> {
        OllamaCompletionBuilder::new(self, model, system_prompt, user_prompt)
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
