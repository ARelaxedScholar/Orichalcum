//! DeepSeek LLM client
//!
//! DeepSeek uses an OpenAI-compatible API, making integration straightforward.

use serde::{Deserialize, Serialize};

use crate::llm::{error::LLMError, Client, HasProvider};

/// Marker type for DeepSeek provider
pub struct DeepSeek;

/// Configuration for DeepSeek client
#[derive(Clone, Debug)]
pub struct DeepSeekConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL (default: https://api.deepseek.com)
    pub base_url: String,
    /// Default model to use (default: deepseek-reasoner)
    pub default_model: String,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.deepseek.com".to_string(),
            default_model: "deepseek-reasoner".to_string(),
        }
    }
}

/// Request structure for DeepSeek chat completions
#[derive(Debug, Serialize)]
pub struct DeepSeekRequest {
    pub model: String,
    pub messages: Vec<DeepSeekMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<DeepSeekResponseFormat>,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct DeepSeekResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

/// A message in DeepSeek's chat format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekMessage {
    pub role: String,
    pub content: String,
}

impl DeepSeekMessage {
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

/// Response from DeepSeek chat completions
#[derive(Debug, Deserialize)]
pub struct DeepSeekResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<DeepSeekChoice>,
    pub usage: DeepSeekUsage,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekChoice {
    pub index: u32,
    pub message: DeepSeekMessage,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

use std::future::IntoFuture;
use std::pin::Pin;

/// Builder for DeepSeek chat completions
pub struct DeepSeekCompletionBuilder<'a, S> {
    pub(crate) client: &'a Client<S>,
    pub(crate) model: Option<String>,
    pub(crate) messages: Vec<DeepSeekMessage>,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) stop_sequences: Option<Vec<String>>,
    pub(crate) json_mode: bool,
}

impl<'a, S> DeepSeekCompletionBuilder<'a, S> {
    pub fn new(client: &'a Client<S>) -> Self {
        Self {
            client,
            model: None,
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop_sequences: None,
            json_mode: false,
        }
    }
}

impl<'a, S> DeepSeekCompletionBuilder<'a, S>
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
        self.messages.push(DeepSeekMessage::system(content));
        self
    }

    /// Add a user message
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(DeepSeekMessage::user(content));
        self
    }

    /// Add an assistant message
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(DeepSeekMessage::assistant(content));
        self
    }

    /// Seed the conversation with existing messages
    pub fn messages(mut self, messages: Vec<DeepSeekMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Set the sampling temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens to generate
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-p sampling value
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
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
        let config = self.client.deepseek_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("DeepSeek not configured".to_string())
        })?;

        let model_to_use = self.model.unwrap_or_else(|| config.default_model.clone());

        // Implicit validation
        let mut cache = self.client.model_cache.deepseek.read().unwrap().clone();
        if cache.is_none() {
            if let Ok(models) = self.client.deepseek_list_models().await {
                let names: Vec<String> = models.into_iter().map(|m| m.id).collect();
                *self.client.model_cache.deepseek.write().unwrap() = Some(names.clone());
                cache = Some(names);
            }
        }

        if let Some(valid_models) = cache {
            if !valid_models.contains(&model_to_use) {
                return Err(LLMError::InvalidModel(format!(
                    "Model '{}' not found in DeepSeek available models",
                    model_to_use
                )));
            }
        }

        let response = self
            .client
            .call_deepseek(
                model_to_use,
                self.messages,
                self.temperature,
                self.max_tokens,
                self.top_p,
                self.stop_sequences,
                self.json_mode,
            )
            .await?;

        response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| LLMError::InvalidResponse("No choices in response".to_string()))
    }
}

impl<'a, S> IntoFuture for DeepSeekCompletionBuilder<'a, S>
where
    S: HasProvider<DeepSeek> + Send + Sync + Clone + 'static,
{
    type Output = Result<String, LLMError>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Model information from DeepSeek
#[derive(Debug, Deserialize)]
pub struct DeepSeekModel {
    pub id: String,
}

#[derive(Debug, Deserialize)]
struct DeepSeekModelsResponse {
    pub data: Vec<DeepSeekModel>,
}

impl<S> Client<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// List available models from DeepSeek
    pub async fn deepseek_list_models(&self) -> Result<Vec<DeepSeekModel>, LLMError> {
        let config = self.deepseek_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("DeepSeek not configured".to_string())
        })?;

        let response = self
            .client
            .get(format!("{}/v1/models", config.base_url))
            .header("Authorization", format!("Bearer {}", config.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LLMError::DeepSeekError(format!(
                "Failed to list models: HTTP {}",
                response.status()
            )));
        }

        let res: DeepSeekModelsResponse = response.json().await?;
        Ok(res.data)
    }

    /// Call DeepSeek's chat completion API
    pub async fn call_deepseek(
        &self,
        model: impl Into<String>,
        messages: Vec<DeepSeekMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        top_p: Option<f32>,
        stop: Option<Vec<String>>,
        json_mode: bool,
    ) -> Result<DeepSeekResponse, LLMError> {
        let config = self.deepseek_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("DeepSeek not configured".to_string())
        })?;

        let response_format = if json_mode {
            Some(DeepSeekResponseFormat {
                format_type: "json_object".to_string(),
            })
        } else {
            None
        };

        let request = DeepSeekRequest {
            model: model.into(),
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            response_format,
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", config.base_url))
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::DeepSeekError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let deepseek_response: DeepSeekResponse = response.json().await?;
        Ok(deepseek_response)
    }

    pub fn deepseek_complete(&self) -> DeepSeekCompletionBuilder<'_, S> 
    where S: HasProvider<DeepSeek>
    {
        self.deepseek_complete_internal()
    }

    pub(crate) fn deepseek_complete_internal(&self) -> DeepSeekCompletionBuilder<'_, S> {
        DeepSeekCompletionBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_message_constructors() {
        let system = DeepSeekMessage::system("You are helpful");
        assert_eq!(system.role, "system");

        let user = DeepSeekMessage::user("Hello");
        assert_eq!(user.role, "user");

        let assistant = DeepSeekMessage::assistant("Hi there!");
        assert_eq!(assistant.role, "assistant");
    }

    #[test]
    fn test_deepseek_request_serialization() {
        let request = DeepSeekRequest {
            model: "deepseek-chat".to_string(),
            messages: vec![DeepSeekMessage::user("Test")],
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            stop: None,
            response_format: None,
            stream: false,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("deepseek-chat"));
        assert!(json.contains("temperature"));
        assert!(!json.contains("max_tokens"));
    }
}
