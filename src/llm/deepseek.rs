//! DeepSeek LLM client
//!
//! DeepSeek uses an OpenAI-compatible API, making integration straightforward.

use serde::{Deserialize, Serialize};
// use serde_json::json;

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
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.deepseek.com".to_string(),
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
    pub stream: bool,
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
    client: &'a Client<S>,
    model: String,
    system_prompt: String,
    user_prompt: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl<'a, S> DeepSeekCompletionBuilder<'a, S>
where
    S: HasProvider<DeepSeek> + Send + Sync + 'static,
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
            max_tokens: None,
        }
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
}

impl<'a, S> IntoFuture for DeepSeekCompletionBuilder<'a, S>
where
    S: HasProvider<DeepSeek> + Send + Sync + Clone + 'static,
{
    type Output = Result<String, LLMError>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let messages = vec![
                DeepSeekMessage::system(self.system_prompt),
                DeepSeekMessage::user(self.user_prompt),
            ];

            let response = self
                .client
                .call_deepseek(self.model, messages, self.temperature, self.max_tokens)
                .await?;

            response
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| LLMError::InvalidResponse("No choices in response".to_string()))
        })
    }
}

impl<S> Client<S>
where
    S: HasProvider<DeepSeek> + Clone + Send + Sync + 'static,
{
    /// Call DeepSeek's chat completion API
    ///
    /// # Arguments
    /// * `model` - Model to use (e.g., "deepseek-chat", "deepseek-coder")
    /// * `messages` - Conversation messages
    /// * `temperature` - Sampling temperature (0.0 - 2.0)
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Example
    /// ```ignore
    /// let client = Client::new().with_deepseek("your-api-key");
    /// let messages = vec![
    ///     DeepSeekMessage::system("You are a helpful assistant."),
    ///     DeepSeekMessage::user("Hello!"),
    /// ];
    /// let response = client.call_deepseek("deepseek-chat", messages, Some(0.7), None).await?;
    /// ```
    pub async fn call_deepseek(
        &self,
        model: impl Into<String>,
        messages: Vec<DeepSeekMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<DeepSeekResponse, LLMError> {
        let config = self.deepseek_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("DeepSeek not configured".to_string())
        })?;

        let request = DeepSeekRequest {
            model: model.into(),
            messages,
            temperature,
            max_tokens,
            top_p: None,
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

    /// Convenience method for simple single-turn completions using a builder pattern
    ///
    /// # Example
    /// ```ignore
    /// let result = client.deepseek_complete("deepseek-chat", "You are a helpful assistant.", "Hello!")
    ///     .temperature(0.7)
    ///     .await?;
    /// ```
    pub fn deepseek_complete(
        &self,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        user_prompt: impl Into<String>,
    ) -> DeepSeekCompletionBuilder<'_, S> {
        DeepSeekCompletionBuilder::new(self, model, system_prompt, user_prompt)
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
            stream: false,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("deepseek-chat"));
        assert!(json.contains("temperature"));
        // max_tokens should be skipped since it's None
        assert!(!json.contains("max_tokens"));
    }
}
