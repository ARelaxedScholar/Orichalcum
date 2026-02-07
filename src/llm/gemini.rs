//! Google Gemini LLM client
//!
//! Supports the Gemini API for text generation.

use serde::{Deserialize, Serialize};

use crate::llm::{error::LLMError, Client, HasProvider};

/// Marker type for Gemini provider
pub struct Gemini;

/// Configuration for Gemini client
#[derive(Clone, Debug)]
pub struct GeminiConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL (default: https://generativelanguage.googleapis.com)
    pub base_url: String,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
        }
    }
}

/// Request structure for Gemini generate content
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
}

/// Content structure for Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<GeminiPart>,
}

/// A part of content (text, image, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl GeminiContent {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Some("user".to_string()),
            parts: vec![GeminiPart {
                text: Some(text.into()),
            }],
        }
    }

    pub fn model(text: impl Into<String>) -> Self {
        Self {
            role: Some("model".to_string()),
            parts: vec![GeminiPart {
                text: Some(text.into()),
            }],
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: None, // System instructions don't have a role
            parts: vec![GeminiPart {
                text: Some(text.into()),
            }],
        }
    }
}

/// Generation configuration for Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

impl Default for GeminiGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(40),
            max_output_tokens: None,
            stop_sequences: None,
        }
    }
}

/// Response from Gemini generate content
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCandidate {
    pub content: GeminiContent,
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub safety_ratings: Vec<GeminiSafetyRating>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiSafetyRating {
    pub category: String,
    pub probability: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiUsageMetadata {
    pub prompt_token_count: u32,
    pub candidates_token_count: u32,
    pub total_token_count: u32,
}

use std::future::IntoFuture;
use std::pin::Pin;

/// Builder for Gemini content generation
pub struct GeminiCompletionBuilder<'a, S> {
    client: &'a Client<S>,
    model: String,
    system_prompt: String,
    user_prompt: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl<'a, S> GeminiCompletionBuilder<'a, S>
where
    S: HasProvider<Gemini> + Send + Sync + 'static,
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

impl<'a, S> IntoFuture for GeminiCompletionBuilder<'a, S>
where
    S: HasProvider<Gemini> + Send + Sync + Clone + 'static,
{
    type Output = Result<String, LLMError>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let contents = vec![GeminiContent::user(self.user_prompt)];
            let system_instruction = Some(GeminiContent::system(self.system_prompt));

            let generation_config = Some(GeminiGenerationConfig {
                temperature: self.temperature,
                max_output_tokens: self.max_tokens,
                ..Default::default()
            });

            let response = self
                .client
                .call_gemini(self.model, contents, system_instruction, generation_config)
                .await?;

            response
                .candidates
                .first()
                .and_then(|c| c.content.parts.first())
                .and_then(|p| p.text.clone())
                .ok_or_else(|| LLMError::InvalidResponse("No text in response".to_string()))
        })
    }
}

impl<S> Client<S>
where
    S: HasProvider<Gemini> + Clone + Send + Sync + 'static,
{
    /// Call Gemini's generate content API
    ///
    /// # Arguments
    /// * `model` - Model to use (e.g., "gemini-pro", "gemini-1.5-flash")
    /// * `contents` - Conversation contents
    /// * `system_instruction` - Optional system instruction
    /// * `generation_config` - Optional generation configuration
    ///
    /// # Example
    /// ```ignore
    /// let client = Client::new().with_gemini("your-api-key");
    /// let contents = vec![GeminiContent::user("Hello!")];
    /// let response = client.call_gemini(
    ///     "gemini-1.5-flash",
    ///     contents,
    ///     Some(GeminiContent::system("You are helpful.")),
    ///     None
    /// ).await?;
    /// ```
    pub async fn call_gemini(
        &self,
        model: impl Into<String>,
        contents: Vec<GeminiContent>,
        system_instruction: Option<GeminiContent>,
        generation_config: Option<GeminiGenerationConfig>,
    ) -> Result<GeminiResponse, LLMError> {
        let config = self.gemini_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("Gemini not configured".to_string())
        })?;

        let model_name = model.into();
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            config.base_url, model_name, config.api_key
        );

        let request = GeminiRequest {
            contents,
            system_instruction,
            generation_config,
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::GeminiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let gemini_response: GeminiResponse = response.json().await?;
        Ok(gemini_response)
    }

    /// Convenience method for simple single-turn completions using a builder pattern
    ///
    /// # Example
    /// ```ignore
    /// let result = client.gemini_complete("gemini-1.5-flash", "You are a helpful assistant.", "Hello!")
    ///     .temperature(0.7)
    ///     .await?;
    /// ```
    pub fn gemini_complete(
        &self,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        user_prompt: impl Into<String>,
    ) -> GeminiCompletionBuilder<'_, S> {
        GeminiCompletionBuilder::new(self, model, system_prompt, user_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_content_constructors() {
        let user = GeminiContent::user("Hello");
        assert_eq!(user.role, Some("user".to_string()));

        let model = GeminiContent::model("Hi there!");
        assert_eq!(model.role, Some("model".to_string()));

        let system = GeminiContent::system("Be helpful");
        assert_eq!(system.role, None);
    }

    #[test]
    fn test_gemini_request_serialization() {
        let request = GeminiRequest {
            contents: vec![GeminiContent::user("Test")],
            system_instruction: Some(GeminiContent::system("Be concise")),
            generation_config: Some(GeminiGenerationConfig {
                temperature: Some(0.5),
                ..Default::default()
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("contents"));
        assert!(json.contains("systemInstruction")); // camelCase
        assert!(json.contains("generationConfig"));
    }
}
