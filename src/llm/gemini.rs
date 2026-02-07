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
    /// Default model to use (default: gemini-3-flash-preview)
    pub default_model: String,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            default_model: "gemini-3-flash-preview".to_string(),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
}

impl Default for GeminiGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(40),
            max_output_tokens: None,
            stop_sequences: None,
            response_mime_type: None,
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
    pub(crate) client: &'a Client<S>,
    pub(crate) model: Option<String>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) contents: Vec<GeminiContent>,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) stop_sequences: Option<Vec<String>>,
    pub(crate) json_mode: bool,
}

impl<'a, S> GeminiCompletionBuilder<'a, S> {
    pub fn new(client: &'a Client<S>) -> Self {
        Self {
            client,
            model: None,
            system_prompt: None,
            contents: Vec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            json_mode: false,
        }
    }
}

impl<'a, S> GeminiCompletionBuilder<'a, S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Set the model for this completion (overrides default)
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the system prompt
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.system_prompt = Some(content.into());
        self
    }

    /// Add a user message
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.contents.push(GeminiContent::user(content));
        self
    }

    /// Add an assistant (model) message
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.contents.push(GeminiContent::model(content));
        self
    }

    /// Seed the conversation with existing contents
    pub fn messages(mut self, contents: Vec<GeminiContent>) -> Self {
        self.contents.extend(contents);
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

    /// Set the top-k sampling value
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
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
        let config = self.client.gemini_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("Gemini not configured".to_string())
        })?;

        let model_to_use = self.model.unwrap_or_else(|| config.default_model.clone());

        // Implicit validation
        let mut cache = self.client.model_cache.gemini.read().unwrap().clone();
        if cache.is_none() {
            if let Ok(models) = self.client.gemini_list_models().await {
                let names: Vec<String> = models
                    .into_iter()
                    .map(|m| {
                        m.name
                            .strip_prefix("models/")
                            .unwrap_or(&m.name)
                            .to_string()
                    })
                    .collect();
                *self.client.model_cache.gemini.write().unwrap() = Some(names.clone());
                cache = Some(names);
            }
        }

        if let Some(valid_models) = cache {
            if !valid_models.contains(&model_to_use) {
                return Err(LLMError::InvalidModel(format!(
                    "Model '{}' not found in Gemini available models",
                    model_to_use
                )));
            }
        }

        let system_instruction = self.system_prompt.map(GeminiContent::system);

        let generation_config = Some(GeminiGenerationConfig {
            temperature: self.temperature,
            max_output_tokens: self.max_tokens,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences,
            response_mime_type: if self.json_mode { Some("application/json".to_string()) } else { None },
        });

        let response = self
            .client
            .call_gemini(
                model_to_use,
                self.contents,
                system_instruction,
                generation_config,
                self.json_mode,
            )
            .await?;

        response
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .and_then(|p| p.text.clone())
            .ok_or_else(|| LLMError::InvalidResponse("No text in response".to_string()))
    }
}

impl<'a, S> IntoFuture for GeminiCompletionBuilder<'a, S>
where
    S: HasProvider<Gemini> + Send + Sync + Clone + 'static,
{
    type Output = Result<String, LLMError>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Model information from Gemini
#[derive(Debug, Deserialize)]
pub struct GeminiModel {
    pub name: String,
    pub version: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
}

#[derive(Debug, Deserialize)]
struct GeminiModelsResponse {
    pub models: Vec<GeminiModel>,
}

impl<S> Client<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// List available models from Gemini
    pub async fn gemini_list_models(&self) -> Result<Vec<GeminiModel>, LLMError> {
        let config = self.gemini_config.as_ref().ok_or_else(|| {
            LLMError::ProviderNotConfigured("Gemini not configured".to_string())
        })?;

        let url = format!("{}/v1beta/models?key={}", config.base_url, config.api_key);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(LLMError::GeminiError(format!(
                "Failed to list models: HTTP {}",
                response.status()
            )));
        }

        let res: GeminiModelsResponse = response.json().await?;
        Ok(res.models)
    }

    /// Call Gemini's generate content API
    pub async fn call_gemini(
        &self,
        model: impl Into<String>,
        contents: Vec<GeminiContent>,
        system_instruction: Option<GeminiContent>,
        generation_config: Option<GeminiGenerationConfig>,
        _json_mode: bool,
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

    pub fn gemini_complete(&self) -> GeminiCompletionBuilder<'_, S> 
    where S: HasProvider<Gemini>
    {
        self.gemini_complete_internal()
    }

    pub(crate) fn gemini_complete_internal(&self) -> GeminiCompletionBuilder<'_, S> {
        GeminiCompletionBuilder::new(self)
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
        assert!(json.contains("systemInstruction"));
        assert!(json.contains("generationConfig"));
    }
}
