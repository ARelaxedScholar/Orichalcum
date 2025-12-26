use chrono::{DateTime, Utc};
use serde_json::json;

use crate::llm::{error::LLMError, Client, HasProvider};

pub struct Ollama;

#[derive(serde::Deserialize, Debug)]
pub struct OllamaResponse {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub response: String,
    pub done: bool,
    pub done_reason: String,
    pub context: Vec<u32>,
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: u32,
    pub prompt_eval_duration: u64,
    pub eval_count: u32,
    pub eval_duration: u64,
}

impl<S> Client<S>
where
    S: HasProvider<Ollama>,
{
    pub async fn call_ollama(
        &self,
        model: impl Into<String>,
        prompt: impl Into<String>,
        stream: bool,
    ) -> Result<OllamaResponse, LLMError> {
        // Extract the config
        let ollama_host: &str = self
            .ollama_host
            .as_ref()
            .expect("Client<S> should have Some<Ollama> when HasProvider<Ollama> is true");

        // Create the payload for querying Ollama
        let payload = json!({
            "model": model.into(),
            "prompt": prompt.into(),
            "stream": stream
        });

        // Create the response
        let response = self
            .client
            .post(format!("{}/api/generate", ollama_host))
            .json(&payload)
            .send()
            .await?
            .json::<OllamaResponse>()
            .await?;

        // Return the extracted response
        Ok(response)
    }
}
