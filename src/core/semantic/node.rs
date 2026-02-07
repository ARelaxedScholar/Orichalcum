use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::llm::Client;
use crate::LLMError;
use crate::core::async_impl::async_node::{AsyncNodeLogic, AsyncNode};
use crate::core::sync_impl::NodeValue;
use crate::core::semantic::{Sealable, Promptable};
use crate::core::Executable;
use crate::core::sealed::SealedNode;
use crate::core::semantic::signature::Signature;

/// Vanilla logic for a semantic LLM node.
#[derive(Clone)]
pub struct SemanticLLMLogic<S> {
    client: Client<S>,
    signature: Signature,
    instruction: String,
    task_id: String,
    model_override: Option<String>,
}

impl<S> SemanticLLMLogic<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub fn new(
        client: Client<S>,
        signature: Signature,
        instruction: String,
        task_id: String,
    ) -> Self {
        Self {
            client,
            signature,
            instruction,
            task_id,
            model_override: None,
        }
    }

    fn instruction_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.instruction.hash(&mut hasher);
        for field in &self.signature.inputs {
            field.description.hash(&mut hasher);
        }
        for field in &self.signature.outputs {
            field.description.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    }
}

#[async_trait]
impl<S> AsyncNodeLogic for SemanticLLMLogic<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let mut inputs = HashMap::new();
        for field in &self.signature.inputs {
            if let Some(val) = shared.get(&field.name) {
                inputs.insert(field.name.clone(), val.clone());
            } else {
                inputs.insert(field.name.clone(), NodeValue::Null);
            }
        }
        json!(inputs)
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let mut prompt = format!("Task Instruction: {}\n\nInput Data:\n{}\n\n", self.instruction, input);
        prompt.push_str("Respond ONLY with a valid JSON object matching the following output keys:\n");
        for field in &self.signature.outputs {
            prompt.push_str(&format!("- {}: {}\n", field.name, field.description));
        }

        let model = self.model_override.clone();
        let result: Result<String, LLMError> = self.execute_llm(&prompt, model).await;
        
        match result {
            Ok(json_str) => json!(json_str),
            Err(e) => json!({ "error": e.to_string() }),
        }
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        if let Some(json_str) = exec_res.as_str() {
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(json_str) {
                for field in &self.signature.outputs {
                    if let Some(val) = map.get(&field.name) {
                        shared.insert(field.name.clone(), val.clone());
                    } else {
                        log::warn!("LLM missed required output field: {}", field.name);
                    }
                }
            } else if let Ok(Value::Object(map)) = serde_json::from_value::<Value>(exec_res.clone()) {
                // If exec_res is already an object (though expected string from execute_llm)
                 for field in &self.signature.outputs {
                    if let Some(val) = map.get(&field.name) {
                        shared.insert(field.name.clone(), val.clone());
                    }
                }
            }
        }
        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new(self.clone())
    }

    fn as_sealable(&self) -> Option<&dyn Sealable> {
        Some(self)
    }

    fn as_promptable(&self) -> Option<&dyn Promptable> {
        Some(self)
    }
}

impl<S> SemanticLLMLogic<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn execute_llm(&self, prompt: &str, model: Option<String>) -> Result<String, LLMError> {
        // We cannot call specific provider methods directly because S is generic.
        // However, we can use the trait bounds if we had them, or use the config flags.
        // Since we want this to be generic, we'll use a dispatch approach.
        
        // This requires the Client methods to be available without specific typestate if configs are present,
        // but currently they are constrained by HasProvider<T>.
        
        // Let's use a workaround: since we are inside the crate, we can see the configs.
        // We'll implement a hidden method on Client<S> that allows dispatching.
        
        self.client.dispatch_complete(prompt, model).await
    }
}

impl<S> Sealable for SemanticLLMLogic<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn signature(&self) -> Signature {
        self.signature.clone()
    }

    fn task_id(&self) -> String {
        self.task_id.clone()
    }
}

impl<S> Promptable for SemanticLLMLogic<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn instruction(&self) -> Option<&str> {
        Some(&self.instruction)
    }

    fn model(&self) -> Option<&str> {
        self.model_override.as_deref()
    }
}

/// Builder for creating semantic LLM nodes.
pub struct SemanticNodeBuilder<S> {
    client: Client<S>,
    signature: Option<Signature>,
    instruction: Option<String>,
    task_id: Option<String>,
    model_override: Option<String>,
}

impl<S> SemanticNodeBuilder<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub fn new(client: Client<S>) -> Self {
        Self {
            client,
            signature: None,
            instruction: None,
            task_id: None,
            model_override: None,
        }
    }

    pub fn signature(mut self, sig: impl Into<Signature>) -> Self {
        self.signature = Some(sig.into());
        self
    }

    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    pub fn task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }

    pub fn seal(self) -> Executable {
        let signature = self.signature.expect("Signature is required for semantic node");
        let instruction = self.instruction.expect("Instruction is required for semantic node");
        
        let task_id = self.task_id.unwrap_or_else(|| {
            let id = format!("autogen_{}", uuid::Uuid::new_v4().simple());
            log::warn!(
                "Auto-generated task_id '{}'. For production stability, use .task_id().",
                id
            );
            id
        });

        let mut logic = SemanticLLMLogic::new(self.client.clone(), signature.clone(), instruction, task_id.clone());
        logic.model_override = self.model_override;

        let sig_hash = signature.structural_hash();
        let instr_hash = logic.instruction_hash();
        let model_name = logic.execute_model_name();

        let node = AsyncNode::new(logic);
        Executable::Sealed(Arc::new(SealedNode::new(
            Executable::Async(node),
            task_id,
            signature,
            sig_hash,
            instr_hash,
            model_name,
        )))
    }
}

impl<S> SemanticLLMLogic<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn execute_model_name(&self) -> String {
        if let Some(m) = &self.model_override {
            return m.clone();
        }
        
        if let Some(config) = &self.client.deepseek_config { return config.default_model.clone(); }
        if let Some(config) = &self.client.gemini_config { return config.default_model.clone(); }
        if let Some(config) = &self.client.ollama_config { return config.default_model.clone(); }
        
        "unknown".to_string()
    }
}

impl<S> Client<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Creates a builder for a semantic node tied to this client.
    pub fn semantic_node(&self) -> SemanticNodeBuilder<S> {
        SemanticNodeBuilder::new(self.clone())
    }
}
