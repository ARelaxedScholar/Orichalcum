use std::sync::Arc;
use crate::core::Executable;
use crate::core::sync_impl::NodeValue;
use std::collections::HashMap;
use crate::core::telemetry::Telemetry;
use crate::core::semantic::Sealable;
use crate::core::semantic::signature::Signature;
use futures::future::BoxFuture;

/// An immutable wrapper that encapsulates any executable unit.
/// Sealing a node snapshots its identity (hashes) and locks its logic for production or optimization.
pub struct SealedNode {
    pub(crate) inner: Executable,
    pub(crate) task_id: String,
    pub(crate) signature: Signature,
    pub(crate) signature_hash: String,
    pub(crate) instruction_hash: String,
    pub(crate) model_name: String,
    
    // Optimization metadata slots
    pub training_hash: Option<String>,
    pub optimization_config_hash: Option<String>,
    pub fitness_score: Option<f64>,
    pub weights_path: Option<std::path::PathBuf>,
}

impl SealedNode {
    /// Creates a new sealed node.
    pub fn new(
        inner: Executable,
        task_id: String,
        signature: Signature,
        signature_hash: String,
        instruction_hash: String,
        model_name: String,
    ) -> Self {
        Self {
            inner,
            task_id,
            signature,
            signature_hash,
            instruction_hash,
            model_name,
            training_hash: None,
            optimization_config_hash: None,
            fitness_score: None,
            weights_path: None,
        }
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn signature_hash(&self) -> &str {
        &self.signature_hash
    }

    pub fn instruction_hash(&self) -> &str {
        &self.instruction_hash
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Executes the sealed node by delegating to its inner logic and recording telemetry.
    pub fn run<'a>(
        &'a self,
        shared: &'a mut HashMap<String, NodeValue>,
        telemetry: Option<&'a dyn Telemetry>,
    ) -> BoxFuture<'a, Option<String>> {
        Box::pin(async move {
            // We need to capture the results for telemetry
            // This is easier if we call the phases directly or if run() returns them.
            // For now, we'll let the inner run() do its work, and we'll record telemetry here
            // but we need the inputs/outputs.
            
            // Re-implementing the run loop here to capture I/O
            let (p, e, action) = match &self.inner {
                Executable::Sync(node) => {
                    let p = node.behaviour.prep(&node.data.params, shared);
                    let e = node.behaviour.exec(p.clone());
                    let action = node.behaviour.post(shared, p.clone(), e.clone());
                    (p, e, action)
                }
                Executable::Async(node) => {
                    let p = node.behaviour.prep(&node.data.params, shared).await;
                    let e = node.behaviour.exec(p.clone()).await;
                    let action = node.behaviour.post(shared, p.clone(), e.clone()).await;
                    (p, e, action)
                }
                Executable::Sealed(sealed) => {
                    // Nested sealed nodes will record their own telemetry
                    return sealed.run(shared, telemetry).await;
                }
            };

            if let Some(t) = telemetry {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                t.record(crate::core::telemetry::TraceEntry {
                    timestamp,
                    task_id: self.task_id.clone(),
                    signature_hash: self.signature_hash.clone(),
                    instruction_hash: self.instruction_hash.clone(),
                    inputs: p,
                    outputs: e,
                    model_name: self.model_name.clone(),
                    training_hash: self.training_hash.clone(),
                    fitness_score: self.fitness_score,
                    metadata: HashMap::new(),
                });
            }

            action
        })
    }
    
    pub fn inner(&self) -> &Executable {
        &self.inner
    }
}

impl Sealable for SealedNode {
    fn signature(&self) -> Signature {
        self.signature.clone()
    }

    fn task_id(&self) -> String {
        self.task_id.clone()
    }
}
