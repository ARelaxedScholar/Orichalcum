use crate::core::sync_impl::AsAny;
use crate::core::sync_impl::NodeValue;
use crate::core::Executable;
use crate::core::semantic::{Promptable, Sealable};
use crate::core::sealed::SealedNode;
use crate::core::telemetry::{Telemetry, TraceEntry};
use std::collections::HashMap;
use std::sync::Arc;

/// A node in a workflow graph.
pub struct Node {
    /// Internal node data including parameters and successors
    pub data: NodeCore,
    /// The logic implementation that defines the node's behavior
    pub behaviour: Box<dyn NodeLogic>,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            behaviour: self.behaviour.clone_box(),
        }
    }
}

impl Node {
    /// Creates a new node with the given logic.
    pub fn new<L: NodeLogic + 'static>(behaviour: L) -> Self {
        Node {
            data: NodeCore::default(),
            behaviour: Box::new(behaviour),
        }
    }

    /// Seals the node, making it immutable and Snapshotting its identity.
    pub fn seal(self) -> Result<Arc<SealedNode>, String> {
        let sealable = self.behaviour.as_sealable()
            .ok_or_else(|| "Node behavior does not implement Sealable".to_string())?;
        
        let task_id = sealable.task_id();
        let signature = sealable.signature();
        let sig_hash = signature.structural_hash();
        
        let instr_hash = if let Some(promptable) = self.behaviour.as_promptable() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            promptable.instruction().hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        } else {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            let mut keys: Vec<_> = self.data.params.keys().collect();
            keys.sort();
            for k in keys {
                k.hash(&mut hasher);
                if let Some(s) = self.data.params.get(k).and_then(|v| v.as_str()) {
                    s.hash(&mut hasher);
                }
            }
            format!("{:016x}", hasher.finish())
        };

        let model_name = self.behaviour.as_promptable()
            .and_then(|p| p.model())
            .unwrap_or("native")
            .to_string();

        Ok(Arc::new(SealedNode::new(
            Executable::Sync(self),
            task_id,
            signature,
            sig_hash,
            instr_hash,
            model_name,
        )))
    }

    /// Sets the node's parameters.
    pub fn set_params(&mut self, params: HashMap<String, NodeValue>) {
        self.data.params = params;
    }

    /// Chains another node to execute after this node via the "default" action.
    pub fn next(self, node: Executable) -> Self {
        self.next_on("default", node)
    }

    /// Chains another node to execute after this node when the specified action is returned.
    pub fn next_on(mut self, action: &str, node: Executable) -> Self {
        if self.data.successors.contains_key(action) {
            log::warn!(
                "Warning: Action {} was found in successors, Overwriting key {}.",
                &action,
                &action
            );
        }
        self.data.successors.insert(action.to_string(), node);
        self
    }

    /// Executes the node with its current parameters.
    pub fn run(&self, shared: &mut HashMap<String, NodeValue>) -> Option<String> {
        self.run_with_telemetry(shared, None)
    }

    /// Executes the node and records telemetry.
    pub fn run_with_telemetry(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _telemetry: Option<&dyn Telemetry>,
    ) -> Option<String> {
        let p = self.behaviour.prep(&self.data.params, shared);
        let e = self.behaviour.exec(p.clone());
        self.behaviour.post(shared, p, e)
    }

    /// Executes the node with the given parameters, ignoring stored parameters.
    pub fn run_with_params(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        param: &HashMap<String, NodeValue>,
    ) -> Option<String> {
        let p = self.behaviour.prep(param, shared);
        let e = self.behaviour.exec(p.clone());
        self.behaviour.post(shared, p, e)
    }
}

/// Internal data structure for a node.
#[derive(Default, Clone)]
pub struct NodeCore {
    /// Parameters specific to this node instance
    pub params: HashMap<String, NodeValue>,
    /// Mapping from action strings to successor nodes
    pub successors: HashMap<String, Executable>,
}

/// Defines the behavior of a workflow node.
pub trait NodeLogic: AsAny + Send + Sync + 'static {
    /// Prepare inputs for execution.
    fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        _shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        NodeValue::default()
    }

    /// Execute the core logic of the node.
    fn exec(&self, _input: NodeValue) -> NodeValue {
        NodeValue::default()
    }

    /// Post-process results and update shared state.
    fn post(
        &self,
        _shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        _exec_res: NodeValue,
    ) -> Option<String> {
        None
    }

    /// Create a boxed clone of this trait object.
    fn clone_box(&self) -> Box<dyn NodeLogic>;

    /// Optional: Returns a reference to the sealable interface if this node implements it.
    fn as_sealable(&self) -> Option<&dyn Sealable> {
        None
    }

    /// Optional: Returns a reference to the promptable interface if this node implements it.
    fn as_promptable(&self) -> Option<&dyn Promptable> {
        None
    }
}

impl Clone for Box<dyn NodeLogic> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
