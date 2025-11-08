use crate::core::sync_impl::AsAny;
use crate::core::sync_impl::NodeValue;
use crate::core::Executable;
use std::collections::HashMap;

/// ------ Base Node Logic -------------------------------------------------------
/// Defines the fundamental logic that is common to any "Node" of the system
#[derive(Clone)]
pub struct Node {
    pub data: NodeCore,
    pub behaviour: Box<dyn NodeLogic>,
}

impl Node {
    pub fn new<L: NodeLogic + 'static>(behaviour: L) -> Self {
        Node {
            data: NodeCore::default(),
            behaviour: Box::new(behaviour),
        }
    }
    pub fn set_params(&mut self, params: HashMap<String, NodeValue>) {
        self.data.params = params;
    }
    pub fn next(self, node: Executable) -> Self {
        self.next_on("default", node)
    }
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

    pub fn run(&self, shared: &mut HashMap<String, NodeValue>) -> Option<String> {
        let p = self.behaviour.prep(&self.data.params, shared);
        let e = self.behaviour.exec(p.clone());
        self.behaviour.post(shared, p, e)
    }

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

#[derive(Default, Clone)]
pub struct NodeCore {
    pub params: HashMap<String, NodeValue>,
    pub successors: HashMap<String, Executable>,
}

pub trait NodeLogic: AsAny + Send + Sync + 'static {
    fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        _shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        NodeValue::default()
    }
    fn exec(&self, _input: NodeValue) -> NodeValue {
        NodeValue::default()
    }
    fn post(
        &self,
        _shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        _exec_res: NodeValue,
    ) -> Option<String> {
        None
    }

    fn clone_box(&self) -> Box<dyn NodeLogic>;
}

impl Clone for Box<dyn NodeLogic> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
