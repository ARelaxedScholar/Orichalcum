use std::collections::HashMap;

use crate::core::sync_impl::node::NodeCore;
use crate::core::sync_impl::AsAny;
use crate::core::sync_impl::NodeValue;
use crate::core::Executable;

use async_trait::async_trait;

/// Async Node
pub struct AsyncNode {
    pub data: NodeCore,
    pub behaviour: Box<dyn AsyncNodeLogic>,
}

impl Clone for AsyncNode {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            behaviour: self.behaviour.clone_box(),
        }
    }
}

impl AsyncNode {
    pub fn new<L: AsyncNodeLogic>(behaviour: L) -> Self {
        AsyncNode {
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

    pub async fn run(&self, shared: &mut HashMap<String, NodeValue>) -> Option<String> {
        let p = self.behaviour.prep(&self.data.params, shared).await;
        let e = self.behaviour.exec(p.clone()).await;
        self.behaviour.post(shared, p, e).await
    }

    pub async fn run_with_params(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        param: &HashMap<String, NodeValue>,
    ) -> Option<String> {
        let p = self.behaviour.prep(param, shared).await;
        let e = self.behaviour.exec(p.clone()).await;
        self.behaviour.post(shared, p, e).await
    }
}

// More or less the same logic as NodeLogic
#[async_trait]
pub trait AsyncNodeLogic: AsAny + Send + Sync + 'static {
    // Required method
    fn clone_box(&self) -> Box<dyn AsyncNodeLogic>;

    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        _shared: &HashMap<String, NodeValue>,
    ) -> NodeValue;
    async fn exec(&self, _input: NodeValue) -> NodeValue;
    async fn post(
        &self,
        _shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        _exec_res: NodeValue,
    ) -> Option<String>;
}
