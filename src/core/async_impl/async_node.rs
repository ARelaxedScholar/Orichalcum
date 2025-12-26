use std::collections::HashMap;

use crate::core::sync_impl::node::NodeCore;
use crate::core::sync_impl::AsAny;
use crate::core::sync_impl::NodeValue;
use crate::core::Executable;

use async_trait::async_trait;

/// An asynchronous node in a workflow graph.
///
/// Similar to [`Node`](crate::core::sync_impl::node::Node) but with async/await support.
/// An `AsyncNode` encapsulates an asynchronous unit of work with three-phase execution:
/// 1. **Prep**: Prepare inputs from parameters and shared state (async)
/// 2. **Exec**: Execute the core logic (async)
/// 3. **Post**: Process results and update shared state, optionally returning the next action (async)
///
/// Async nodes can be mixed with sync nodes in an [`AsyncFlow`](crate::core::async_impl::async_flow::AsyncFlow).
pub struct AsyncNode {
    /// Internal node data including parameters and successors
    pub data: NodeCore,
    /// The async logic implementation that defines the node's behavior
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
    /// Creates a new async node with the given logic.
    ///
    /// # Arguments
    /// * `behaviour` - An implementation of [`AsyncNodeLogic`] that defines the node's behavior
    pub fn new<L: AsyncNodeLogic>(behaviour: L) -> Self {
        AsyncNode {
            data: NodeCore::default(),
            behaviour: Box::new(behaviour),
        }
    }

    /// Sets the node's parameters.
    ///
    /// Parameters are node-specific configuration that can be accessed
    /// in the [`prep`](AsyncNodeLogic::prep) phase.
    pub fn set_params(&mut self, params: HashMap<String, NodeValue>) {
        self.data.params = params;
    }

    /// Chains another node to execute after this node via the "default" action.
    ///
    /// Equivalent to `self.next_on("default", node)`.
    pub fn next(self, node: Executable) -> Self {
        self.next_on("default", node)
    }

    /// Chains another node to execute after this node when the specified action is returned.
    ///
    /// If the node already has a successor for the given action, it will be overwritten
    /// with a warning logged.
    ///
    /// # Arguments
    /// * `action` - The action string that should trigger this successor
    /// * `node` - The node to execute when this action is returned from [`post`](AsyncNodeLogic::post)
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

    /// Executes the node with its current parameters (async).
    ///
    /// Runs the three-phase execution (prep, exec, post) using the node's
    /// stored parameters.
    ///
    /// # Arguments
    /// * `shared` - Mutable reference to the shared state
    ///
    /// # Returns
    /// The action returned by [`post`](AsyncNodeLogic::post), or `None` if the workflow should terminate
    pub async fn run(&self, shared: &mut HashMap<String, NodeValue>) -> Option<String> {
        let p = self.behaviour.prep(&self.data.params, shared).await;
        let e = self.behaviour.exec(p.clone()).await;
        self.behaviour.post(shared, p, e).await
    }

    /// Executes the node with the given parameters, ignoring stored parameters (async).
    ///
    /// Useful for one-off executions with different parameters without modifying
    /// the node's internal state.
    ///
    /// # Arguments
    /// * `shared` - Mutable reference to the shared state
    /// * `param` - Parameters to use for this execution
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

/// Defines the asynchronous behavior of a workflow node.
///
/// Implement this trait to create custom async node logic. The trait provides
/// a three-phase execution model with async support.
///
/// See [`NodeLogic`](crate::core::sync_impl::node::NodeLogic) for the synchronous version.
#[async_trait]
pub trait AsyncNodeLogic: AsAny + Send + Sync + 'static {
    /// Create a boxed clone of this trait object.
    ///
    /// Required for cloning `Box<dyn AsyncNodeLogic>`.
    fn clone_box(&self) -> Box<dyn AsyncNodeLogic>;

    /// Prepare inputs for execution (async).
    ///
    /// This phase extracts necessary data from node parameters and shared state,
    /// returning a value that will be passed to [`exec`](AsyncNodeLogic::exec).
    ///
    /// # Arguments
    /// * `params` - Node-specific parameters
    /// * `shared` - Shared state accessible to all nodes in the workflow
    ///
    /// # Returns
    /// A [`NodeValue`] to be processed by [`exec`](AsyncNodeLogic::exec)
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        _shared: &HashMap<String, NodeValue>,
    ) -> NodeValue;

    /// Execute the core logic of the node (async).
    ///
    /// This phase performs the main computation or operation using the
    /// value returned from [`prep`](AsyncNodeLogic::prep).
    ///
    /// # Arguments
    /// * `input` - The value returned by [`prep`](AsyncNodeLogic::prep)
    ///
    /// # Returns
    /// A [`NodeValue`] representing the execution result
    async fn exec(&self, _input: NodeValue) -> NodeValue;

    /// Post-process results and update shared state (async).
    ///
    /// This final phase can store results in the shared state and
    /// determine which node should execute next by returning an action string.
    ///
    /// # Arguments
    /// * `shared` - Mutable reference to the shared state
    /// * `prep_res` - The value returned by [`prep`](AsyncNodeLogic::prep)
    /// * `exec_res` - The value returned by [`exec`](AsyncNodeLogic::exec)
    ///
    /// # Returns
    /// * `Some(action)` - Execute the successor mapped to this action
    /// * `None` - Terminate the workflow
    async fn post(
        &self,
        _shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        _exec_res: NodeValue,
    ) -> Option<String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct AsyncTestLogic {
        id: String,
        next_action: Option<String>,
    }

    #[async_trait]
    impl AsyncNodeLogic for AsyncTestLogic {
        async fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            // Simulate async operation
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            shared.get(&self.id).cloned().unwrap_or(NodeValue::Null)
        }

        async fn exec(&self, input: NodeValue) -> NodeValue {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            if let Some(num) = input.as_f64() {
                json!(num * 3.0)
            } else {
                json!("async_processed")
            }
        }

        async fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            prep_res: NodeValue,
            exec_res: NodeValue,
        ) -> Option<String> {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            shared.insert(format!("{}_prep", self.id), prep_res);
            shared.insert(format!("{}_exec", self.id), exec_res);
            self.next_action.clone()
        }

        fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn test_async_node_creation() {
        let node = AsyncNode::new(AsyncTestLogic {
            id: "test".to_string(),
            next_action: None,
        });
        assert!(node.data.params.is_empty());
        assert!(node.data.successors.is_empty());
    }

    #[tokio::test]
    async fn test_async_node_set_params() {
        let mut node = AsyncNode::new(AsyncTestLogic {
            id: "test".to_string(),
            next_action: None,
        });
        let mut params = HashMap::new();
        params.insert("key".to_string(), json!("value"));
        node.set_params(params.clone());
        assert_eq!(node.data.params, params);
    }

    #[tokio::test]
    async fn test_async_node_next() {
        let node1 = AsyncNode::new(AsyncTestLogic {
            id: "node1".to_string(),
            next_action: None,
        });
        let node2 = AsyncNode::new(AsyncTestLogic {
            id: "node2".to_string(),
            next_action: None,
        });
        let node1_with_next = node1.next(Executable::Async(node2));

        assert_eq!(node1_with_next.data.successors.len(), 1);
        assert!(node1_with_next.data.successors.contains_key("default"));
    }

    #[tokio::test]
    async fn test_async_node_next_on() {
        let node1 = AsyncNode::new(AsyncTestLogic {
            id: "node1".to_string(),
            next_action: None,
        });
        let node2 = AsyncNode::new(AsyncTestLogic {
            id: "node2".to_string(),
            next_action: None,
        });
        let node1_with_next = node1.next_on("custom", Executable::Async(node2));

        assert_eq!(node1_with_next.data.successors.len(), 1);
        assert!(node1_with_next.data.successors.contains_key("custom"));
    }

    #[tokio::test]
    async fn test_async_node_run() {
        let node = AsyncNode::new(AsyncTestLogic {
            id: "test".to_string(),
            next_action: Some("default".to_string()),
        });
        let mut shared = HashMap::new();
        shared.insert("test".to_string(), json!(10));

        let action = node.run(&mut shared).await;
        assert_eq!(action, Some("default".to_string()));
        assert_eq!(shared.get("test_prep"), Some(&json!(10)));
        assert_eq!(shared.get("test_exec"), Some(&json!(30.0)));
    }

    #[tokio::test]
    async fn test_async_node_run_with_params() {
        let node = AsyncNode::new(AsyncTestLogic {
            id: "test".to_string(),
            next_action: None,
        });
        let mut shared = HashMap::new();
        let mut params = HashMap::new();
        params.insert("param".to_string(), json!("value"));

        let action = node.run_with_params(&mut shared, &params).await;
        assert_eq!(action, None); // next_action is None
                                  // prep should be null since "test" key doesn't exist in shared
        assert_eq!(shared.get("test_prep"), Some(&NodeValue::Null));
        assert_eq!(shared.get("test_exec"), Some(&json!("async_processed")));
    }

    #[tokio::test]
    async fn test_async_node_logic_required_methods() {
        #[derive(Clone)]
        struct SimpleAsyncLogic;

        #[async_trait]
        impl AsyncNodeLogic for SimpleAsyncLogic {
            async fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                NodeValue::Null
            }

            async fn exec(&self, _input: NodeValue) -> NodeValue {
                NodeValue::Null
            }

            async fn post(
                &self,
                _shared: &mut HashMap<String, NodeValue>,
                _prep_res: NodeValue,
                _exec_res: NodeValue,
            ) -> Option<String> {
                None
            }

            fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
                Box::new(self.clone())
            }
        }

        let logic = SimpleAsyncLogic;
        let params = HashMap::new();
        let shared = HashMap::new();
        let mut shared_mut = HashMap::new();

        let prep = logic.prep(&params, &shared).await;
        assert_eq!(prep, NodeValue::Null);

        let exec = logic.exec(NodeValue::Null).await;
        assert_eq!(exec, NodeValue::Null);

        let post = logic
            .post(&mut shared_mut, NodeValue::Null, NodeValue::Null)
            .await;
        assert_eq!(post, None);
    }
}
