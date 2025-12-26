use crate::core::async_impl::async_node::{AsyncNode, AsyncNodeLogic};
use crate::core::sync_impl::NodeValue;
use crate::core::{Executable, Executable::Async, Executable::Sync};
use async_trait::async_trait;
use std::collections::HashMap;

/// The logic that is specif
#[derive(Clone)]
pub struct AsyncFlowLogic {
    start: Executable,
}

/// A flow really, just is a Node with orchestration logic
/// to enforce that, we will create a NewType with a "factory" which prebuilds it.
#[derive(Clone)]
pub struct AsyncFlow(AsyncNode);

/// The Derefs are needed to be able to access the inside `Node` of the `Flow` easily
impl std::ops::Deref for AsyncFlow {
    type Target = AsyncNode;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for AsyncFlow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsyncFlow {
    pub fn new(start: Executable) -> AsyncFlow {
        AsyncFlow(AsyncNode::new(AsyncFlowLogic { start }))
    }

    pub fn start(&mut self, start: Executable) {
        // extract the `NodeLogic` from the Flow
        let behaviour: &mut dyn AsyncNodeLogic = &mut *self.behaviour;

        if let Some(flow_logic) = behaviour.as_any_mut().downcast_mut::<AsyncFlowLogic>() {
            // Should always be possible if the Flow as created through the factory
            flow_logic.start = start;
        } else {
            // This should never happen, but somehow it did
            panic!("Error: Flow's logic is not of type FlowLogic");
        }
    }
}

#[async_trait]
impl AsyncNodeLogic for AsyncFlowLogic {
    async fn prep(
        &self,
        params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        serde_json::to_value((params, shared)).expect("If this works, I'll be so lit")
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        //  This is the init (Basically, we deserialize the value that was passed from the previous
        //  step)
        let (params, mut shared): (HashMap<String, NodeValue>, HashMap<String, NodeValue>) =
            if let Some(arr) = input.as_array() {
                if arr.len() != 2 {
                    log::error!("serde_json::to_value() failed to convert the params and shared.");
                    (HashMap::new(), HashMap::new())
                } else {
                    // We can proceed
                    let params = serde_json::from_value(arr[0].clone()).unwrap_or_default();
                    let shared = serde_json::from_value(arr[1].clone()).unwrap_or_default();
                    (params, shared)
                }
            } else {
                (HashMap::new(), HashMap::new())
            };
        let mut current: Option<Executable> = Some(self.start.clone());
        let mut last_action: String = "".into();

        // This is the orchestration logic
        while let Some(mut curr) = current {
            last_action = match curr {
                Sync(ref mut sync_node) => {
                    let mut sync_clone = sync_node.clone();
                    sync_clone.set_params(params.clone());
                    let mut shared_clone = shared.clone();

                    // not ideal, but not cloning here would
                    // require a significant refactoring
                    // afaik (switching everything to use
                    // Arc/Rc)
                    // Will be next step if benchmarking shows me this is actually
                    // worth the hassle
                    match tokio::task::spawn_blocking(move || {
                        let action = sync_clone
                            .run(&mut shared_clone)
                            .unwrap_or("default".into());
                        (action, shared_clone)
                    })
                    .await
                    {
                        Ok((next_action, modified_shared)) => {
                            // Happy path: the task completed successfully
                            shared = modified_shared;
                            next_action
                        }
                        Err(join_error) => {
                            // The background task panicked!
                            log::error!("A synchronous node panicked: {:?}", join_error);
                            // For now, just log it and go to the default action
                            "default".into()
                        }
                    }
                }
                Async(ref mut async_node) => {
                    async_node.set_params(params.clone());
                    async_node
                        .run(&mut shared)
                        .await
                        .unwrap_or("default".into())
                }
            };

            // Uses method implemented on Executable
            let next_executable = &curr.successors().get(&last_action).cloned();

            current = next_executable.clone();
        }
        serde_json::to_value((last_action.to_string(), shared))
            .expect("Serializing string and HashMap should be doable")
    }
    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        let (last_action, shared_post): (String, HashMap<String, NodeValue>) = if let Some(array) =
            exec_res.as_array()
        {
            if array.len() != 2 {
                log::error!(
                    "Serialization into array succeeded, but got unexpected length for array: {}! Returning default values",
                    array.len()
                );
                ("default".into(), shared.clone())
            } else {
                // happy path
                let last_action = serde_json::from_value(array[0].clone()).unwrap_or_default();
                let shared = serde_json::from_value(array[1].clone()).unwrap_or_default();
                (last_action, shared)
            }
        } else {
            log::error!("Serialization of last action and shared failed! Returning default values");
            ("default".into(), shared.clone())
        };

        // modify the shared state
        *shared = shared_post;

        // return the final action (since Flow is also just a node)
        Some(last_action)
    }
    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new((*self).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::async_impl::async_node::AsyncNodeLogic;
    use crate::core::sync_impl::node::{Node, NodeLogic};
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct SimpleAsyncLogic {
        id: String,
        next_action: Option<String>,
    }

    #[async_trait]
    impl AsyncNodeLogic for SimpleAsyncLogic {
        async fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            _shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            json!(self.id.clone())
        }

        async fn exec(&self, input: NodeValue) -> NodeValue {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            input
        }

        async fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            prep_res: NodeValue,
            _exec_res: NodeValue,
        ) -> Option<String> {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            let id = prep_res.as_str().unwrap_or("unknown");
            shared.insert(format!("visited_{}", id), json!(true));
            self.next_action.clone()
        }

        fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct SimpleSyncLogic {
        id: String,
        next_action: Option<String>,
    }

    impl NodeLogic for SimpleSyncLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            _shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            json!(self.id.clone())
        }

        fn exec(&self, input: NodeValue) -> NodeValue {
            input
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            prep_res: NodeValue,
            _exec_res: NodeValue,
        ) -> Option<String> {
            let id = prep_res.as_str().unwrap_or("unknown");
            shared.insert(format!("visited_{}", id), json!(true));
            self.next_action.clone()
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn test_async_flow_creation() {
        let async_node = AsyncNode::new(SimpleAsyncLogic {
            id: "test".to_string(),
            next_action: None,
        });
        let flow = AsyncFlow::new(Executable::Async(async_node));

        // Flow should deref to AsyncNode
        assert!(flow.data.params.is_empty());
    }

    #[tokio::test]
    async fn test_async_flow_single_async_node() {
        let async_node = AsyncNode::new(SimpleAsyncLogic {
            id: "single".to_string(),
            next_action: None,
        });
        let flow = AsyncFlow::new(Executable::Async(async_node));
        let mut shared = HashMap::new();

        let action = flow.run(&mut shared).await;

        assert_eq!(shared.get("visited_single"), Some(&json!(true)));
        assert_eq!(action, Some("default".to_string()));
    }

    #[tokio::test]
    async fn test_async_flow_single_sync_node() {
        let sync_node = Node::new(SimpleSyncLogic {
            id: "sync_single".to_string(),
            next_action: None,
        });
        let flow = AsyncFlow::new(Executable::Sync(sync_node));
        let mut shared = HashMap::new();

        let action = flow.run(&mut shared).await;

        assert_eq!(shared.get("visited_sync_single"), Some(&json!(true)));
        assert_eq!(action, Some("default".to_string()));
    }

    #[tokio::test]
    async fn test_async_flow_mixed_nodes() {
        // Create a chain: async -> sync -> async
        let node3 = AsyncNode::new(SimpleAsyncLogic {
            id: "async3".to_string(),
            next_action: None,
        });

        let node2 = Node::new(SimpleSyncLogic {
            id: "sync2".to_string(),
            next_action: Some("default".to_string()),
        })
        .next(Executable::Async(node3));

        let node1 = AsyncNode::new(SimpleAsyncLogic {
            id: "async1".to_string(),
            next_action: Some("default".to_string()),
        })
        .next(Executable::Sync(node2));

        let flow = AsyncFlow::new(Executable::Async(node1));
        let mut shared = HashMap::new();

        let action = flow.run(&mut shared).await;

        // All nodes should have been visited
        assert_eq!(shared.get("visited_async1"), Some(&json!(true)));
        assert_eq!(shared.get("visited_sync2"), Some(&json!(true)));
        assert_eq!(shared.get("visited_async3"), Some(&json!(true)));
        assert_eq!(action, Some("default".to_string()));
    }

    #[tokio::test]
    async fn test_async_flow_branching() {
        // Create branching with mixed node types
        let async_branch = AsyncNode::new(SimpleAsyncLogic {
            id: "async_branch".to_string(),
            next_action: None,
        });

        let sync_branch = Node::new(SimpleSyncLogic {
            id: "sync_branch".to_string(),
            next_action: None,
        });

        #[derive(Clone)]
        struct BranchingAsyncLogic {
            branch_to: String,
        }

        #[async_trait]
        impl AsyncNodeLogic for BranchingAsyncLogic {
            async fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                json!(self.branch_to.clone())
            }

            async fn exec(&self, input: NodeValue) -> NodeValue {
                input
            }

            async fn post(
                &self,
                shared: &mut HashMap<String, NodeValue>,
                prep_res: NodeValue,
                _exec_res: NodeValue,
            ) -> Option<String> {
                let branch = prep_res.as_str().unwrap_or("default");
                shared.insert("branch_taken".to_string(), json!(branch));
                Some(branch.to_string())
            }

            fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
                Box::new(self.clone())
            }
        }

        let start = AsyncNode::new(BranchingAsyncLogic {
            branch_to: "async".to_string(),
        })
        .next_on("async", Executable::Async(async_branch))
        .next_on("sync", Executable::Sync(sync_branch));

        let flow = AsyncFlow::new(Executable::Async(start));
        let mut shared = HashMap::new();

        let action = flow.run(&mut shared).await;

        // Should have taken async branch
        assert_eq!(shared.get("branch_taken"), Some(&json!("async")));
        assert_eq!(shared.get("visited_async_branch"), Some(&json!(true)));
        assert!(shared.get("visited_sync_branch").is_none());
        assert_eq!(action, Some("default".to_string()));
    }

    #[tokio::test]
    async fn test_async_flow_with_params() {
        #[derive(Clone)]
        struct ParamAsyncLogic;

        #[async_trait]
        impl AsyncNodeLogic for ParamAsyncLogic {
            async fn prep(
                &self,
                params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                params.get("test_param").cloned().unwrap_or(NodeValue::Null)
            }

            async fn exec(&self, input: NodeValue) -> NodeValue {
                input
            }

            async fn post(
                &self,
                shared: &mut HashMap<String, NodeValue>,
                prep_res: NodeValue,
                _exec_res: NodeValue,
            ) -> Option<String> {
                shared.insert("received_param".to_string(), prep_res);
                None
            }

            fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
                Box::new(self.clone())
            }
        }

        let async_node = AsyncNode::new(ParamAsyncLogic);
        let flow = AsyncFlow::new(Executable::Async(async_node));
        let mut shared = HashMap::new();
        let mut params = HashMap::new();
        params.insert("test_param".to_string(), json!("test_value"));

        // Set params on the flow's internal node
        let mut flow_clone = flow.clone();
        flow_clone.set_params(params);

        let action = flow_clone.run(&mut shared).await;

        assert_eq!(shared.get("received_param"), Some(&json!("test_value")));
        assert_eq!(action, Some("default".to_string()));
    }

    #[tokio::test]
    async fn test_async_flow_sync_node_panic_handling() {
        #[derive(Clone)]
        struct PanicSyncLogic;

        impl NodeLogic for PanicSyncLogic {
            fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                NodeValue::Null
            }

            fn exec(&self, _input: NodeValue) -> NodeValue {
                NodeValue::Null
            }

            fn post(
                &self,
                _shared: &mut HashMap<String, NodeValue>,
                _prep_res: NodeValue,
                _exec_res: NodeValue,
            ) -> Option<String> {
                panic!("Sync node panicked!");
            }

            fn clone_box(&self) -> Box<dyn NodeLogic> {
                Box::new(self.clone())
            }
        }

        let sync_node = Node::new(PanicSyncLogic);
        let flow = AsyncFlow::new(Executable::Sync(sync_node));
        let mut shared = HashMap::new();

        // This should not panic in the test, but the sync node panic should be caught
        // and logged, resulting in "default" action
        let action = flow.run(&mut shared).await;

        // The flow should handle the panic and continue with default action
        assert_eq!(action, Some("default".to_string()));
    }
}
