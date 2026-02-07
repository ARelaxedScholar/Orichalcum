use crate::core::async_impl::async_node::{AsyncNode, AsyncNodeLogic};
use crate::core::sync_impl::NodeValue;
use crate::core::telemetry::Telemetry;
use crate::core::validation::ValidationResult;
use crate::core::{Executable, Executable::Async, Executable::Sync, Executable::Sealed};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The logic that is specific to orchestration of async nodes.
#[derive(Clone)]
pub struct AsyncFlowLogic {
    start: Executable,
    telemetry: Option<Arc<dyn Telemetry>>,
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
        AsyncFlow(AsyncNode::new(AsyncFlowLogic { start, telemetry: None }))
    }

    pub async fn run(&self, shared: &mut HashMap<String, NodeValue>) -> Option<String> {
        self.run_with_telemetry(shared, None).await
    }

    /// Executes the workflow and records telemetry if a logger is provided.
    pub async fn run_with_telemetry(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        telemetry: Option<Arc<dyn Telemetry>>,
    ) -> Option<String> {
        let mut cloned_self = self.clone();
        if let Some(logic) = cloned_self.behaviour.as_any_mut().downcast_mut::<AsyncFlowLogic>() {
            logic.telemetry = telemetry.clone();
        }
        
        cloned_self.0.run_with_telemetry(shared, telemetry.as_deref().map(|t| t as &dyn Telemetry)).await
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

    /// Validates the data flow integrity of the entire workflow (Async).
    pub fn validate(&self, initial_keys: Vec<String>) -> ValidationResult {
        let mut result = ValidationResult::new();
        let mut visited = HashSet::new();
        let mut available_keys = initial_keys.into_iter().collect::<HashSet<_>>();

        let behaviour: &dyn AsyncNodeLogic = &*self.behaviour;
        if let Some(flow_logic) = behaviour.as_any().downcast_ref::<AsyncFlowLogic>() {
            self.validate_recursive(
                &flow_logic.start,
                &mut available_keys,
                &mut visited,
                &mut result,
            );
        }

        result
    }

    fn validate_recursive(
        &self,
        current: &Executable,
        available_keys: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        result: &mut ValidationResult,
    ) {
        let sealable = match current {
            Executable::Sync(node) => node.behaviour.as_sealable(),
            Executable::Async(node) => node.behaviour.as_sealable(),
            Executable::Sealed(sealed) => Some(sealed.as_ref() as &dyn crate::core::semantic::Sealable),
        };

        if let Some(s) = sealable {
            let signature = s.signature();
            let task_id = s.task_id();

            if visited.contains(&task_id) {
                return;
            }
            visited.insert(task_id.clone());

            for input in signature.inputs {
                if !available_keys.contains(&input.name) {
                    result.add_error(format!(
                        "Node '{}' requires input '{}' which is missing from the shared state.",
                        task_id, input.name
                    ));
                }
            }

            for output in signature.outputs {
                available_keys.insert(output.name);
            }
        }

        for successor in current.successors().values() {
            let mut branch_keys = available_keys.clone();
            self.validate_recursive(successor, &mut branch_keys, visited, result);
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
        let (params, mut shared): (HashMap<String, NodeValue>, HashMap<String, NodeValue>) =
            if let Some(arr) = input.as_array() {
                if arr.len() != 2 {
                    log::error!("serde_json::to_value() failed to convert the params and shared.");
                    (HashMap::new(), HashMap::new())
                } else {
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
                    let telemetry_ref = self.telemetry.clone();

                    match tokio::task::spawn_blocking(move || {
                        let action = sync_clone
                            .run_with_telemetry(&mut shared_clone, telemetry_ref.as_deref().map(|t| t as &dyn Telemetry))
                            .unwrap_or("default".into());
                        (action, shared_clone)
                    })
                    .await
                    {
                        Ok((next_action, modified_shared)) => {
                            shared = modified_shared;
                            next_action
                        }
                        Err(join_error) => {
                            log::error!("A synchronous node panicked: {:?}", join_error);
                            "default".into()
                        }
                    }
                }
                Async(ref mut async_node) => {
                    async_node.set_params(params.clone());
                    async_node
                        .run_with_telemetry(&mut shared, self.telemetry.as_deref())
                        .await
                        .unwrap_or("default".into())
                }
                Sealed(ref sealed_node) => {
                    sealed_node.run(&mut shared, self.telemetry.as_deref())
                        .await
                        .unwrap_or("default".into())
                }
            };

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
                let last_action = serde_json::from_value(array[0].clone()).unwrap_or_default();
                let shared = serde_json::from_value(array[1].clone()).unwrap_or_default();
                (last_action, shared)
            }
        } else {
            log::error!("Serialization of last action and shared failed! Returning default values");
            ("default".into(), shared.clone())
        };

        *shared = shared_post;
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
}
