use crate::core::sync_impl::node::{Node, NodeLogic};
use crate::core::sync_impl::NodeValue;
use crate::core::telemetry::Telemetry;
use crate::core::validation::ValidationResult;
use crate::core::Executable;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The logic that is specific to orchestration of nodes.
#[derive(Clone)]
pub struct FlowLogic {
    start: Node,
    telemetry: Option<Arc<dyn Telemetry>>,
}

/// A flow really, just is a Node with orchestration logic
/// to enforce that, we will create a NewType with a "factory" which prebuilds it.
#[derive(Clone)]
pub struct Flow(Node);

/// The Derefs are needed to be able to access the inside `Node` of the `Flow` easily
impl std::ops::Deref for Flow {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Flow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Flow {
    pub fn new(start: Node) -> Flow {
        Flow(Node::new(FlowLogic {
            start,
            telemetry: None,
        }))
    }

    pub fn run(&self, shared: &mut HashMap<String, NodeValue>) -> Option<String> {
        self.run_with_telemetry(shared, None)
    }

    /// Executes the workflow and records telemetry if a logger is provided.
    pub fn run_with_telemetry(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        telemetry: Option<Arc<dyn Telemetry>>,
    ) -> Option<String> {
        let mut cloned_self = self.clone();
        if let Some(logic) = cloned_self
            .behaviour
            .as_any_mut()
            .downcast_mut::<FlowLogic>()
        {
            logic.telemetry = telemetry.clone();
        }

        cloned_self
            .0
            .run_with_telemetry(shared, telemetry.as_deref().map(|t| t as &dyn Telemetry))
    }

    pub fn start(&mut self, start: Node) {
        // extract the `NodeLogic` from the Flow
        let behaviour: &mut dyn NodeLogic = &mut *self.behaviour;

        if let Some(flow_logic) = behaviour.as_any_mut().downcast_mut::<FlowLogic>() {
            // Should always be possible if the Flow as created through the factory
            flow_logic.start = start;
        } else {
            // This should never happen, but somehow it did
            panic!("Error: Flow's logic is not of type FlowLogic");
        }
    }

    /// Validates the data flow integrity of the entire workflow.
    pub fn validate(&self, initial_keys: Vec<String>) -> ValidationResult {
        let mut result = ValidationResult::new();
        let mut visited = HashSet::new();
        let mut available_keys = initial_keys.into_iter().collect::<HashSet<_>>();

        let behaviour: &dyn NodeLogic = &*self.behaviour;
        if let Some(flow_logic) = behaviour.as_any().downcast_ref::<FlowLogic>() {
            self.validate_recursive(
                &Executable::Sync(flow_logic.start.clone()),
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
            Executable::Sealed(sealed) => {
                Some(sealed.as_ref() as &dyn crate::core::semantic::Sealable)
            }
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

impl NodeLogic for FlowLogic {
    fn prep(
        &self,
        params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        serde_json::to_value((params, shared)).expect("If this works, I'll be so lit")
    }

    fn exec(&self, input: NodeValue) -> NodeValue {
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
        let mut current: Option<Node> = Some(self.start.clone());
        let mut last_action: String = "".into();

        // This is the orchestration logic
        while let Some(mut curr) = current {
            curr.set_params(params.clone());
            last_action = curr
                .run_with_telemetry(
                    &mut shared,
                    self.telemetry.as_deref().map(|t| t as &dyn Telemetry),
                )
                .unwrap_or("default".into());
            let next_executable = curr.data.successors.get(&last_action).cloned();

            match next_executable {
                Some(Executable::Sync(sync_node)) => current = Some(sync_node),
                Some(Executable::Async(_)) => {
                    panic!(
                        "Flow cannot handle AsyncNode, if you require to use regular Nodes with AsyncNodes, please use AsyncNode."
                    );
                }
                Some(Executable::Sealed(_sealed_node)) => {
                    // This is a bit complex for sync flow to handle sealed nodes that might be async
                    // For now, let's assume if it's in a Sync flow, it must be sync-wrapped or handled.
                    // But our SealedNode::run is async.

                    // Let's implement a block_on or restrict Sealed nodes in Sync flow for now.
                    log::error!("Sync Flow encountered a SealedNode. Handling of SealedNodes in Sync Flows is currently restricted.");
                    current = None;
                }
                None => {
                    current = None;
                }
            }
        }
        serde_json::to_value((last_action.to_string(), shared))
            .expect("Serializing string and HashMap should be doable")
    }
    fn post(
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
    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new((*self).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sync_impl::node::NodeLogic;
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct SimpleLogic {
        id: String,
        next_action: Option<String>,
    }

    impl NodeLogic for SimpleLogic {
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

    #[test]
    fn test_flow_creation() {
        let node = Node::new(SimpleLogic {
            id: "test".to_string(),
            next_action: None,
        });
        let flow = Flow::new(node);
        assert!(flow.data.params.is_empty());
    }

    #[test]
    fn test_flow_single_node() {
        let node = Node::new(SimpleLogic {
            id: "single".to_string(),
            next_action: None,
        });
        let flow = Flow::new(node);
        let mut shared = HashMap::new();
        let action = flow.run(&mut shared);
        assert_eq!(shared.get("visited_single"), Some(&json!(true)));
        assert_eq!(action, Some("default".to_string()));
    }
}
