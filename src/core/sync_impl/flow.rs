use crate::core::Executable;
use crate::core::sync_impl::NodeValue;
use crate::core::sync_impl::node::{Node, NodeLogic};
use std::collections::HashMap;

/// The logic that is specif
#[derive(Clone)]
pub struct FlowLogic {
    start: Node,
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
        Flow(Node::new(FlowLogic { start }))
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
        let mut current: Option<Node> = Some(self.start.clone());
        let mut last_action: String = "".into();

        // This is the orchestration logic
        while let Some(mut curr) = current {
            curr.set_params(params.clone());
            last_action = curr.run(&mut shared).unwrap_or("default".into());
            let next_executable = curr.data.successors.get(&last_action).cloned();

            match next_executable {
                Some(Executable::Sync(sync_node)) => current = Some(sync_node),
                Some(Executable::Async(_)) => {
                    panic!(
                        "Flow cannot handle AsyncNode, if you require to use regular Nodes with AsyncNodes, please use AsyncNode."
                    );
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
    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new((*self).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sync_impl::node::NodeLogic;
    use async_trait::async_trait;
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
            // Just pass through
            input
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            prep_res: NodeValue,
            _exec_res: NodeValue,
        ) -> Option<String> {
            // Store the node ID in shared state
            let id = prep_res.as_str().unwrap_or("unknown");
            shared.insert(format!("visited_{}", id), json!(true));
            
            // Return the next action or None to terminate
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
        
        // Flow should deref to Node
        assert!(flow.data.params.is_empty());
    }

    #[test]
    fn test_flow_single_node() {
        let node = Node::new(SimpleLogic {
            id: "single".to_string(),
            next_action: None, // Terminal node
        });
        let flow = Flow::new(node);
        let mut shared = HashMap::new();
        
        let action = flow.run(&mut shared);
        
        // Flow should have visited the single node
        assert_eq!(shared.get("visited_single"), Some(&json!(true)));
        // Flow's action should be from the node (None -> "default"? Actually flow returns Some("default")?)
        // The flow returns the last action which would be "default" since node returns None
        // and flow's post converts None to "default"
        assert_eq!(action, Some("default".to_string()));
    }

    #[test]
    fn test_flow_chain() {
        // Create a chain: node1 -> node2 -> node3
        let node3 = Node::new(SimpleLogic {
            id: "node3".to_string(),
            next_action: None,
        });
        
        let node2 = Node::new(SimpleLogic {
            id: "node2".to_string(),
            next_action: Some("default".to_string()),
        }).next(Executable::Sync(node3));
        
        let node1 = Node::new(SimpleLogic {
            id: "node1".to_string(),
            next_action: Some("default".to_string()),
        }).next(Executable::Sync(node2));
        
        let flow = Flow::new(node1);
        let mut shared = HashMap::new();
        
        let action = flow.run(&mut shared);
        
        // All nodes should have been visited
        assert_eq!(shared.get("visited_node1"), Some(&json!(true)));
        assert_eq!(shared.get("visited_node2"), Some(&json!(true)));
        assert_eq!(shared.get("visited_node3"), Some(&json!(true)));
        assert_eq!(action, Some("default".to_string()));
    }

    #[test]
    fn test_flow_branching() {
        // Create branching: node1 branches to either node2a or node2b
        let node2b = Node::new(SimpleLogic {
            id: "node2b".to_string(),
            next_action: None,
        });
        
        let node2a = Node::new(SimpleLogic {
            id: "node2a".to_string(),
            next_action: None,
        });
        
        #[derive(Clone)]
        struct BranchingLogic {
            branch_to: String,
        }
        
        impl NodeLogic for BranchingLogic {
            fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                json!(self.branch_to.clone())
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
                let branch = prep_res.as_str().unwrap_or("default");
                shared.insert("branch_taken".to_string(), json!(branch));
                Some(branch.to_string())
            }
            
            fn clone_box(&self) -> Box<dyn NodeLogic> {
                Box::new(self.clone())
            }
        }
        
        let node1 = Node::new(BranchingLogic {
            branch_to: "branch_a".to_string(),
        })
        .next_on("branch_a", Executable::Sync(node2a))
        .next_on("branch_b", Executable::Sync(node2b));
        
        let flow = Flow::new(node1);
        let mut shared = HashMap::new();
        
        let action = flow.run(&mut shared);
        
        // Should have taken branch_a
        assert_eq!(shared.get("branch_taken"), Some(&json!("branch_a")));
        assert_eq!(shared.get("visited_node2a"), Some(&json!(true)));
        assert!(shared.get("visited_node2b").is_none());
        assert_eq!(action, Some("default".to_string()));
    }

    #[test]
    fn test_flow_with_params() {
        #[derive(Clone)]
        struct ParamLogic;
        
        impl NodeLogic for ParamLogic {
            fn prep(
                &self,
                params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                params.get("test_param").cloned().unwrap_or(NodeValue::Null)
            }
            
            fn exec(&self, input: NodeValue) -> NodeValue {
                // Return the input value
                input
            }
            
            fn post(
                &self,
                shared: &mut HashMap<String, NodeValue>,
                prep_res: NodeValue,
                _exec_res: NodeValue,
            ) -> Option<String> {
                shared.insert("received_param".to_string(), prep_res);
                None
            }
            
            fn clone_box(&self) -> Box<dyn NodeLogic> {
                Box::new(self.clone())
            }
        }
        
        let node = Node::new(ParamLogic);
        let flow = Flow::new(node);
        let mut shared = HashMap::new();
        let mut params = HashMap::new();
        params.insert("test_param".to_string(), json!("test_value"));
        
        // Set params on the flow's internal node
        let mut flow_clone = flow.clone();
        flow_clone.set_params(params);
        
        let action = flow_clone.run(&mut shared);
        
        // Should have received the parameter
        assert_eq!(shared.get("received_param"), Some(&json!("test_value")));
        assert_eq!(action, Some("default".to_string()));
    }

    #[test]
    #[should_panic(expected = "Flow cannot handle AsyncNode")]
    fn test_flow_panics_on_async_node() {
        use crate::core::async_impl::async_node::AsyncNode;
        
        #[derive(Clone)]
        struct AsyncTestLogic;
        
        #[async_trait]
        impl crate::core::async_impl::async_node::AsyncNodeLogic for AsyncTestLogic {
            fn clone_box(&self) -> Box<dyn crate::core::async_impl::async_node::AsyncNodeLogic> {
                Box::new(self.clone())
            }
            
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
                Some("default".to_string())
            }
        }
        
        let async_node = AsyncNode::new(AsyncTestLogic);
        let node = Node::new(SimpleLogic {
            id: "test".to_string(),
            next_action: Some("default".to_string()),
        }).next(Executable::Async(async_node));
        
        let flow = Flow::new(node);
        let mut shared = HashMap::new();
        
        // This should panic when it encounters the async node
        flow.run(&mut shared);
    }
}
