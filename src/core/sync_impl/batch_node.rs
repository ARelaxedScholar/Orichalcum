use crate::core::sync_impl::NodeValue;
use crate::core::sync_impl::node::{Node, NodeLogic};
use std::collections::HashMap;

/// ------- BatchNode -------------------------------------------------------------
/// This logic is fairly easy to implement since the core logic is really just about taking
/// many items and applying the logic on all of them. But the more powerful approach here
/// is just have BatchNode be generic over NodeLogic, this way it is composable with `Node`
#[derive(Clone)]
pub struct BatchLogic<L: NodeLogic> {
    logic: L,
}

/// Convenience functions to create new BatchLogic (note that in our approach)
/// A `BatchLogic` and a `BatchNode` are not the same.
/// `BatchLogic` is simply a conceptual struct which marks what we'd want to be batched.
/// `BatchNode` which we define through the composition of a `Node` with a `NodeLogic` which is
/// `Clone`-able, is simply a `Node` which applies its logic to a bunch of items (sequentially.)
impl<L: NodeLogic> BatchLogic<L> {
    pub fn new(logic: L) -> Self {
        BatchLogic { logic }
    }
}

/// The advent of the BatchNode
/// Defining the logic for what is a `BatchLogic` which is a "true" `NodeLogic`.
impl<L: NodeLogic + Clone> NodeLogic for BatchLogic<L> {
    fn prep(
        &self,
        params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        self.logic.prep(params, shared)
    }

    fn exec(&self, items: NodeValue) -> NodeValue {
        // Check that input is indeed an array
        if let Some(arr) = items.as_array() {
            let results: Vec<NodeValue> = arr
                .iter()
                .map(|item| self.logic.exec(item.clone()))
                .collect();

            results.into()
        } else {
            log::error!("items is not an array");
            NodeValue::Null
        }
    }

    fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        self.logic.post(shared, prep_res, exec_res)
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new((*self).clone())
    }
}

/// The `BatchNode` factory
pub fn new_batch_node<L: NodeLogic + Clone>(logic: L) -> Node {
    Node::new(BatchLogic { logic })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Executable;
    use crate::core::sync_impl::flow::Flow;
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct MultiplyLogic;

    impl NodeLogic for MultiplyLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            _shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            NodeValue::Null
        }

        fn exec(&self, input: NodeValue) -> NodeValue {
            if let Some(num) = input.as_f64() {
                json!(num * 2.0)
            } else {
                input
            }
        }

        fn post(
            &self,
            _shared: &mut HashMap<String, NodeValue>,
            _prep_res: NodeValue,
            _exec_res: NodeValue,
        ) -> Option<String> {
            Some("default".to_string())
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_batch_logic_creation() {
        let logic = MultiplyLogic;
        let batch_logic = BatchLogic::new(logic);
        
        // BatchLogic should wrap the inner logic
        // We can test this by checking exec behavior
        let items = json!([1, 2, 3]);
        let result = batch_logic.exec(items);
        
        assert!(result.is_array());
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], json!(2.0));
        assert_eq!(arr[1], json!(4.0));
        assert_eq!(arr[2], json!(6.0));
    }

    #[test]
    fn test_batch_logic_with_non_array_input() {
        let logic = MultiplyLogic;
        let batch_logic = BatchLogic::new(logic);
        
        // Non-array input should return Null
        let result = batch_logic.exec(json!("not an array"));
        assert!(result.is_null());
    }

    #[test]
    fn test_batch_logic_passthrough() {
        // Test that prep and post are passed through to inner logic
        #[derive(Clone)]
        struct TrackingLogic {
            prep_called: bool,
            post_called: bool,
        }
        
        impl NodeLogic for TrackingLogic {
            fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                let mut new_self = self.clone();
                new_self.prep_called = true;
                // Store self in Box to track? Simpler: return a marker
                json!("prep_called")
            }
            
            fn exec(&self, input: NodeValue) -> NodeValue {
                input
            }
            
            fn post(
                &self,
                shared: &mut HashMap<String, NodeValue>,
                prep_res: NodeValue,
                exec_res: NodeValue,
            ) -> Option<String> {
                shared.insert("prep_res".to_string(), prep_res);
                shared.insert("exec_res".to_string(), exec_res);
                shared.insert("post_called".to_string(), json!(true));
                Some("default".to_string())
            }
            
            fn clone_box(&self) -> Box<dyn NodeLogic> {
                Box::new(self.clone())
            }
        }
        
        let inner_logic = TrackingLogic {
            prep_called: false,
            post_called: false,
        };
        
        let batch_logic = BatchLogic::new(inner_logic.clone());
        let params = HashMap::new();
        let shared = HashMap::new();
        let mut shared_mut = HashMap::new();
        
        let prep_result = batch_logic.prep(&params, &shared);
        assert_eq!(prep_result, json!("prep_called"));
        
        let exec_result = batch_logic.exec(json!([1, 2, 3]));
        assert!(exec_result.is_array());
        
        let post_result = batch_logic.post(&mut shared_mut, prep_result, exec_result);
        assert_eq!(post_result, Some("default".to_string()));
        assert_eq!(shared_mut.get("post_called"), Some(&json!(true)));
        assert!(shared_mut.get("prep_res").is_some());
        assert!(shared_mut.get("exec_res").is_some());
    }

    #[test]
    fn test_new_batch_node() {
        let logic = MultiplyLogic;
        let mut batch_node = new_batch_node(logic);
        
        // Should create a Node with BatchLogic inside
        let mut shared = HashMap::new();
        let items = json!([1, 2, 3]);
        let mut params = HashMap::new();
        params.insert("items".to_string(), items.clone());
        
        batch_node.set_params(params);
        
        // When we run the node, it will use prep/exec/post
        // Since our MultiplyLogic doesn't use params/shared in prep,
        // and returns Some("default") from post, we can test exec via run_with_params
        let action = batch_node.run_with_params(&mut shared, &HashMap::new());
        assert_eq!(action, Some("default".to_string()));
    }

    #[test]
    fn test_batch_node_in_flow() {
        // Test that a batch node can be used in a flow
        let batch_node = new_batch_node(MultiplyLogic);
        
        #[derive(Clone)]
        struct PrepareItemsLogic;
        
        impl NodeLogic for PrepareItemsLogic {
            fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                json!([1, 2, 3, 4, 5])
            }
            
            fn exec(&self, input: NodeValue) -> NodeValue {
                input
            }
            
            fn post(
                &self,
                shared: &mut HashMap<String, NodeValue>,
                prep_res: NodeValue,
                exec_res: NodeValue,
            ) -> Option<String> {
                shared.insert("items".to_string(), prep_res.clone());
                shared.insert("exec_input".to_string(), exec_res);
                Some("default".to_string())
            }
            
            fn clone_box(&self) -> Box<dyn NodeLogic> {
                Box::new(self.clone())
            }
        }
        
        let prepare_node = Node::new(PrepareItemsLogic)
            .next(Executable::Sync(batch_node));
        
        let flow = Flow::new(prepare_node);
        let mut shared = HashMap::new();
        
        let action = flow.run(&mut shared);
        
        // The batch node should have processed the items
        // We can check that items were stored (though batch node doesn't modify shared)
        assert_eq!(action, Some("default".to_string()));
    }
}
