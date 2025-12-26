use crate::core::sync_impl::node::{Node, NodeLogic};
use crate::core::sync_impl::NodeValue;
use std::collections::HashMap;

/// A BatchFlow is a `Node` (so orchestrable) which runs
/// a `Flow` many times with different params.
/// Therefore, a `BatchFlow` must deref the `Node`
/// But it's actual exectution logic should be implemented
/// on a struct which carries the `Flow` to batch.
///
/// Damn, I might be a prophet. But yup, after reconsideration
/// everything up there holds true, except the `BatchFlowLogic` holds a `Node` (this was the most
/// straightforward way I could think to enable BatchFlow nesting)
pub struct BatchFlow(Node);

/// The Derefs are needed to be able to access the inside `Node` of the `Flow` easily
impl std::ops::Deref for BatchFlow {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for BatchFlow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone)]
pub struct BatchFlowLogic<F>
where
    F: Fn(&HashMap<String, NodeValue>, &HashMap<String, NodeValue>) -> NodeValue
        + Clone
        + Send
        + Sync
        + 'static,
{
    // We have node so that we may nest BatchFlow'self
    // Technically, you could BatchFlow a single node as well?
    // But it's not as helpful
    flow: Node,
    prep_fn: F,
}

impl<F> NodeLogic for BatchFlowLogic<F>
where
    F: Fn(&HashMap<String, NodeValue>, &HashMap<String, NodeValue>) -> NodeValue
        + Clone
        + Send
        + Sync
        + 'static,
{
    fn prep(
        &self,
        params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        // Call the user-defined closure
        serde_json::to_value((shared, (self.prep_fn)(params, shared)))
            .expect("Serialization of shared to thing should work")
    }

    fn exec(&self, input: NodeValue) -> NodeValue {
        if let Some(array) = input.as_array() {
            if array.len() != 2 {
                panic!("Well shit");
            }
            // Ok, we covered our bases now
            let mut shared: HashMap<String, NodeValue> =
                serde_json::from_value(array[0].clone()).unwrap_or_default();
            let params_array: Vec<HashMap<String, NodeValue>> =
                serde_json::from_value(array[1].clone()).unwrap_or_default();
            params_array.into_iter().for_each(|params| {
                let mut combined_params: HashMap<String, NodeValue> = params.clone();
                combined_params.extend(self.flow.data.params.clone());
                let mut flow = self.flow.clone();
                flow.set_params(combined_params);
                flow.run(&mut shared);
            });

            serde_json::to_value(shared).expect("Serialization of shared dictionary should work!")
        } else {
            panic!("Serialization failure occured");
        }
    }

    fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        if let Ok(shared_post) = serde_json::from_value(exec_res) {
            *shared = shared_post
        } else {
            log::error!(
                "A deserialization error occured in BatchFlow, will proceed with non-updated shared"
            );
        }
        // In PocketFlow they return the exec_res, but I think it's cleaner like this. If
        // you're not happy with this, you can also just implement your custom
        // BatchFlowLogic
        // (This allows basic chaining)
        Some("default".into())
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new((*self).clone())
    }
}

impl BatchFlow {
    pub fn new<F>(flow: Node, prep_fn: F) -> Self
    where
        F: Fn(&HashMap<String, NodeValue>, &HashMap<String, NodeValue>) -> NodeValue
            + Clone
            + Send
            + Sync
            + 'static,
    {
        BatchFlow(Node::new(BatchFlowLogic { flow, prep_fn }))
    }
}
