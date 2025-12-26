use crate::core::async_impl::async_node::{AsyncNode, AsyncNodeLogic};
use crate::core::sync_impl::NodeValue;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct AsyncBatchLogic<L: AsyncNodeLogic> {
    logic: L,
}

impl<L: AsyncNodeLogic> AsyncBatchLogic<L> {
    pub fn new(logic: L) -> Self {
        AsyncBatchLogic { logic }
    }
}

#[async_trait]
impl<L: AsyncNodeLogic + Clone> AsyncNodeLogic for AsyncBatchLogic<L> {
    async fn prep(
        &self,
        params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        self.logic.prep(params, shared).await
    }

    async fn exec(&self, items: NodeValue) -> NodeValue {
        // Check that input is indeed an array
        if let Some(arr) = items.as_array() {
            let owned_items: Vec<NodeValue> = arr.iter().cloned().collect();
            let logic = Arc::new(self.logic.clone());
            let results: Vec<NodeValue> = stream::iter(owned_items)
                .then(move |item| {
                    let l = Arc::clone(&logic);
                    async move { l.exec(item).await }
                })
                .collect()
                .await;

            results.into()
        } else {
            log::error!("items is not an array");
            NodeValue::Null
        }
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        self.logic.post(shared, prep_res, exec_res).await
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new((*self).clone())
    }
}

/// The `AsyncBatchNode` factory
pub fn new_async_batch_node<L: AsyncNodeLogic + Clone>(logic: L) -> AsyncNode {
    AsyncNode::new(AsyncBatchLogic { logic })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct AsyncMultiplyLogic;

    #[async_trait]
    impl AsyncNodeLogic for AsyncMultiplyLogic {
        async fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            _shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            NodeValue::Null
        }

        async fn exec(&self, input: NodeValue) -> NodeValue {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            if let Some(num) = input.as_f64() {
                json!(num * 2.0)
            } else {
                input
            }
        }

        async fn post(
            &self,
            _shared: &mut HashMap<String, NodeValue>,
            _prep_res: NodeValue,
            _exec_res: NodeValue,
        ) -> Option<String> {
            Some("default".to_string())
        }

        fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn test_async_batch_logic_creation() {
        let logic = AsyncMultiplyLogic;
        let batch_logic = AsyncBatchLogic::new(logic);

        let items = json!([1, 2, 3]);
        let result = batch_logic.exec(items).await;

        assert!(result.is_array());
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], json!(2.0));
        assert_eq!(arr[1], json!(4.0));
        assert_eq!(arr[2], json!(6.0));
    }

    #[tokio::test]
    async fn test_async_batch_logic_with_non_array_input() {
        let logic = AsyncMultiplyLogic;
        let batch_logic = AsyncBatchLogic::new(logic);

        let result = batch_logic.exec(json!("not an array")).await;
        assert!(result.is_null());
    }

    #[tokio::test]
    async fn test_async_batch_logic_passthrough() {
        #[derive(Clone)]
        struct TrackingAsyncLogic {
            marker: String,
        }

        #[async_trait]
        impl AsyncNodeLogic for TrackingAsyncLogic {
            async fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                json!(self.marker.clone())
            }

            async fn exec(&self, input: NodeValue) -> NodeValue {
                input
            }

            async fn post(
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

            fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
                Box::new(self.clone())
            }
        }

        let inner_logic = TrackingAsyncLogic {
            marker: "test_marker".to_string(),
        };

        let batch_logic = AsyncBatchLogic::new(inner_logic.clone());
        let params = HashMap::new();
        let shared = HashMap::new();
        let mut shared_mut = HashMap::new();

        let prep_result = batch_logic.prep(&params, &shared).await;
        assert_eq!(prep_result, json!("test_marker"));

        let exec_result = batch_logic.exec(json!([1, 2, 3])).await;
        assert!(exec_result.is_array());

        let post_result = batch_logic
            .post(&mut shared_mut, prep_result, exec_result)
            .await;
        assert_eq!(post_result, Some("default".to_string()));
        assert_eq!(shared_mut.get("post_called"), Some(&json!(true)));
        assert!(shared_mut.get("prep_res").is_some());
        assert!(shared_mut.get("exec_res").is_some());
    }

    #[tokio::test]
    async fn test_new_async_batch_node() {
        let logic = AsyncMultiplyLogic;
        let batch_node = new_async_batch_node(logic);

        let mut shared = HashMap::new();
        let action = batch_node.run(&mut shared).await;
        assert_eq!(action, Some("default".to_string()));
    }
}
