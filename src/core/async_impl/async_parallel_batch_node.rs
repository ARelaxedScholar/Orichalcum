use crate::core::async_impl::async_node::{AsyncNode, AsyncNodeLogic};
use crate::core::sync_impl::NodeValue;
use async_trait::async_trait;
use futures::stream::{FuturesOrdered, StreamExt};
use std::collections::HashMap;

const DEFAULT_MAX_CONCURRENCY: usize = 50;

#[derive(Clone)]
pub struct AsyncParallelBatchLogic<L: AsyncNodeLogic> {
    logic: L,
    max_concurrency: usize,
}

impl<L: AsyncNodeLogic> AsyncParallelBatchLogic<L> {
    pub fn new(logic: L) -> Self {
        AsyncParallelBatchLogic {
            logic,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
        }
    }

    pub fn with_concurrency(self, max_concurrency: usize) -> Self {
        assert!(
            max_concurrency > 0,
            "Max concurrency must be greater than 0"
        );
        AsyncParallelBatchLogic {
            logic: self.logic,
            max_concurrency,
        }
    }
}

#[async_trait]
impl<L: AsyncNodeLogic + Clone> AsyncNodeLogic for AsyncParallelBatchLogic<L> {
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
            let mut results: Vec<NodeValue> = Vec::new();

            let futures = arr.iter().map(|item| self.logic.exec(item.clone()));
            let mut futures_ordered : FuturesOrdered<_>= futures.collect();

            while let Some(result) = futures_ordered.next().await {
                results.push(result);
            }
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
pub fn new_async_parallel_batch_node<L: AsyncNodeLogic + Clone>(
    logic: AsyncParallelBatchLogic<L>,
) -> AsyncNode {
    AsyncNode::new(logic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct AsyncDelayLogic {
        delay_ms: u64,
    }

    #[async_trait]
    impl AsyncNodeLogic for AsyncDelayLogic {
        async fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            _shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            NodeValue::Null
        }

        async fn exec(&self, input: NodeValue) -> NodeValue {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
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
    async fn test_async_parallel_batch_logic_creation() {
        let logic = AsyncDelayLogic { delay_ms: 1 };
        let parallel_logic = AsyncParallelBatchLogic::new(logic);
        
        assert_eq!(parallel_logic.max_concurrency, DEFAULT_MAX_CONCURRENCY);
        
        let items = json!([1, 2, 3]);
        let result = parallel_logic.exec(items).await;
        
        assert!(result.is_array());
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        // All should be multiplied by 2
        assert!(arr.contains(&json!(2.0)));
        assert!(arr.contains(&json!(4.0)));
        assert!(arr.contains(&json!(6.0)));
    }

    #[tokio::test]
    async fn test_async_parallel_batch_logic_with_concurrency() {
        let logic = AsyncDelayLogic { delay_ms: 10 };
        let parallel_logic = AsyncParallelBatchLogic::new(logic)
            .with_concurrency(2);
        
        assert_eq!(parallel_logic.max_concurrency, 2);
        
        // Test that it still works
        let items = json!([1, 2]);
        let result = parallel_logic.exec(items).await;
        
        assert!(result.is_array());
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[tokio::test]
    #[should_panic(expected = "Max concurrency must be greater than 0")]
    async fn test_async_parallel_batch_logic_zero_concurrency_panics() {
        let logic = AsyncDelayLogic { delay_ms: 1 };
        let parallel_logic = AsyncParallelBatchLogic::new(logic);
        
        // This should panic
        let _ = parallel_logic.with_concurrency(0);
    }

    #[tokio::test]
    async fn test_async_parallel_batch_logic_with_non_array_input() {
        let logic = AsyncDelayLogic { delay_ms: 1 };
        let parallel_logic = AsyncParallelBatchLogic::new(logic);
        
        let result = parallel_logic.exec(json!("not an array")).await;
        assert!(result.is_null());
    }

    #[tokio::test]
    async fn test_async_parallel_batch_logic_passthrough() {
        #[derive(Clone)]
        struct TrackingAsyncLogic;
        
        #[async_trait]
        impl AsyncNodeLogic for TrackingAsyncLogic {
            async fn prep(
                &self,
                _params: &HashMap<String, NodeValue>,
                _shared: &HashMap<String, NodeValue>,
            ) -> NodeValue {
                json!("prep_marker")
            }
            
            async fn exec(&self, input: NodeValue) -> NodeValue {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
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
                Some("default".to_string())
            }
            
            fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
                Box::new(self.clone())
            }
        }
        
        let inner_logic = TrackingAsyncLogic;
        let parallel_logic = AsyncParallelBatchLogic::new(inner_logic.clone());
        let params = HashMap::new();
        let shared = HashMap::new();
        let mut shared_mut = HashMap::new();
        
        let prep_result = parallel_logic.prep(&params, &shared).await;
        assert_eq!(prep_result, json!("prep_marker"));
        
        let exec_result = parallel_logic.exec(json!([1, 2, 3])).await;
        assert!(exec_result.is_array());
        
        let post_result = parallel_logic.post(&mut shared_mut, prep_result, exec_result).await;
        assert_eq!(post_result, Some("default".to_string()));
        assert_eq!(shared_mut.get("prep_res"), Some(&json!("prep_marker")));
        assert!(shared_mut.get("exec_res").is_some());
    }

    #[tokio::test]
    async fn test_new_async_parallel_batch_node() {
        let logic = AsyncDelayLogic { delay_ms: 1 };
        let parallel_logic = AsyncParallelBatchLogic::new(logic);
        let batch_node = new_async_parallel_batch_node(parallel_logic);
        
        let mut shared = HashMap::new();
        let action = batch_node.run(&mut shared).await;
        assert_eq!(action, Some("default".to_string()));
    }
}
