//! Integration test for the basic_flow example
//!
//! This test ensures that the example in `examples/basic_flow.rs` works correctly.

use orichalcum::prelude::*;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn test_basic_flow_example_logic() {
    // Recreate the logic from the example to test it in isolation

    #[derive(Clone)]
    struct ValidateInputLogic;

    impl NodeLogic for ValidateInputLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            shared.get("user_name").cloned().unwrap_or(NodeValue::Null)
        }

        fn exec(&self, input: NodeValue) -> NodeValue {
            if input.is_null() || input.as_str().map_or(true, |s| s.is_empty()) {
                "Guest".into()
            } else {
                input
            }
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            _prep_res: NodeValue,
            exec_res: NodeValue,
        ) -> Option<String> {
            shared.insert("user_name".to_string(), exec_res);
            Some("default".to_string())
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct GenerateGreetingLogic;

    impl NodeLogic for GenerateGreetingLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            shared.get("user_name").cloned().unwrap_or("Unknown".into())
        }

        fn exec(&self, input: NodeValue) -> NodeValue {
            let name = input.as_str().unwrap_or("Unknown");
            let greeting = format!("Hello, {}! Welcome to Orichalcum.", name);
            greeting.into()
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            _prep_res: NodeValue,
            exec_res: NodeValue,
        ) -> Option<String> {
            shared.insert("greeting".to_string(), exec_res.clone());
            Some("default".to_string())
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct AnalyzeSentimentLogic;

    impl NodeLogic for AnalyzeSentimentLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            shared.get("user_name").cloned().unwrap_or("Guest".into())
        }

        fn exec(&self, input: NodeValue) -> NodeValue {
            let name = input.as_str().unwrap_or("Guest");
            let sentiment = if name == "Guest" {
                "neutral"
            } else {
                "positive"
            };
            sentiment.into()
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            _prep_res: NodeValue,
            exec_res: NodeValue,
        ) -> Option<String> {
            let sentiment = exec_res.as_str().unwrap_or("neutral");
            shared.insert("sentiment".to_string(), exec_res.clone());
            Some(sentiment.to_string())
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct PositiveResponseLogic;

    impl NodeLogic for PositiveResponseLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            _shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            NodeValue::Null
        }

        fn exec(&self, _input: NodeValue) -> NodeValue {
            "We're glad to have you here! 🎉".into()
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            _prep_res: NodeValue,
            exec_res: NodeValue,
        ) -> Option<String> {
            shared.insert("response".to_string(), exec_res);
            None
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct NeutralResponseLogic;

    impl NodeLogic for NeutralResponseLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            _shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            NodeValue::Null
        }

        fn exec(&self, _input: NodeValue) -> NodeValue {
            "Feel free to explore and let us know if you need anything!".into()
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            _prep_res: NodeValue,
            exec_res: NodeValue,
        ) -> Option<String> {
            shared.insert("response".to_string(), exec_res);
            None
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    // Test with custom name
    {
        let validate_node = Node::new(ValidateInputLogic);
        let greeting_node = Node::new(GenerateGreetingLogic);
        let sentiment_node = Node::new(AnalyzeSentimentLogic);
        let positive_node = Node::new(PositiveResponseLogic);
        let neutral_node = Node::new(NeutralResponseLogic);

        let sentiment_with_branches = sentiment_node
            .next_on("positive", Executable::Sync(positive_node))
            .next_on("neutral", Executable::Sync(neutral_node));

        let greeting_node_with_sentiment =
            greeting_node.next(Executable::Sync(sentiment_with_branches));

        let start_node = validate_node.next(Executable::Sync(greeting_node_with_sentiment));

        let flow = Flow::new(start_node);
        let mut state = HashMap::new();
        state.insert("user_name".to_string(), "Alice".into());

        flow.run(&mut state);

        // Verify results
        assert_eq!(state.get("user_name"), Some(&json!("Alice")));
        assert!(state
            .get("greeting")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("Alice"));
        assert_eq!(state.get("sentiment"), Some(&json!("positive")));
        assert!(state
            .get("response")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("glad"));
    }

    // Test with default name
    {
        let validate_node = Node::new(ValidateInputLogic);
        let greeting_node = Node::new(GenerateGreetingLogic);
        let sentiment_node = Node::new(AnalyzeSentimentLogic);
        let positive_node = Node::new(PositiveResponseLogic);
        let neutral_node = Node::new(NeutralResponseLogic);

        let sentiment_with_branches = sentiment_node
            .next_on("positive", Executable::Sync(positive_node))
            .next_on("neutral", Executable::Sync(neutral_node));

        let greeting_node_with_sentiment =
            greeting_node.next(Executable::Sync(sentiment_with_branches));

        let start_node = validate_node.next(Executable::Sync(greeting_node_with_sentiment));

        let flow = Flow::new(start_node);
        let mut state = HashMap::new();

        flow.run(&mut state);

        // Verify results
        assert_eq!(state.get("user_name"), Some(&json!("Guest")));
        assert!(state
            .get("greeting")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("Guest"));
        assert_eq!(state.get("sentiment"), Some(&json!("neutral")));
        assert!(state
            .get("response")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("explore"));
    }
}
