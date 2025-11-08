//! A complete example showing how to build a multi-step workflow with Orichalcum.
//!
//! This example demonstrates:
//! - Creating custom node logic
//! - Chaining nodes together
//! - Using the shared state to pass data between nodes
//! - Conditional branching based on node outputs
//! - Building a Flow that orchestrates everything

use orichalcum::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Step 1: Input Validation Node
// ============================================================================

/// This node validates that we have a user name in the shared state.
/// If not, it adds a default one.
#[derive(Clone)]
struct ValidateInputLogic;

impl NodeLogic for ValidateInputLogic {
    fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        // Extract the name from shared state, if it exists
        shared.get("user_name").cloned().unwrap_or(NodeValue::Null)
    }

    fn exec(&self, input: NodeValue) -> NodeValue {
        // Check if we have a valid name
        if input.is_null() || input.as_str().map_or(true, |s| s.is_empty()) {
            // Return a default name
            "Guest".into()
        } else {
            // Keep the existing name
            input
        }
    }

    fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        // Store the validated name back in shared state
        shared.insert("user_name".to_string(), exec_res);
        println!("[ValidateInput] Name validated and stored");

        // Proceed to the next node via the "default" path
        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 2: Greeting Generator Node
// ============================================================================

/// This node creates a personalized greeting message.
#[derive(Clone)]
struct GenerateGreetingLogic;

impl NodeLogic for GenerateGreetingLogic {
    fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        // Get the validated name
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
        // Store the greeting
        shared.insert("greeting".to_string(), exec_res.clone());

        let greeting = exec_res.as_str().unwrap_or("");
        println!("[GenerateGreeting] {}", greeting);

        // Proceed to the sentiment analysis
        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 3: Sentiment Analysis Node (with branching)
// ============================================================================

/// This node analyzes the sentiment and branches based on the result.
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

        // Simple sentiment analysis: if name is "Guest", sentiment is neutral
        // Otherwise, it's positive
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

        println!("[AnalyzeSentiment] Sentiment: {}", sentiment);

        // Branch based on sentiment
        // This returns different action strings that will route to different nodes
        Some(sentiment.to_string())
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 4a: Positive Response Node
// ============================================================================

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
        println!("[PositiveResponse] {}", exec_res.as_str().unwrap());
        shared.insert("response".to_string(), exec_res);

        // This is a terminal node, so return None to end the flow
        None
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 4b: Neutral Response Node
// ============================================================================

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
        println!("[NeutralResponse] {}", exec_res.as_str().unwrap());
        shared.insert("response".to_string(), exec_res);

        // Terminal node
        None
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Main: Build and Execute the Flow
// ============================================================================

fn main() {
    println!("=== Orichalcum Basic Flow Example ===\n");

    // Create all our nodes
    let validate_node = Node::new(ValidateInputLogic);
    let greeting_node = Node::new(GenerateGreetingLogic);
    let sentiment_node = Node::new(AnalyzeSentimentLogic);
    let positive_node = Node::new(PositiveResponseLogic);
    let neutral_node = Node::new(NeutralResponseLogic);

    // Build the workflow by chaining nodes
    // sentiment_node branches to two different nodes based on its output
    let sentiment_with_branches = sentiment_node
        .next_on("positive", Executable::Sync(positive_node))
        .next_on("neutral", Executable::Sync(neutral_node));

    // Chain the main flow
    let start_node = validate_node
        .next(Executable::Sync(greeting_node))
        .next(Executable::Sync(sentiment_with_branches));

    // Create the flow
    let flow = Flow::new(start_node);

    // --- Example 1: Run with a custom name ---
    println!("\n--- Example 1: With custom name ---");
    let mut state1 = HashMap::new();
    state1.insert("user_name".to_string(), "Alice".into());
    flow.run(&mut state1);

    println!("\nFinal state:");
    for (key, value) in &state1 {
        println!("  {}: {:?}", key, value);
    }

    // --- Example 2: Run with no name (will use default) ---
    println!("\n\n--- Example 2: With default name ---");
    let mut state2 = HashMap::new();
    flow.run(&mut state2);

    println!("\nFinal state:");
    for (key, value) in &state2 {
        println!("  {}: {:?}", key, value);
    }

    println!("\n=== Flow completed successfully! ===");
}
