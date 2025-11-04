# Orichalcum: An Agent Orchestration Framework in Rust

**License**: [MIT](LICENSE) | **Crates.io**: [v0.2.5](https://crates.io/crates/orichalcum) | **Docs**: [docs.rs](https://docs.rs/orichalcum)

A brutally-safe, composable agent orchestration framework for building complex, multi-step workflows.

## Status

⚠️ **This library is in early development (0.x). The API is unstable and may change.**

## What is Orichalcum?

You've looked at LLM agent frameworks and thought, "This is neat. But is it *memory-safe*?"

You crave the sweet agony of the borrow checker. You yearn for the moral superiority that comes with writing everything in Rust. You, my friend, are a true masochist. And this is the LLM framework for you.

Orichalcum is a spiritual successor to Python's [PocketFlow](https://github.com/r-portas/PocketFlow), inheriting its philosophy of extreme composability. It allows you to define complex workflows (or "Flows") by chaining together simple, reusable components ("Nodes"). Each Node is a self-contained unit of work that can read from and write to a shared state, making decisions about what Node to execute next.

### Core Concepts

*   **Node**: The fundamental unit of work. A `Node` encapsulates a piece of logic with three steps: `prep` (prepare inputs), `exec` (execute the core logic), and `post` (process results and update state).
*   **Flow**: A special `Node` that orchestrates a graph of other `Node`s. It manages the execution sequence based on the outputs of each `Node`.
*   **Shared State**: A `HashMap` that is passed through the entire `Flow`. Nodes can read from this state to get context and write to it to pass results to subsequent nodes.
*   **Composition over Inheritance**: Instead of inheriting from base classes, you build complex functionality by wrapping nodes in other nodes. For example, `BatchNode` takes a `Node` and applies its logic to a list of items.

## Installation

Add Orichalcum to your project's `Cargo.toml`:

```toml
[dependencies]
orichalcum = "0.2.5"

# For LLM features (e.g., Ollama client)
# orichalcum = { version = "0.2.5", features = ["llm"] }
```

## Get Started: A Simple Example

Here's a complete example of a synchronous flow with two nodes. The first node adds a name to the shared state, and the second node greets that name.

```rust
use orichalcum::core::sync_impl::{
    flow::Flow,
    node::{Node, NodeLogic},
    NodeValue,
};
use std::collections::HashMap;

// --- Define the logic for our first node ---
#[derive(Clone)]
struct AddNameLogic;

impl NodeLogic for AddNameLogic {
    // `prep` is a required method for the trait.
    // Here we're just returning a default value.
    fn prep(&self, _params: &HashMap<String, NodeValue>, _shared: &HashMap<String, NodeValue>) -> NodeValue {
        NodeValue::Null
    }

    // `exec` is also required.
    fn exec(&self, _input: NodeValue) -> NodeValue {
        NodeValue::Null
    }
    
    // In the `post` step, we modify the shared state.
    fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        _exec_res: NodeValue,
    ) -> Option<String> {
        shared.insert("name".to_string(), "Orichalcum".into());
        println!("(Node 1) Added name to shared state.");
        // Return "default" to proceed to the next node connected via the default path.
        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
}

// --- Define the logic for our second node ---
#[derive(Clone)]
struct GreetLogic;

impl NodeLogic for GreetLogic {
    fn prep(&self, _params: &HashMap<String, NodeValue>, _shared: &HashMap<String, NodeValue>) -> NodeValue {
        NodeValue::Null
    }

    fn exec(&self, _input: NodeValue) -> NodeValue {
        NodeValue::Null
    }

    // This node reads from the state that the first node set.
    fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        _exec_res: NodeValue,
    ) -> Option<String> {
        if let Some(name) = shared.get("name").and_then(|v| v.as_str()) {
            println!("(Node 2) Hello, {}!", name);
        } else {
            println!("(Node 2) Hello, world!");
        }
        // This is the last node, so we return None to terminate the flow.
        None
    }
    
    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
}

fn main() {
    // 1. Create instances of our nodes.
    let add_name_node = Node::new(AddNameLogic);
    let greet_node = Node::new(GreetLogic);

    // 2. Chain them together. `add_name_node` runs first, and on its "default"
    //    action, it transitions to `greet_node`.
    //    Note: We must wrap Nodes in the `Executable` enum to add them as successors.
    let start_node = add_name_node.next(orichalcum::core::Executable::Sync(greet_node));

    // 3. Create a Flow that starts with our first node.
    let mut flow = Flow::new(start_node);

    // 4. Initialize the shared state and run the flow.
    let mut shared_state = HashMap::new();
    flow.run(&mut shared_state);

    // Verify the final state
    assert_eq!(shared_state.get("name").unwrap(), "Orichalcum");
    println!("Flow finished!");
}
```

## Features

*   **Blazingly Fast?** Probably. It's written in Rust, so it *feels* faster.
*   **Memory-Safe Workflows:** Let the compiler be your first line of defense against runtime errors.
*   **Synchronous & Asynchronous Execution:** Full support for both `sync` and `async` nodes and flows, including parallel batch processing.
*   **Composable by Design:** No inheritance, just pure, unadulterated composition. Wrap nodes in other nodes for batching, parallelism, and more.
*   **Optional LLM Integrations:** Built-in support for providers like Ollama behind a feature flag, keeping the core light.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
