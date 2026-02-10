# Orichalcum: An Agent Orchestration Framework in Rust

**License**: [MIT](LICENSE) | **Crates.io**: [v0.4.0](https://crates.io/crates/orichalcum) | **Docs**: [docs.rs](https://docs.rs/orichalcum)

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
*   **Semantic Layer (v0.4.0)**: Define structural contracts for your nodes using `Signature`. This allows for compile-time or runtime validation of your workflows.

## Installation

Add Orichalcum to your project's `Cargo.toml`:

```toml
[dependencies]
orichalcum = "0.4.0"

# For LLM features (Ollama, Gemini, DeepSeek)
# orichalcum = { version = "0.4.0", features = ["llm"] }

# For Telemetry features (tracing, optimization registry)
# orichalcum = { version = "0.4.0", features = ["telemetry"] }
```

## Quick Start: Semantic LLM Nodes (v0.4.0)

The most powerful way to use Orichalcum is via **Semantic Nodes**. These nodes have defined input/output contracts and are "sealed" for production stability.

```rust
use orichalcum::prelude::*;

#[tokio::main]
async fn main() {
    // 1. Initialize an LLM client (requires "llm" feature)
    let client = Client::with_ollama();

    // 2. Define a semantic signature
    let signature = signature!("document -> summary, sentiment");

    // 3. Build a semantic node
    let node = client.semantic_node()
        .signature(signature)
        .instruction("Summarize the document and analyze its sentiment.")
        .task_id("doc_processor_v1")
        .seal(); // Returns a SealedNode (wrapped in Executable)

    // 4. Run it in a flow
    let mut flow = AsyncFlow::new_from_executable(node);
    let mut state = HashMap::new();
    state.insert("document".to_string(), "Rust is a multi-paradigm, general-purpose programming language...".into());

    flow.run(&mut state).await;

    println!("Summary: {}", state.get("summary").unwrap());
    println!("Sentiment: {}", state.get("sentiment").unwrap());
}
```

## Traditional Example: A Simple Sync Flow

Orichalcum still supports pure Rust logic nodes for local processing.

```rust
use orichalcum::prelude::*;

#[derive(Clone)]
struct AddNameLogic;

impl NodeLogic for AddNameLogic {
    fn post(&self, shared: &mut HashMap<String, NodeValue>, _prep: NodeValue, _exec: NodeValue) -> Option<String> {
        shared.insert("name".to_string(), "Orichalcum".into());
        Some("default".to_string())
    }
    fn clone_box(&self) -> Box<dyn NodeLogic> { Box::new(self.clone()) }
}

#[derive(Clone)]
struct GreetLogic;

impl NodeLogic for GreetLogic {
    fn post(&self, shared: &mut HashMap<String, NodeValue>, _prep: NodeValue, _exec: NodeValue) -> Option<String> {
        if let Some(name) = shared.get("name").and_then(|v| v.as_str()) {
            println!("Hello, {}!", name);
        }
        None
    }
    fn clone_box(&self) -> Box<dyn NodeLogic> { Box::new(self.clone()) }
}

fn main() {
    let start_node = Node::new(AddNameLogic).next(Executable::Sync(Node::new(GreetLogic)));
    let flow = Flow::new(start_node);
    let mut state = HashMap::new();
    flow.run(&mut state);
}
```

## Features

*   **Semantic Layer**: Define I/O contracts with `Signature` for brutally-safe data flow.
*   **Telemetry (v0.4.0)**: Built-in tracing for I/O, model names, and execution timestamps.
*   **Unified LLM Builders**: Fluent API for `Gemini`, `DeepSeek`, and `Ollama`.
*   **Async & Parallel**: First-class support for `tokio` and parallel batch processing.
*   **Nix Support**: Includes `flake.nix` for a reproducible development environment.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
