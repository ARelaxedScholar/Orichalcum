//! # Orichalcum
//!
//! A brutally-safe, composable agent orchestration framework for building complex,
//! multi-step workflows in Rust.
//!
//! ## Features
//!
//! - **Memory-Safe Workflows**: Let the compiler catch errors at compile time
//! - **Sync & Async Support**: Full support for both synchronous and asynchronous execution
//! - **Composable Design**: Build complex workflows by composing simple, reusable nodes
//! - **Optional LLM Integration**: Built-in support for LLM providers (feature-gated)
//! - **Pick-and-choose Philosophy**: I have a few ideas of things I could add, but everything shall always be feature-gated.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use orichalcum::prelude::*;
//! use std::collections::HashMap;
//!
//! // Define your node logic
//! #[derive(Clone)]
//! struct MyLogic;
//!
//! impl NodeLogic for MyLogic {
//!     fn prep(&self, _params: &HashMap<String, NodeValue>, _shared: &HashMap<String, NodeValue>) -> NodeValue {
//!         NodeValue::Null
//!     }
//!     
//!     fn exec(&self, _input: NodeValue) -> NodeValue {
//!         NodeValue::Null
//!     }
//!     
//!     fn post(&self, shared: &mut HashMap<String, NodeValue>, _prep: NodeValue, _exec: NodeValue) -> Option<String> {
//!         shared.insert("result".to_string(), "done".into());
//!         None
//!     }
//!     
//!     fn clone_box(&self) -> Box<dyn NodeLogic> {
//!         Box::new(self.clone())
//!     }
//! }
//!
//! // Create and run a simple flow
//! let node = Node::new(MyLogic);
//! let mut flow = Flow::new(node);
//! let mut state = HashMap::new();
//! flow.run(&mut state);
//! ```
//!
//! ## Module Organization
//!
//! - [`sync`]: Synchronous node and flow implementations
//! - [`async_impl`]: Asynchronous node and flow implementations  
//! - [`prelude`]: Commonly used types and traits (import with `use orichalcum::prelude::*`)
//! - [`sync_prelude`]: Only synchronous types (import with `use orichalcum::sync_prelude::*`)
//! - [`async_prelude`]: Only asynchronous types (import with `use orichalcum::async_prelude::*`)

// ============================================================================
// Core Module
// ============================================================================

mod core;

// ============================================================================
// Public Re-exports - Granular Imports
// ============================================================================

// Core types
pub use core::Executable;

// Synchronous implementations
pub use core::sync_impl::batch_flow::BatchFlow;
pub use core::sync_impl::batch_node::{new_batch_node, BatchLogic};
pub use core::sync_impl::flow::{Flow, FlowLogic};
pub use core::sync_impl::node::{Node, NodeCore, NodeLogic};
pub use core::sync_impl::NodeValue;

// Asynchronous implementations
pub use core::async_impl::async_batch_node::{new_async_batch_node, AsyncBatchLogic};
pub use core::async_impl::async_flow::{AsyncFlow, AsyncFlowLogic};
pub use core::async_impl::async_node::{AsyncNode, AsyncNodeLogic};
pub use core::async_impl::async_parallel_batch_node::{
    new_async_parallel_batch_node, AsyncParallelBatchLogic,
};

// ============================================================================
// Prelude Modules - Convenient Bulk Imports
// ============================================================================

/// The main prelude: imports everything you need for both sync and async workflows.
///
/// # Example
/// ```rust
/// use orichalcum::prelude::*;
/// ```
pub mod prelude {
    pub use super::{
        new_async_batch_node,
        new_async_parallel_batch_node,
        new_batch_node,
        AsyncBatchLogic,
        AsyncFlow,
        AsyncFlowLogic,
        // Async
        AsyncNode,
        AsyncNodeLogic,
        AsyncParallelBatchLogic,
        BatchFlow,

        BatchLogic,
        // Core
        Executable,
        Flow,
        FlowLogic,
        // Sync
        Node,
        NodeCore,
        NodeLogic,
        NodeValue,
    };
}

/// Prelude for synchronous-only workflows.
///
/// Use this when you only need synchronous execution and want to avoid
/// importing async types.
///
/// # Example
/// ```rust
/// use orichalcum::sync_prelude::*;
/// ```
pub mod sync_prelude {
    pub use super::{
        new_batch_node, BatchFlow, BatchLogic, Executable, Flow, FlowLogic, Node, NodeCore,
        NodeLogic, NodeValue,
    };
}

/// Prelude for asynchronous-only workflows.
///
/// Use this when you only need async execution and want to avoid
/// importing sync types.
///
/// # Example
/// ```rust
/// use orichalcum::async_prelude::*;
/// ```
pub mod async_prelude {
    pub use super::{
        new_async_batch_node, new_async_parallel_batch_node, AsyncBatchLogic, AsyncFlow,
        AsyncFlowLogic, AsyncNode, AsyncNodeLogic, AsyncParallelBatchLogic, Executable, NodeValue,
    };
}

// ============================================================================
// LLM Feature
// ============================================================================

#[cfg(feature = "llm")]
pub mod llm;

#[cfg(feature = "llm")]
pub use llm::{
    error::LLMError,
    ollama::{Ollama, OllamaResponse},
    Client,
};

// ============================================================================
// Re-export commonly used external types for convenience
// ============================================================================

pub use serde_json::Value as JsonValue;
pub use std::collections::HashMap;

// ============================================================================
// Library Metadata
// ============================================================================

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The name of this crate.
pub const NAME: &str = env!("CARGO_PKG_NAME");
