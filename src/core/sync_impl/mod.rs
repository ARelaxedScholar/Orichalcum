//! Synchronous implementation of the Orichalcum workflow engine.
//!
//! This module contains all synchronous types and traits for building
//! and executing workflows:
//! - [`Node`] and [`NodeLogic`] for defining individual workflow steps
//! - [`Flow`] for orchestrating multiple nodes
//! - [`BatchLogic`] and [`new_batch_node`] for batch processing
//! - [`NodeValue`] type alias for JSON values used in shared state

pub mod batch_flow;
pub mod batch_node;
pub mod flow;
pub mod node;

/// The Alias for serde_json::Value since I use it a lot
pub type NodeValue = serde_json::Value;

use std::any::Any;

/// A helper trait that just provides the `as_any` method.
/// Needed for convenient downcasting of `FlowLogic` among other things
/// (More here for separation of concerns, since I could have added that to `NodeLogic` directly
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
