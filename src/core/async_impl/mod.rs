//! Asynchronous implementation of the Orichalcum workflow engine.
//!
//! This module contains all asynchronous types and traits for building
//! and executing workflows with async/await support:
//! - [`AsyncNode`] and [`AsyncNodeLogic`] for defining async workflow steps
//! - [`AsyncFlow`] for orchestrating mixed sync/async nodes
//! - [`AsyncBatchLogic`] and [`new_async_batch_node`] for async batch processing
//! - [`AsyncParallelBatchLogic`] and [`new_async_parallel_batch_node`] for parallel async batch processing

pub mod async_batch_node;
pub mod async_flow;
pub mod async_node;
pub mod async_parallel_batch_node;
