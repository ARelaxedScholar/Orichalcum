pub mod async_impl;
pub mod sync_impl;

use async_impl::async_node::AsyncNode;
use std::collections::HashMap;
use sync_impl::node::Node;

/// The General Executable Enum
#[derive(Clone)]
pub enum Executable {
    Sync(Node),
    Async(AsyncNode),
}

impl Executable {
    pub fn successors(&self) -> &HashMap<String, Executable> {
        match self {
            // In this arm, `node` is a `&Node`
            Executable::Sync(node) => &node.data.successors,

            // In this arm, `node` is an `&AsyncNode`
            Executable::Async(node) => &node.data.successors,
        }
    }
}
