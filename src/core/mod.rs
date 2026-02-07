pub mod async_impl;
pub mod sealed;
pub mod semantic;
pub mod sync_impl;
pub mod telemetry;
pub mod validation;

use async_impl::async_node::AsyncNode;
use sealed::SealedNode;
use std::collections::HashMap;
use std::sync::Arc;
use sync_impl::node::Node;
use sync_impl::NodeValue;
use telemetry::Telemetry;

/// The General Executable Enum
#[derive(Clone)]
pub enum Executable {
    Sync(Node),
    Async(AsyncNode),
    Sealed(Arc<SealedNode>),
}

impl Executable {
    pub fn successors(&self) -> &HashMap<String, Executable> {
        match self {
            Executable::Sync(node) => &node.data.successors,
            Executable::Async(node) => &node.data.successors,
            Executable::Sealed(sealed) => sealed.inner().successors(),
        }
    }

    pub async fn run_with_telemetry(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        telemetry: Option<&dyn Telemetry>,
    ) -> Option<String> {
        match self {
            Executable::Sync(node) => node.run_with_telemetry(shared, telemetry),
            Executable::Async(node) => node.run_with_telemetry(shared, telemetry).await,
            Executable::Sealed(sealed) => sealed.run(shared, telemetry).await,
        }
    }
}
