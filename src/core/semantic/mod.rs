pub mod node;
pub mod registry;
pub mod signature;

use crate::core::sync_impl::AsAny;
use signature::Signature;

/// Trait for units that have a defined structural contract and global identity.
/// Implementing this allows a node or flow to be validated and optimized.
pub trait Sealable: AsAny + Send + Sync {
    /// Returns the semantic signature (input/output contract) of this unit.
    fn signature(&self) -> Signature;

    /// Returns a unique identifier for this specific task instance.
    fn task_id(&self) -> String;
}

/// Trait for units whose behavior is driven by a natural language instruction and a model.
pub trait Promptable: Send + Sync {
    /// Returns the system instruction/prompt currently used by this unit.
    fn instruction(&self) -> Option<&str>;

    /// Returns the name of the model bound to this unit.
    fn model(&self) -> Option<&str>;
}
