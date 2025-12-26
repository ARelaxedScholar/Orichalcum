use crate::core::sync_impl::AsAny;
use crate::core::sync_impl::NodeValue;
use crate::core::Executable;
use std::collections::HashMap;

/// A node in a workflow graph.
///
/// A `Node` encapsulates a unit of work with three-phase execution:
/// 1. **Prep**: Prepare inputs from parameters and shared state
/// 2. **Exec**: Execute the core logic
/// 3. **Post**: Process results and update shared state, optionally returning the next action
///
/// Nodes can be chained together using [`next`](Node::next) or [`next_on`](Node::next_on)
/// to create complex workflows.
///
/// # Example
/// ```
/// use orichalcum::sync_prelude::*;
/// use std::collections::HashMap;
///
/// #[derive(Clone)]
/// struct GreetLogic;
///
/// impl NodeLogic for GreetLogic {
///     fn prep(&self, _params: &HashMap<String, NodeValue>, shared: &HashMap<String, NodeValue>) -> NodeValue {
///         shared.get("name").cloned().unwrap_or("World".into())
///     }
///     
///     fn exec(&self, input: NodeValue) -> NodeValue {
///         let name = input.as_str().unwrap_or("World");
///         format!("Hello, {}!", name).into()
///     }
///     
///     fn post(&self, shared: &mut HashMap<String, NodeValue>, _prep: NodeValue, exec: NodeValue) -> Option<String> {
///         shared.insert("greeting".to_string(), exec);
///         None
///     }
///     
///     fn clone_box(&self) -> Box<dyn NodeLogic> {
///         Box::new(self.clone())
///     }
/// }
///
/// let node = Node::new(GreetLogic);
/// let mut shared = HashMap::new();
/// shared.insert("name".to_string(), "Orichalcum".into());
/// 
/// node.run(&mut shared);
/// assert_eq!(shared.get("greeting").unwrap().as_str().unwrap(), "Hello, Orichalcum!");
/// ```
#[derive(Clone)]
pub struct Node {
    /// Internal node data including parameters and successors
    pub data: NodeCore,
    /// The logic implementation that defines the node's behavior
    pub behaviour: Box<dyn NodeLogic>,
}

impl Node {
    /// Creates a new node with the given logic.
    ///
    /// # Arguments
    /// * `behaviour` - An implementation of [`NodeLogic`] that defines the node's behavior
    pub fn new<L: NodeLogic + 'static>(behaviour: L) -> Self {
        Node {
            data: NodeCore::default(),
            behaviour: Box::new(behaviour),
        }
    }
    
    /// Sets the node's parameters.
    ///
    /// Parameters are node-specific configuration that can be accessed
    /// in the [`prep`](NodeLogic::prep) phase.
    pub fn set_params(&mut self, params: HashMap<String, NodeValue>) {
        self.data.params = params;
    }
    
    /// Chains another node to execute after this node via the "default" action.
    ///
    /// Equivalent to `self.next_on("default", node)`.
    pub fn next(self, node: Executable) -> Self {
        self.next_on("default", node)
    }
    
    /// Chains another node to execute after this node when the specified action is returned.
    ///
    /// If the node already has a successor for the given action, it will be overwritten
    /// with a warning logged.
    ///
    /// # Arguments
    /// * `action` - The action string that should trigger this successor
    /// * `node` - The node to execute when this action is returned from [`post`](NodeLogic::post)
    pub fn next_on(mut self, action: &str, node: Executable) -> Self {
        if self.data.successors.contains_key(action) {
            log::warn!(
                "Warning: Action {} was found in successors, Overwriting key {}.",
                &action,
                &action
            );
        }
        self.data.successors.insert(action.to_string(), node);
        self
    }

    /// Executes the node with its current parameters.
    ///
    /// Runs the three-phase execution (prep, exec, post) using the node's
    /// stored parameters.
    ///
    /// # Arguments
    /// * `shared` - Mutable reference to the shared state
    ///
    /// # Returns
    /// The action returned by [`post`](NodeLogic::post), or `None` if the workflow should terminate
    pub fn run(&self, shared: &mut HashMap<String, NodeValue>) -> Option<String> {
        let p = self.behaviour.prep(&self.data.params, shared);
        let e = self.behaviour.exec(p.clone());
        self.behaviour.post(shared, p, e)
    }

    /// Executes the node with the given parameters, ignoring stored parameters.
    ///
    /// Useful for one-off executions with different parameters without modifying
    /// the node's internal state.
    ///
    /// # Arguments
    /// * `shared` - Mutable reference to the shared state
    /// * `param` - Parameters to use for this execution
    pub fn run_with_params(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        param: &HashMap<String, NodeValue>,
    ) -> Option<String> {
        let p = self.behaviour.prep(param, shared);
        let e = self.behaviour.exec(p.clone());
        self.behaviour.post(shared, p, e)
    }
}

/// Internal data structure for a node.
///
/// Contains the node's parameters and successor mappings.
#[derive(Default, Clone)]
pub struct NodeCore {
    /// Parameters specific to this node instance
    pub params: HashMap<String, NodeValue>,
    /// Mapping from action strings to successor nodes
    pub successors: HashMap<String, Executable>,
}

/// Defines the behavior of a workflow node.
///
/// Implement this trait to create custom node logic. The trait provides
/// a three-phase execution model with default implementations for each phase.
pub trait NodeLogic: AsAny + Send + Sync + 'static {
    /// Prepare inputs for execution.
    ///
    /// This phase extracts necessary data from node parameters and shared state,
    /// returning a value that will be passed to [`exec`](NodeLogic::exec).
    ///
    /// # Arguments
    /// * `params` - Node-specific parameters
    /// * `shared` - Shared state accessible to all nodes in the workflow
    ///
    /// # Returns
    /// A [`NodeValue`] to be processed by [`exec`](NodeLogic::exec)
    fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        _shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        NodeValue::default()
    }
    
    /// Execute the core logic of the node.
    ///
    /// This phase performs the main computation or operation using the
    /// value returned from [`prep`](NodeLogic::prep).
    ///
    /// # Arguments
    /// * `input` - The value returned by [`prep`](NodeLogic::prep)
    ///
    /// # Returns
    /// A [`NodeValue`] representing the execution result
    fn exec(&self, _input: NodeValue) -> NodeValue {
        NodeValue::default()
    }
    
    /// Post-process results and update shared state.
    ///
    /// This final phase can store results in the shared state and
    /// determine which node should execute next by returning an action string.
    ///
    /// # Arguments
    /// * `shared` - Mutable reference to the shared state
    /// * `prep_res` - The value returned by [`prep`](NodeLogic::prep)
    /// * `exec_res` - The value returned by [`exec`](NodeLogic::exec)
    ///
    /// # Returns
    /// * `Some(action)` - Execute the successor mapped to this action
    /// * `None` - Terminate the workflow
    fn post(
        &self,
        _shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        _exec_res: NodeValue,
    ) -> Option<String> {
        None
    }

    /// Create a boxed clone of this trait object.
    ///
    /// Required for cloning `Box<dyn NodeLogic>`.
    fn clone_box(&self) -> Box<dyn NodeLogic>;
}

impl Clone for Box<dyn NodeLogic> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Clone)]
    struct TestLogic;

    impl NodeLogic for TestLogic {
        fn prep(
            &self,
            _params: &HashMap<String, NodeValue>,
            shared: &HashMap<String, NodeValue>,
        ) -> NodeValue {
            // Return a value from shared state or default
            shared.get("test").cloned().unwrap_or(NodeValue::Null)
        }

        fn exec(&self, input: NodeValue) -> NodeValue {
            // Double a number or return string
            if let Some(num) = input.as_f64() {
                json!(num * 2.0)
            } else {
                json!("processed")
            }
        }

        fn post(
            &self,
            shared: &mut HashMap<String, NodeValue>,
            prep_res: NodeValue,
            exec_res: NodeValue,
        ) -> Option<String> {
            // Store results in shared state
            shared.insert("prep".to_string(), prep_res);
            shared.insert("exec".to_string(), exec_res);
            // Return action to determine next node
            Some("default".to_string())
        }

        fn clone_box(&self) -> Box<dyn NodeLogic> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_node_creation() {
        let node = Node::new(TestLogic);
        assert!(node.data.params.is_empty());
        assert!(node.data.successors.is_empty());
    }

    #[test]
    fn test_node_set_params() {
        let mut node = Node::new(TestLogic);
        let mut params = HashMap::new();
        params.insert("key".to_string(), json!("value"));
        node.set_params(params.clone());
        assert_eq!(node.data.params, params);
    }

    #[test]
    fn test_node_next() {
        let node1 = Node::new(TestLogic);
        let node2 = Node::new(TestLogic);
        let node1_with_next = node1.next(Executable::Sync(node2));
        
        assert_eq!(node1_with_next.data.successors.len(), 1);
        assert!(node1_with_next.data.successors.contains_key("default"));
    }

    #[test]
    fn test_node_next_on() {
        let node1 = Node::new(TestLogic);
        let node2 = Node::new(TestLogic);
        let node1_with_next = node1.next_on("custom", Executable::Sync(node2));
        
        assert_eq!(node1_with_next.data.successors.len(), 1);
        assert!(node1_with_next.data.successors.contains_key("custom"));
    }

    #[test]
    fn test_node_run() {
        let node = Node::new(TestLogic);
        let mut shared = HashMap::new();
        shared.insert("test".to_string(), json!(42));
        
        let action = node.run(&mut shared);
        assert_eq!(action, Some("default".to_string()));
        assert_eq!(shared.get("prep"), Some(&json!(42)));
        assert_eq!(shared.get("exec"), Some(&json!(84.0)));
    }

    #[test]
    fn test_node_run_with_params() {
        let node = Node::new(TestLogic);
        let mut shared = HashMap::new();
        let mut params = HashMap::new();
        params.insert("param".to_string(), json!("value"));
        
        let action = node.run_with_params(&mut shared, &params);
        assert_eq!(action, Some("default".to_string()));
        // prep should be null since "test" key doesn't exist in shared
        assert_eq!(shared.get("prep"), Some(&NodeValue::Null));
        assert_eq!(shared.get("exec"), Some(&json!("processed")));
    }

    #[test]
    fn test_node_logic_default_implementations() {
        #[derive(Clone)]
        struct DefaultLogic;
        
        impl NodeLogic for DefaultLogic {
            fn clone_box(&self) -> Box<dyn NodeLogic> {
                Box::new(self.clone())
            }
        }
        
        let logic = DefaultLogic;
        let params = HashMap::new();
        let shared = HashMap::new();
        let mut shared_mut = HashMap::new();
        
        let prep = logic.prep(&params, &shared);
        assert_eq!(prep, NodeValue::default());
        
        let exec = logic.exec(NodeValue::Null);
        assert_eq!(exec, NodeValue::default());
        
        let post = logic.post(&mut shared_mut, NodeValue::Null, NodeValue::Null);
        assert_eq!(post, None);
    }
}
