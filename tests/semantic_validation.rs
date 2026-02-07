use orichalcum::prelude::*;
use orichalcum::llm::semantic::signature::Signature;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::json;

#[test]
fn test_signature_parsing() {
    let sig: Signature = "context, question -> answer".parse().unwrap();
    assert_eq!(sig.inputs.len(), 2);
    assert_eq!(sig.inputs[0].name, "context");
    assert_eq!(sig.inputs[1].name, "question");
    assert_eq!(sig.outputs.len(), 1);
    assert_eq!(sig.outputs[0].name, "answer");
}

#[test]
fn test_signature_hash() {
    let sig1: Signature = "a, b -> c".parse().unwrap();
    let sig2: Signature = "a, b -> c".parse().unwrap();
    let sig3: Signature = "b, a -> c".parse().unwrap();
    
    assert_eq!(sig1.structural_hash(), sig2.structural_hash());
    // Order matters for structural identity
    assert_ne!(sig1.structural_hash(), sig3.structural_hash());
}

#[derive(Clone)]
struct MockSealableLogic {
    task_id: String,
    signature: Signature,
}

impl NodeLogic for MockSealableLogic {
    fn clone_box(&self) -> Box<dyn NodeLogic> {
        Box::new(self.clone())
    }
    
    fn as_sealable(&self) -> Option<&dyn Sealable> {
        Some(self)
    }

    fn prep(&self, _params: &HashMap<String, NodeValue>, shared: &HashMap<String, NodeValue>) -> NodeValue {
        let mut inputs = HashMap::new();
        for f in &self.signature.inputs {
            inputs.insert(f.name.clone(), shared.get(&f.name).cloned().unwrap_or(NodeValue::Null));
        }
        json!(inputs)
    }

    fn exec(&self, input: NodeValue) -> NodeValue {
        input
    }
}

impl Sealable for MockSealableLogic {
    fn signature(&self) -> Signature {
        self.signature.clone()
    }
    
    fn task_id(&self) -> String {
        self.task_id.clone()
    }
}

#[test]
fn test_flow_validation_success() {
    let logic1 = MockSealableLogic {
        task_id: "node1".to_string(),
        signature: "input1 -> output1".parse().unwrap(),
    };
    let logic2 = MockSealableLogic {
        task_id: "node2".to_string(),
        signature: "output1 -> result".parse().unwrap(),
    };
    
    let node1 = Node::new(logic1);
    let node2 = Node::new(logic2);
    let start_node = node1.next(Executable::Sync(node2));
    
    let flow = Flow::new(start_node);
    
    // Validate with initial keys
    let result = flow.validate(vec!["input1".to_string()]);
    assert!(result.is_safe());
}

#[test]
fn test_flow_validation_failure() {
    let logic1 = MockSealableLogic {
        task_id: "node1".to_string(),
        signature: "input1 -> output1".parse().unwrap(),
    };
    
    let node1 = Node::new(logic1);
    let flow = Flow::new(node1);
    
    // Validate without required initial keys
    let result = flow.validate(vec![]);
    assert!(!result.is_safe());
    
    match &result.issues[0] {
        ValidationIssue::Error(msg) => {
            assert!(msg.contains("requires input 'input1'"));
        }
        _ => panic!("Expected error issue"),
    }
}

#[tokio::test]
async fn test_telemetry_recording() {
    let logic = MockSealableLogic {
        task_id: "test_telemetry".to_string(),
        signature: "in -> out".parse().unwrap(),
    };
    
    let node = Node::new(logic);
    let sealed = node.seal().unwrap();
    
    let telemetry = Arc::new(MemoryTelemetry::new());
    let mut shared = HashMap::new();
    shared.insert("in".to_string(), json!("data"));
    
    sealed.run(&mut shared, Some(telemetry.as_ref())).await;
    
    let traces = telemetry.get_traces();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].task_id, "test_telemetry");
    assert_eq!(traces[0].inputs, json!({"in": "data"}));
}
