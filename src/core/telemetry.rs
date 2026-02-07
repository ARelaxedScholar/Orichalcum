use std::collections::HashMap;
use crate::core::sync_impl::NodeValue;
use serde::{Serialize, Deserialize};

/// A single entry in the execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub timestamp: u64,
    pub task_id: String,
    pub signature_hash: String,
    pub instruction_hash: String,
    pub inputs: NodeValue,
    pub outputs: NodeValue,
    pub model_name: String,
    pub training_hash: Option<String>,
    pub fitness_score: Option<f64>,
    pub metadata: HashMap<String, String>,
}

/// Trait for recording execution traces.
pub trait Telemetry: Send + Sync {
    fn record(&self, entry: TraceEntry);
    fn flush(&self);
}

/// Simple in-memory collector for traces.
pub struct MemoryTelemetry {
    traces: std::sync::Mutex<Vec<TraceEntry>>,
}

impl MemoryTelemetry {
    pub fn new() -> Self {
        Self {
            traces: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn get_traces(&self) -> Vec<TraceEntry> {
        self.traces.lock().unwrap().clone()
    }
}

impl Telemetry for MemoryTelemetry {
    fn record(&self, entry: TraceEntry) {
        self.traces.lock().unwrap().push(entry);
    }

    fn flush(&self) {
        // No-op for memory collector
    }
}
