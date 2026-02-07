use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Represents a record of an optimization run for a specific task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecord {
    pub task_id: String,
    pub signature_hash: String,
    pub instruction_hash: String,
    pub training_hash: Option<String>,
    pub optimization_config_hash: Option<String>,
    pub fitness_score: Option<f64>,
    pub weights_path: Option<PathBuf>,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Basic registry for storing and retrieving optimizations.
/// This implementation is a placeholder for a full SQLite-based registry.
pub struct OptimizationRegistry {
    records: HashMap<String, OptimizationRecord>,
}

impl OptimizationRegistry {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    pub fn register(&mut self, record: OptimizationRecord) {
        self.records.insert(record.task_id.clone(), record);
    }

    pub fn get_by_task_id(&self, task_id: &str) -> Option<&OptimizationRecord> {
        self.records.get(task_id)
    }

    pub fn find_best_match(
        &self,
        signature_hash: &str,
        instruction_hash: &str,
    ) -> Option<&OptimizationRecord> {
        self.records.values()
            .filter(|r| r.signature_hash == signature_hash && r.instruction_hash == instruction_hash)
            .max_by(|a, b| a.fitness_score.partial_cmp(&b.fitness_score).unwrap_or(std::cmp::Ordering::Equal))
    }
}
