/// Represents the availability of a key in the shared state during workflow validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAvailability {
    /// The key is guaranteed to exist on every possible execution path.
    Always,
    /// The key exists on some execution paths but not all (e.g., due to branching).
    Sometimes,
    /// The key is not available on any path leading to the current node.
    Never,
}

/// Represents an issue found during workflow validation.
#[derive(Debug, Clone)]
pub enum ValidationIssue {
    /// A hard error: a required input key is guaranteed to be missing.
    Error(String),
    /// A warning: a required input key might be missing on some execution paths.
    Warning(String),
}

/// The result of a workflow validation pass.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.issues.push(ValidationIssue::Error(msg.into()));
    }

    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.issues.push(ValidationIssue::Warning(msg.into()));
    }

    pub fn is_safe(&self) -> bool {
        !self.issues.iter().any(|i| matches!(i, ValidationIssue::Error(_)))
    }

    pub fn has_warnings(&self) -> bool {
        self.issues.iter().any(|i| matches!(i, ValidationIssue::Warning(_)))
    }
    
    pub fn print_summary(&self) {
        if self.is_safe() && !self.has_warnings() {
            println!("✅ Workflow validation passed: All data-flow contracts are satisfied.");
            return;
        }

        for issue in &self.issues {
            match issue {
                ValidationIssue::Error(msg) => println!("❌ Error: {}", msg),
                ValidationIssue::Warning(msg) => println!("⚠️ Warning: {}", msg),
            }
        }
    }
}
