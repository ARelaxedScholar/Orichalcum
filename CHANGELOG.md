# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-02-10

### Added
- **Semantic Layer**: Introduced `Signature`, `Field`, and `Sealable` traits for defining structural contracts.
- **Sealed Nodes**: Added `SealedNode` for validated, globally identifiable task instances.
- **Telemetry**: Implementation of `Telemetry` trait with `MemoryTelemetry` for recording and inspecting execution traces.
- **Validation**: Added `ValidationIssue` and `ValidationResult` for flow-level contract verification.
- Builder pattern for LLM completion methods (`deepseek_complete`, `gemini_complete`, `ollama_complete`)
- Multi-turn fluent message support (`.system()`, `.user()`, `.assistant()`) in builders
- Implicit model validation with thread-safe caching
- Convenience default methods for client configuration (`with_deepseek`, `with_gemini`, `with_ollama`)
- Standardized model selection across all providers with best-in-class defaults (e.g., `gemini-1.5-flash`, `deepseek-chat`, `phi4`)
- Model discovery API (`list_models`) for each provider

### Changed
- Refactored LLM client configuration to use simpler default methods
- Moved custom URL configuration to `with_*_at` methods
- Standardized builder-based completions across all providers

### Fixed
- Improved API discoverability and reduced boilerplate in common use cases

## [0.3.0] - 2025-12-26

### Added
- AsyncFlow: full asynchronous flow implementation for async node orchestration
- AsyncNode: asynchronous node logic with async trait support
- AsyncBatchNode: batch processing for async nodes
- AsyncParallelBatchNode: parallel batch processing for async nodes
- Flake.nix and cargo2nix support for Nix users
- More professional README with comprehensive examples
- Convenience function to get successors from any node
- Edit function to Ollama LLM client (feature-gated)

### Changed
- Node now expects `Executable` instead of `Node` as next step, enabling mixed sync/async workflows
- Flow is now aware of `Executable` types (though synchronous flow still only handles sync nodes)
- Improved internal architecture with better separation of sync and async implementations

### Fixed
- Fixed lifetime errors in AsyncFlow implementation
- Fixed logic for sequential async batch processing
- Fixed trait bound on wrong struct that broke client-side functionality
- Cleaned up code with clippy fixes

### Notes
- This release introduces a complete async counterpart to the existing synchronous API.
- The async API is feature-complete with parallel batch processing capabilities.
- The crate now supports Nix-based development environments via flake.nix.