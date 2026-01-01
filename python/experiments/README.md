# Experiments Directory

This directory contains benchmark- and paper-specific experimental setups for the AdaptiveMultimodalRAG framework.

## Structure

- **`datasets/`**: Dataset loaders and preprocessing utilities for research benchmarks
- **`metrics/`**: Evaluation metrics and scoring functions for experimental analysis
- **`runners/`**: Experiment execution scripts and configurations for specific research scenarios
- **`results/`**: Output artifacts from experimental runs (results, logs, checkpoints)

## Framework Separation

The core experiment orchestration logic and lifecycle management reside in `python/src/experiment_core/`. This directory (`python/experiments/`) focuses on the research-specific implementations that utilize the framework's core capabilities.

## Purpose

This directory serves as the scaffold for reproducible research experiments, separating domain-specific experimental setups from the framework-level experiment execution engine.

