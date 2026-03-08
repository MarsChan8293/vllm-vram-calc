# Calculation Accuracy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve weight and runtime overhead accuracy for fetched models without breaking manual quantization comparison.

**Architecture:** Extend Hugging Face metadata extraction with checkpoint-size and hidden-size fields, store them in calculator state, and update calculation logic to prefer true model metadata when it is compatible with the user's selected quantization. Keep existing fallback estimation paths for manual inputs and overridden quantization.

**Tech Stack:** Static HTML, browser JavaScript, Node verification scripts

---

### Task 1: Persist richer fetched model metadata

**Files:**
- Modify: `index.html`

**Steps:**
1. Add internal state for fetched `hiddenSize`, `checkpointBytes`, `checkpointBytesSource`, and `detectedQuantMethod`.
2. Extend Hugging Face fetch flow to request `model.safetensors.index.json`.
3. Update model extraction helpers to derive checkpoint bytes and quantization from config/tags.
4. Surface the new metadata in the model info card where useful.

### Task 2: Fix runtime calculation inputs

**Files:**
- Modify: `index.html`

**Steps:**
1. Use fetched `hiddenSize / numGpus` for activation and CUDA-graph estimates when available.
2. Keep the existing `attnHeads × headDim` path only as fallback.
3. Choose between checkpoint-derived weights and parameter-based estimates based on selected vs detected quantization.
4. Add a light 4-bit metadata overhead approximation for estimated `int4`/`fp4` paths.

### Task 3: Add regression coverage

**Files:**
- Modify: `verify_calculation.js`
- Create: `tests/accuracy-regressions.mjs`

**Steps:**
1. Add focused regression checks for Qwen hidden-size overhead behavior.
2. Add focused regression checks for DeepSeek checkpoint-size behavior.
3. Keep tests self-contained and runnable with Node.

### Task 4: Verify behavior

**Files:**
- Modify: `README.md` if wording needs to match implementation
- Modify: `README.zh-CN.md` if wording needs to match implementation

**Steps:**
1. Run focused regression tests.
2. Run existing verification scripts.
3. Update docs only if the code behavior has materially changed.
