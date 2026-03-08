# Calculation Accuracy Design

**Context**
The calculator is directionally correct for KV cache sizing, but it over-simplifies two major inputs: runtime activation size and model weight size. This causes visible errors for models like `Qwen/Qwen3-32B` and mixed-dtype checkpoints like `deepseek-ai/DeepSeek-R1`.

**Goals**
- Prefer true `hidden_size` over `attn_heads × head_dim` when estimating activation and CUDA graph memory.
- Prefer checkpoint-derived weight size over `params × bytes_per_param` when the selected quantization matches the fetched model.
- Keep manual quantization comparison working by falling back to estimation when the user overrides the detected quantization.
- Improve automatic quantization detection from Hugging Face config and tags.

**Non-goals**
- Large UI refactors.
- Full MoE-specific runtime overhead modeling.
- Exact engine-internal CUDA graph behavior modeling.

**Design**
1. Extend fetched model metadata with:
   - `hiddenSize`
   - `checkpointBytes`
   - `checkpointBytesSource`
   - `detectedQuantMethod`
2. Fetch `model.safetensors.index.json` in addition to model API + `config.json`.
3. Derive checkpoint size with this preference order:
   - `modelInfo.safetensors.parameters` + dtype-aware byte mapping
   - `model.safetensors.index.json.metadata.total_size`
   - fallback to parameter-count estimation
4. In runtime calculation:
   - activation hidden size uses fetched `hiddenSize / tp` when available
   - weights use checkpoint bytes if the selected quantization matches the fetched model's detected quantization
   - otherwise weights use parameter-based estimation
5. Add a light-weight 4-bit metadata overhead approximation for estimated `int4`/`fp4` weights.

**Validation**
- Verify `DeepSeek-R1` auto-detects MLA and uses fetched quantized checkpoint sizing.
- Verify `Qwen3-32B` uses real `hidden_size=5120` for overhead instead of `64 × 128`.
- Run focused scripts/tests covering weight-source selection and hidden-size-driven overhead.
