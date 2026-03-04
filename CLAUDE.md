# Claude.md - AI Assistant Guidelines

This document provides guidelines for AI assistants (Claude, GPT, etc.) when working on the vLLM VRAM Calculator project.

## Project Overview

**Type**: Single-page web application (vanilla HTML/CSS/JavaScript)
**Purpose**: GPU memory planning calculator for vLLM LLM deployments
**Architecture**: Self-contained index.html with embedded styles and scripts
**Dependencies**: None (external fonts via CDN only)

## Core Principles

### 1. Simplicity First
This is intentionally a single-file application with no build step. Do not suggest:
- Adding frameworks (React, Vue, etc.)
- Build tools (webpack, vite, etc.) unless explicitly requested
- Package managers (npm, yarn) for basic functionality
- Over-engineering solutions

### 2. Calculation Accuracy is Critical
This tool helps users avoid OOM (Out of Memory) errors in production deployments. Incorrect calculations could lead to:
- Failed deployments
- Wasted GPU time and money
- Frustrated users

**Always**:
- Verify calculation formulas against vLLM source code when available
- Add comments explaining non-obvious calculations
- Consider edge cases (TP > KV heads, very large contexts, etc.)
- Test calculations with known configurations

### 3. User Experience Matters
Users are ML engineers and researchers who need quick, reliable answers. Prioritize:
- Fast page load (no heavy dependencies)
- Instant feedback on parameter changes
- Clear error messages and warnings
- Informative tooltips and hints

## Code Organization

### Structure
```
index.html
├── <head>
│   ├── <style> (CSS)
│   └── fonts (Google Fonts CDN)
└── <body>
    ├── HTML structure
    └── <script> (JavaScript)
```

### CSS Guidelines
- Uses CSS custom properties (variables) for theming
- Mobile-first responsive design
- Color scheme optimized for technical readability
- Monospace fonts (JetBrains Mono) for numbers and code

**When modifying CSS**:
- Update CSS variables in `:root` for theme changes
- Maintain consistent spacing (multiples of 0.25rem)
- Test responsiveness at 320px, 768px, 1024px, 1440px
- Preserve color accessibility (sufficient contrast)

### JavaScript Guidelines
- Vanilla ES6+ JavaScript (no jQuery, no frameworks)
- Functional approach where possible
- Clear variable names (prefer verbosity over brevity)
- Constants at top of script section

**When adding features**:
- Follow existing patterns (e.g., preset button handlers)
- Add input validation
- Trigger recalculation on relevant changes
- Update all three output sections (breakdown, capacity, command)

## Key Calculations

### KV Cache Memory
```javascript
// Per token per layer: 2 (K+V) × kv_heads_per_gpu × head_dim × dtype_bytes
const kvHeadsPerGpu = Math.ceil(kvHeads / numGpus);
const bytesPerTokenPerLayer = 2 * kvHeadsPerGpu * headDim * kvCacheDtypeBytes;
const bytesPerToken = bytesPerTokenPerLayer * numLayers;
```

**Important**: With tensor parallelism, KV heads are distributed across GPUs.

### Quantization Overhead
```javascript
// Group-wise quantization
const numGroups = baseParams / groupSize;
const scaleBytes = numGroups * 2;  // FP16 scale factors
const zeroBytes = numGroups * 2;   // For asymmetric quantization
```

**Note**: Different quantization methods have different overhead patterns.

### MLA (Multi-head Latent Attention) Models

MLA models like DeepSeek-V2/V3 use compressed latent vectors instead of traditional K/V matrices:

```javascript
// MLA: 2 (K+V) × kv_lora_rank × dtype_bytes per layer per token
const bytesPerTokenPerLayer = 2 * kvLoraRank * kvCacheDtypeBytes;
```

**Detection**:
- From config.json: `attention_type: "mla"` or architecture name matching
- Default kv_lora_rank values: DeepSeek-V3 (1536), DeepSeek-V2 (512)

**Important**: MLA KV cache is NOT split by tensor parallelism (latent dimension is shared across GPUs).

### Memory Availability
```javascript
const availableVram = gpuVram * gpuUtilization;  // Per GPU
const weightsPerGpu = modelWeights / numGpus;     // Distributed
const kvAvailable = availableVram - weightsPerGpu - cudaGraphs - overhead;
```

## Common Tasks

### Adding a New GPU Preset
1. Add button to `#gpu-presets` div:
```html
<button class="preset-btn" data-vram="80" data-name="H100">H100 (80GB)</button>
```

2. The event listener is already attached to all preset buttons

### Adding a New Model Preset
1. Add button with complete configuration:
```html
<button class="preset-btn"
  data-weights="42"
  data-layers="80"
  data-kvheads="8"
  data-headdim="128"
  data-name="Model-Name"
  data-context="131072"
  data-quant="awq"
  data-bits="4"
  data-baseparams="80">Model-Name</button>
```

2. Event listener handles all presets automatically

### Adding a New Quantization Method
1. Add to `quantPresets` object:
```javascript
'method-name': { bits: 4, hasScales: true, scaleOverhead: 0.1 }
```

2. Add to `#quant-method` select:
```html
<option value="method-name">Method Name (4-bit)</option>
```

3. Update HuggingFace detection in `extractModelConfig()` if needed

### Modifying Calculations
1. Update calculation in `calculate()` function
2. Add comments explaining the formula
3. Test with various configurations
4. Update relevant display sections
5. Consider edge cases

## HuggingFace Integration

### API Endpoints
- Model info: `https://huggingface.co/api/models/{modelId}`
- Config: `https://huggingface.co/{modelId}/resolve/main/config.json`

### Error Handling
- Handle 404 (model not found)
- Handle 403 (private/gated models)
- Handle network errors
- Handle malformed config.json
- Handle missing fields gracefully

### Config Extraction
Different models use different field names:
- Layers: `num_hidden_layers`, `n_layer`, `num_layers`, `n_layers`
- KV heads: `num_key_value_heads`, `num_kv_heads`, `kv_heads`
- Context: `max_position_embeddings`, `max_seq_len`, `n_positions`

Always check multiple variations and provide fallbacks.

## Testing Checklist

When making changes, verify:

### Calculation Tests
- [ ] Small model on large GPU (should show lots of free space)
- [ ] Large model on small GPU (should show OOM warning)
- [ ] Multi-GPU split (weights should divide correctly)
- [ ] FP8 vs FP16 KV cache (2x capacity difference)
- [ ] CUDA graphs on/off (2.5GB difference)
- [ ] Different quantization methods
- [ ] Edge case: TP size > KV heads

### UI Tests
- [ ] All presets work correctly
- [ ] HuggingFace fetch succeeds for public model
- [ ] HuggingFace fetch fails gracefully for non-existent model
- [ ] Real-time updates on input change
- [ ] Memory bar visualization matches numbers
- [ ] Status badge shows correct warnings
- [ ] Generated command includes all necessary flags
- [ ] Responsive layout works on mobile

### Browser Tests
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari
- [ ] Mobile browsers

## Common Pitfalls

### 1. Forgetting to Trigger Recalculation
When adding new inputs, always add event listeners:
```javascript
newInput.addEventListener('input', calculate);
newInput.addEventListener('change', calculate);
```

### 2. Integer Overflow
For large models (>100B params), use careful calculation order:
```javascript
// Good: divide before multiply
const bytes = (params / 1e9) * (bits / 8) * 1e9;

// Bad: may overflow
const bytes = params * bits / 8;
```

### 3. GPU Distribution Assumptions
Don't assume even distribution. Use `Math.ceil()` for heads:
```javascript
const kvHeadsPerGpu = Math.ceil(kvHeads / numGpus);  // Some GPUs may have +1
```

### 4. Unit Confusion
Be explicit about units:
```javascript
const weightsGB = ...;     // Gigabytes
const weightsBytes = ...;  // Bytes
const kvCacheDtypeBytes = 2;  // Bytes per element
```

## Performance Considerations

### Keep It Fast
- Avoid expensive operations in `calculate()` (called on every input)
- Minimize DOM manipulation
- Use CSS transitions for smooth updates
- Debounce if adding async operations

### Memory Bar Animation
The memory bar uses CSS transitions (0.3s ease) for smooth updates when values change.

## Accessibility

When adding UI elements:
- Use semantic HTML (`<button>`, `<select>`, `<label>`)
- Ensure keyboard navigation works
- Provide `aria-label` for icon-only buttons
- Maintain sufficient color contrast (4.5:1 minimum)
- Test with screen reader if possible

## Documentation

When making significant changes:
1. Update inline comments for complex calculations
2. Update README.md if user-facing features change
3. Add to TODO.md if creating new work items
4. Update this CLAUDE.md if changing architecture or patterns

## Version History

Track major changes here:

### Current Version (v1.0)
- Initial release
- GPU and model presets
- HuggingFace integration
- Quantization estimation
- Tensor parallelism support
- Memory visualization
- Command generation

## Useful References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [HuggingFace Hub API](https://huggingface.co/docs/hub/api)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [PagedAttention](https://arxiv.org/abs/2309.06180)

## Questions to Ask

When working on this project, consider:

1. **Does this help users avoid OOM errors?**
2. **Is the calculation based on vLLM's actual behavior?**
3. **Will this work with no build step?**
4. **Is it fast and responsive?**
5. **Does it handle edge cases?**
6. **Is it accessible?**
7. **Is the code maintainable?**

## Getting Help

If uncertain about vLLM internals:
1. Check vLLM source code (especially `worker/model_runner.py`)
2. Look for similar calculations in vLLM's profiler
3. Test with actual vLLM deployment if possible
4. Ask for clarification on specific calculation methodology

## Future Direction

See TODO.md for planned features. When implementing new features:
- Maintain the single-file architecture until it becomes truly limiting
- Prioritize accuracy over features
- Keep the UI fast and responsive
- Consider backward compatibility (URL parameters, saved configs)

Remember: This tool is used in production environments where mistakes are costly. Accuracy and reliability are more important than features or aesthetics.
