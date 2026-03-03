export const GPU_PRESETS = {
  consumer: [
    { name: 'RTX 4070 Ti', vram: 12.9 },
    { name: 'RTX 4080', vram: 17.2 },
    { name: 'RTX 4090', vram: 25.8 },
    { name: 'RTX 5090 (32GB)', vram: 32 },
  ],
  professional: [
    { name: 'RTX 6000 Ada', vram: 51.5 },
    { name: 'L40', vram: 51.5 },
    { name: 'L40S', vram: 51.5 },
    { name: 'A40', vram: 51.5 },
    { name: 'A100 (40GB)', vram: 42.5 },
    { name: 'A100 (80GB)', vram: 85.9 },
    { name: 'H100 (80GB)', vram: 85.9 },
    { name: 'H200 (141GB)', vram: 150.5 },
    { name: 'NPU 910B (64GB)', vram: 64.0 },
  ],
};

export const MODEL_PRESETS = [
  {
    name: 'Llama-3.1-8B',
    weights: 16,
    layers: 32,
    kvHeads: 8,
    headDim: 128,
    attnHeads: 32,
    context: 131072,
    quant: 'none',
    bits: 16,
    baseParams: 8,
  },
  {
    name: 'Llama-3.1-70B',
    weights: 140,
    layers: 80,
    kvHeads: 8,
    headDim: 128,
    attnHeads: 64,
    context: 131072,
    quant: 'none',
    bits: 16,
    baseParams: 70,
  },
  {
    name: 'Qwen2.5-72B',
    weights: 144,
    layers: 80,
    kvHeads: 8,
    headDim: 128,
    attnHeads: 64,
    context: 131072,
    quant: 'none',
    bits: 16,
    baseParams: 72,
  },
  {
    name: 'HyperNova-60B',
    weights: 42,
    layers: 80,
    kvHeads: 8,
    headDim: 128,
    attnHeads: 64,
    context: 131072,
    quant: 'awq',
    bits: 4,
    baseParams: 80,
  },
];

export const QUANT_PRESETS: Record<string, { bits: number; hasScales: boolean; scaleOverhead: number }> = {
  none: { bits: 16, hasScales: false, scaleOverhead: 0 },
  awq: { bits: 4, hasScales: true, scaleOverhead: 0.1 },
  gptq: { bits: 4, hasScales: true, scaleOverhead: 0.1 },
  'gptq-marlin': { bits: 4, hasScales: true, scaleOverhead: 0.05 },
  squeezellm: { bits: 4, hasScales: true, scaleOverhead: 0.08 },
  fp8: { bits: 8, hasScales: false, scaleOverhead: 0 },
};

export const KV_CACHE_DTYPES = [
  { value: 'auto', label: 'Auto (FP16/BF16)' },
  { value: 'fp8', label: 'FP8 (2x capacity)' },
];

export const ACTIVATION_DTYPES = [
  { value: 'auto', label: 'Auto' },
  { value: 'float16', label: 'FP16' },
  { value: 'bfloat16', label: 'BF16' },
];

export const DEFAULT_GPU_CONFIG = {
  vram: 34.2,
  numGpus: 2,
  utilization: 0.90,
};

export const DEFAULT_MODEL_CONFIG = {
  weightsGB: 42,
  numLayers: 80,
  kvHeads: 8,
  headDim: 128,
  attnHeads: 64,
};

export const DEFAULT_QUANT_CONFIG = {
  method: 'awq',
  bits: 4,
  baseParams: 80,
  groupSize: 128,
};

export const DEFAULT_VLLM_CONFIG = {
  maxModelLen: 16384,
  maxNumSeqs: 256,
  maxBatchedTokens: 8192,
  kvCacheDtype: 'auto',
  activationDtype: 'auto',
  cudaGraphs: true,
  overheadPadding: 1.0,
};
