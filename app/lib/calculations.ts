import type { GPUConfig, ModelConfig, QuantizationConfig, VLLMConfig, CalculationResult } from './types';

const DECIMAL_GB = 1_000_000_000;
const DEFAULT_DTYPE_BYTES = 2;
const CUDA_GRAPH_MULTIPLIER = 10;

export function calculateVRAM(
  gpu: GPUConfig,
  model: ModelConfig,
  quant: QuantizationConfig,
  vllm: VLLMConfig
): CalculationResult {
  // Parse inputs
  const gpuVramGB = gpu.vram;
  const numGpus = gpu.numGpus;
  const gpuUtilization = gpu.utilization;

  const modelWeightsGB = model.weightsGB;
  const numLayers = model.numLayers;
  const kvHeads = model.kvHeads;
  const headDim = model.headDim;
  const attnHeads = Math.max(model.attnHeads || kvHeads, kvHeads);

  const maxModelLen = vllm.maxModelLen;
  const maxNumSeqs = vllm.maxNumSeqs;
  const maxBatchedTokens = vllm.maxBatchedTokens;

  const kvCacheDtypeBytes = getKVCacheDtypeBytes(vllm.kvCacheDtype);
  const activationDtypeBytes = getActivationDtypeBytes(vllm.activationDtype);
  const cudaGraphsEnabled = vllm.cudaGraphs;
  const overheadPaddingGB = vllm.overheadPadding;
  const quantizationOverheadBytes = getQuantizationMetadataBytes(quant, numGpus);
  const quantizedWeightEstimateGB = getQuantizedWeightEstimateGB(quant);

  // Calculate available VRAM per GPU
  const totalVramBytes = gpuVramGB * DECIMAL_GB;
  const availableVramBytes = totalVramBytes * gpuUtilization;
  const availableVramGB = availableVramBytes / DECIMAL_GB;

  // Calculate model weights per GPU (distributed via TP)
  const weightsPerGpuGB = modelWeightsGB / numGpus;
  const weightsPerGpuBytes = weightsPerGpuGB * DECIMAL_GB;

  // Activation buffers depend on batch shape and activation dtype.
  const attnHeadsPerGpu = Math.ceil(attnHeads / numGpus);
  const hiddenSizePerGpu = attnHeadsPerGpu * headDim;
  const activationTokens = maxBatchedTokens + maxNumSeqs;
  const activationBytes = activationTokens * hiddenSizePerGpu * activationDtypeBytes;
  const activationOverheadBytes = activationBytes * 2;

  // CUDA graphs memory (per GPU)
  const cudaGraphsBytes = cudaGraphsEnabled ? activationBytes * CUDA_GRAPH_MULTIPLIER : 0;
  const cudaGraphsGB = cudaGraphsBytes / DECIMAL_GB;

  // Extra overhead = activation buffers + manual padding + quantization metadata.
  const overheadBytes = activationOverheadBytes + (overheadPaddingGB * DECIMAL_GB) + quantizationOverheadBytes;
  const overheadGB = overheadBytes / DECIMAL_GB;

  // Calculate KV cache memory per token
  // With tensor parallelism, KV heads are distributed across GPUs
  const kvHeadsPerGpu = Math.ceil(kvHeads / numGpus);

  // KV cache formula: 2 (K+V) × kv_heads_per_gpu × head_dim × dtype_bytes
  const bytesPerTokenPerLayer = 2 * kvHeadsPerGpu * headDim * kvCacheDtypeBytes;
  const bytesPerToken = bytesPerTokenPerLayer * numLayers;

  // KV cache per sequence
  const bytesPerSeq = bytesPerToken * maxModelLen;

  // Available memory for KV cache
  const kvAvailableBytes = availableVramBytes - weightsPerGpuBytes - cudaGraphsBytes - overheadBytes;

  // Check if we're over capacity
  if (kvAvailableBytes <= 0) {
    return {
      availableVramPerGpu: availableVramGB,
      weightsPerGpu: weightsPerGpuGB,
      cudaGraphsMemory: cudaGraphsGB,
      overheadMemory: overheadGB,
      kvBytesPerToken: bytesPerToken,
      kvBytesPerSeq: bytesPerSeq,
      totalKVCacheMemory: 0,
      maxTokensForKV: 0,
      maxConcurrentSequences: 0,
      totalBatchedTokens: 0,
      freeMemory: kvAvailableBytes / DECIMAL_GB,
      memoryUsagePercent: 100,
      isOverCapacity: true,
      warnings: ['Model weights exceed available VRAM. Reduce model size or increase GPU count.'],
      command: '',
    };
  }

  // Calculate maximum tokens that can fit in KV cache
  const maxTokensForKV = Math.floor(kvAvailableBytes / bytesPerToken);

  // Calculate capacity based on user's max_num_seqs and max_model_len
  const tokensPerSeq = maxModelLen;
  const maxConcurrentSeqs = Math.floor(maxTokensForKV / tokensPerSeq);

  // Actual concurrent sequences (limited by user's max_num_seqs)
  const actualMaxNumSeqs = Math.min(maxConcurrentSeqs, maxNumSeqs);

  // vLLM pre-allocates the full KV cache pool at startup.
  const totalKVCacheBytes = kvAvailableBytes;
  const totalKVCacheGB = totalKVCacheBytes / DECIMAL_GB;

  // Free memory
  const usedMemoryBytes = weightsPerGpuBytes + cudaGraphsBytes + overheadBytes + totalKVCacheBytes;
  const freeMemoryBytes = Math.max(0, availableVramBytes - usedMemoryBytes);
  const freeMemoryGB = freeMemoryBytes / DECIMAL_GB;

  // Memory usage percentage
  const memoryUsagePercent = (usedMemoryBytes / availableVramBytes) * 100;

  // Calculate total batched tokens (min of max_batched_tokens and capacity)
  const capacityBatchedTokens = actualMaxNumSeqs * tokensPerSeq;
  const totalBatchedTokens = Math.min(maxBatchedTokens, capacityBatchedTokens);

  // Warnings
  const warnings: string[] = [];
  if (actualMaxNumSeqs < maxNumSeqs) {
    warnings.push(
      `KV cache can only fit ${actualMaxNumSeqs} sequences (you requested ${maxNumSeqs}). ` +
      `Consider reducing max_model_len or increasing GPU memory.`
    );
  }
  const fixedUsagePercent = ((weightsPerGpuBytes + cudaGraphsBytes + overheadBytes) / availableVramBytes) * 100;
  if (fixedUsagePercent > 95) {
    warnings.push('Fixed memory overhead is very high (>95%). Reduce model size or activation settings.');
  }
  if (kvHeadsPerGpu * numGpus > kvHeads) {
    warnings.push(
      `With ${numGpus} GPUs, some GPUs will have ${kvHeadsPerGpu} KV heads while the model has ${kvHeads}. ` +
      `This may cause slight imbalance.`
    );
  }
  if (quantizedWeightEstimateGB !== null) {
    const estimateGap = Math.abs(modelWeightsGB - quantizedWeightEstimateGB) / quantizedWeightEstimateGB;
    if (estimateGap > 0.2) {
      warnings.push(
        `Model weights (${modelWeightsGB.toFixed(2)} GB) differ from ${quant.bits}-bit estimate ` +
        `(${quantizedWeightEstimateGB.toFixed(2)} GB). Verify quantization inputs.`
      );
    }
  }

  // Generate vLLM command
  const command = generateVLLMCommand({
    gpu,
    model,
    vllm,
    actualMaxNumSeqs,
    totalBatchedTokens,
  });

  return {
    availableVramPerGpu: availableVramGB,
    weightsPerGpu: weightsPerGpuGB,
    cudaGraphsMemory: cudaGraphsGB,
    overheadMemory: overheadGB,
    kvBytesPerToken: bytesPerToken,
    kvBytesPerSeq: bytesPerSeq,
    totalKVCacheMemory: totalKVCacheGB,
    maxTokensForKV,
    maxConcurrentSequences: actualMaxNumSeqs,
    totalBatchedTokens,
    freeMemory: freeMemoryGB,
    memoryUsagePercent,
    isOverCapacity: false,
    warnings,
    command,
  };
}

function getKVCacheDtypeBytes(dtype: string): number {
  if (dtype === 'fp8') return 1;
  return DEFAULT_DTYPE_BYTES;
}

function getActivationDtypeBytes(dtype: string): number {
  if (dtype === 'fp8') return 1;
  if (dtype === 'float32') return 4;
  return DEFAULT_DTYPE_BYTES;
}

function getQuantizationMetadataBytes(quant: QuantizationConfig, numGpus: number): number {
  if (quant.method === 'none') return 0;
  if (quant.baseParams <= 0 || quant.groupSize <= 0) return 0;
  if (numGpus <= 0) return 0;

  const parameterCount = quant.baseParams * DECIMAL_GB;
  const numGroups = parameterCount / quant.groupSize;
  const scaleBytes = numGroups * 2;
  const zeroPointBytes = numGroups * 2;

  return (scaleBytes + zeroPointBytes) / numGpus;
}

function getQuantizedWeightEstimateGB(quant: QuantizationConfig): number | null {
  if (quant.baseParams <= 0 || quant.bits <= 0) return null;
  return quant.baseParams * (quant.bits / 8);
}

function generateVLLMCommand(params: {
  gpu: GPUConfig;
  model: ModelConfig;
  vllm: VLLMConfig;
  actualMaxNumSeqs: number;
  totalBatchedTokens: number;
}): string {
  const { gpu, model, vllm, actualMaxNumSeqs, totalBatchedTokens } = params;

  let cmd = 'vllm serve';

  if (model.name) {
    cmd += ` <span class="command-model">${model.name}</span>`;
  } else {
    cmd += ' <span class="command-placeholder">&lt;model-name&gt;</span>';
  }

  cmd += ` \\<br>&nbsp;&nbsp;--max-model-len <span class="command-value">${vllm.maxModelLen}</span>`;
  cmd += ` \\<br>&nbsp;&nbsp;--max-num-seqs <span class="command-value">${actualMaxNumSeqs}</span>`;

  if (totalBatchedTokens !== vllm.maxBatchedTokens) {
    cmd += ` \\<br>&nbsp;&nbsp;--max-num-batched-tokens <span class="command-value">${totalBatchedTokens}</span>`;
  }

  if (vllm.kvCacheDtype !== 'auto') {
    cmd += ` \\<br>&nbsp;&nbsp;--kv-cache-dtype <span class="command-value">${vllm.kvCacheDtype}</span>`;
  }

  if (gpu.numGpus > 1) {
    cmd += ` \\<br>&nbsp;&nbsp;--tensor-parallel-size <span class="command-value">${gpu.numGpus}</span>`;
  }

  if (!vllm.cudaGraphs) {
    cmd += ` \\<br>&nbsp;&nbsp;--enforce-eager`;
  }

  cmd += ` \\<br>&nbsp;&nbsp;--gpu-memory-utilization <span class="command-value">${gpu.utilization.toFixed(2)}</span>`;

  return cmd;
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

export function createDebouncedConfigSaver(
  saveFn: (
    gpu: GPUConfig,
    model: ModelConfig,
    quant: QuantizationConfig,
    vllm: VLLMConfig
  ) => void,
  wait: number
): (
  gpu: GPUConfig,
  model: ModelConfig,
  quant: QuantizationConfig,
  vllm: VLLMConfig
) => void {
  return debounce(saveFn, wait);
}

export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 B';

  const k = 1000; // Decimal (not binary)
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals);
}
