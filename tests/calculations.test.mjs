import test from "node:test";
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";

import { calculateVRAM, createDebouncedConfigSaver } from "../app/lib/calculations.ts";

const gpu = {
  vram: 34.2,
  numGpus: 2,
  utilization: 0.9,
};

const model = {
  weightsGB: 42,
  numLayers: 80,
  kvHeads: 8,
  headDim: 128,
  attnHeads: 64,
};

const quant = {
  method: "awq",
  bits: 4,
  baseParams: 80,
  groupSize: 128,
};

const vllm = {
  maxModelLen: 16384,
  maxNumSeqs: 16,
  maxBatchedTokens: 8192,
  kvCacheDtype: "auto",
  activationDtype: "auto",
  cudaGraphs: true,
  overheadPadding: 1.0,
};

function almostEqual(actual, expected, epsilon = 1e-6) {
  return Math.abs(actual - expected) <= epsilon;
}

test("allocates the full KV pool after fixed overhead", () => {
  const result = calculateVRAM(gpu, model, quant, vllm);

  const fixedOverhead = result.weightsPerGpu + result.cudaGraphsMemory + result.overheadMemory;
  const expectedKvPool = Math.max(result.availableVramPerGpu - fixedOverhead, 0);

  assert.equal(result.isOverCapacity, false);
  assert.equal(almostEqual(result.totalKVCacheMemory, expectedKvPool), true);
  assert.equal(almostEqual(result.freeMemory, 0), true);
});

test("quantization settings change memory estimation", () => {
  const fp16Result = calculateVRAM(
    gpu,
    model,
    { ...quant, method: "none", bits: 16 },
    vllm,
  );
  const awqResult = calculateVRAM(gpu, model, quant, vllm);

  assert.notEqual(awqResult.overheadMemory, fp16Result.overheadMemory);
  assert.notEqual(awqResult.totalKVCacheMemory, fp16Result.totalKVCacheMemory);
});

test("activation dtype changes runtime overhead estimation", () => {
  const fp16Result = calculateVRAM(gpu, model, quant, { ...vllm, activationDtype: "float16" });
  const fp8Result = calculateVRAM(gpu, model, quant, { ...vllm, activationDtype: "fp8" });

  assert.notEqual(fp16Result.cudaGraphsMemory + fp16Result.overheadMemory, fp8Result.cudaGraphsMemory + fp8Result.overheadMemory);
});

test("debounced saver persists the latest config values", async () => {
  const calls = [];
  const saver = (gpuConfig, modelConfig, quantConfig, vllmConfig) => {
    calls.push({ gpuConfig, modelConfig, quantConfig, vllmConfig });
  };

  const saveDebounced = createDebouncedConfigSaver(saver, 25);

  const firstGpu = { ...gpu, utilization: 0.8 };
  const secondGpu = { ...gpu, utilization: 0.91 };

  saveDebounced(firstGpu, model, quant, vllm);
  saveDebounced(secondGpu, model, quant, vllm);

  await delay(40);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].gpuConfig.utilization, secondGpu.utilization);
});
