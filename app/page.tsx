'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import GPUConfigComponent from './components/GPUConfig';
import { calculateVRAM, createDebouncedConfigSaver } from './lib/calculations';
import { saveConfiguration, loadConfiguration } from './lib/storage';
import {
  DEFAULT_GPU_CONFIG,
  DEFAULT_MODEL_CONFIG,
  DEFAULT_QUANT_CONFIG,
  DEFAULT_VLLM_CONFIG,
} from './lib/constants';
import type { GPUConfig, ModelConfig, QuantizationConfig, VLLMConfig, CalculationResult } from './lib/types';
import styles from './page.module.css';

export default function Home() {
  const [gpuConfig, setGpuConfig] = useState<GPUConfig>(DEFAULT_GPU_CONFIG);
  const [modelConfig, setModelConfig] = useState<ModelConfig>(DEFAULT_MODEL_CONFIG);
  const [quantConfig, setQuantConfig] = useState<QuantizationConfig>(DEFAULT_QUANT_CONFIG);
  const [vllmConfig, setVLLMConfig] = useState<VLLMConfig>(DEFAULT_VLLM_CONFIG);
  const [result, setResult] = useState<CalculationResult | null>(null);

  // Load saved configuration on mount
  useEffect(() => {
    const saved = loadConfiguration();
    if (saved) {
      setGpuConfig(saved.gpu);
      setModelConfig(saved.model);
      setQuantConfig(saved.quant);
      setVLLMConfig(saved.vllm);
    }
  }, []);

  // Calculate results
  const calculate = useCallback(() => {
    const calculated = calculateVRAM(gpuConfig, modelConfig, quantConfig, vllmConfig);
    setResult(calculated);
  }, [gpuConfig, modelConfig, quantConfig, vllmConfig]);

  // Recalculate when configs change
  useEffect(() => {
    calculate();
  }, [calculate]);

  // Debounced save
  const debouncedSave = useRef(
    createDebouncedConfigSaver(saveConfiguration, 2000)
  ).current;

  useEffect(() => {
    debouncedSave(gpuConfig, modelConfig, quantConfig, vllmConfig);
  }, [gpuConfig, modelConfig, quantConfig, vllmConfig, debouncedSave]);

  return (
    <div className={styles.container}>
      <header className={styles.header} role="banner">
        <h1>vLLM VRAM Calculator</h1>
        <p className={styles.subtitle}>Plan GPU memory allocation for LLM deployments</p>
      </header>

      <div className={styles.grid}>
        <div className={styles.inputsColumn}>
          <GPUConfigComponent config={gpuConfig} onChange={setGpuConfig} />

          {/* Placeholder for other components - will add in next iteration */}
          <div style={{ padding: '20px', background: 'var(--surface-secondary)', borderRadius: '12px' }}>
            <p style={{ color: 'var(--text-muted)' }}>Model, Quantization, and vLLM configuration components coming next...</p>
          </div>
        </div>

        <div className={styles.resultsColumn}>
          {result && (
            <div style={{ padding: '20px', background: 'var(--surface-secondary)', borderRadius: '12px' }}>
              <h2 style={{ marginBottom: '16px' }}>Memory Breakdown (Per GPU)</h2>
              <div className="monospace">
                <p>Available VRAM: {result.availableVramPerGpu.toFixed(2)} GB</p>
                <p>Weights: {result.weightsPerGpu.toFixed(2)} GB</p>
                <p>KV Cache: {result.totalKVCacheMemory.toFixed(2)} GB</p>
                <p>Free Memory: {result.freeMemory.toFixed(2)} GB</p>
                <p>Usage: {result.memoryUsagePercent.toFixed(1)}%</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
