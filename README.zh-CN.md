# vLLM VRAM 计算器

一个用于在使用 vLLM 部署大型语言模型（LLMs）时规划 GPU 内存分配的高级网页计算器。该工具通过计算内存需求、分析容量限制并生成优化的 vLLM 启动命令来帮助您优化推理部署配置。

🔗 **[在线演示](https://taylorelley.github.io/vllm-vram-calc/)**（GitHub Pages）

## 功能

### GPU 配置
- **GPU 预设下拉菜单**：方便选择 10+ 种常见 GPU：
  - **消费级**：RTX 3090、4070 Ti、4080、4090、5090
  - **专业级**：A6000、A100（40/80GB）、L40S、H100
  - 使用来自 `nvidia-smi` 的实际可用 VRAM（十进制 GB）
- **张量并行（Tensor Parallelism）**：支持多 GPU 部署并自动计算每个 GPU 的分配
- **显存利用率控制**：可配置 GPU 内存利用率（通常 0.85–0.95）
- **单位一致性**：所有计算使用十进制 GB（1 GB = 1,000³ 字节），以匹配 GPU 规格

### 模型配置
- **HuggingFace 集成**：自动从 HuggingFace Hub 获取模型配置
  - 提取架构、层数、KV heads、head 维度
  - 从 safetensors 元数据估算模型权重大小
  - 根据模型标签检测量化信息
  - 只需输入模型 ID 并点击“Fetch”即可
- **手动配置**：完全控制以下参数：
  - 模型权重（GB）
  - 层数
  - KV heads（用于 Grouped Query Attention）
  - head 维度

### 量化支持
全面的量化估算，支持：
- **FP16/BF16**（16 位浮点）
- **FP8**（8 位浮点）
- **MXFP4**（4 位 MX 格式）
- **AWQ**（4 位 Activation-aware Weight Quantization）
- **GPTQ**（4 位 GPT Quantization）
- **BitsAndBytes**（NF4、INT8）
- **EXL2**（可变位宽）
- **GGUF**（可变位宽）

计算器会估算：
- 基于每参数位数的权重大小
- 分组量化的 scale / zero-point 开销
- 含量化元数据的模型总大小

### vLLM 配置
可微调的部署参数：
- **Max Model Length**：最大上下文窗口
- **Max Num Seqs**：并发序列数
- **Max Batched Tokens**：每次前向的 token 数量
- **KV Cache Dtype**：BF16/FP16 或 FP8 压缩
- **CUDA Graphs**：启用/禁用（影响约 ~2.5GB 的开销）
- **额外开销估算**：计入激活和缓冲区占用

### 内存分析
计算器提供详细的内存分解：
- **每 GPU 内存使用情况**：
  - 模型权重（在 GPU 间分布）
  - KV cache（根据活动 tokens 估算）
  - CUDA graphs 开销
  - 框架（framework）开销
  - 可用余量
- **可视化内存条**：按颜色区分的内存分配可视化
- **容量分析**：
  - KV cache 可容纳的最大 token 数
  - 单序列最大上下文长度
  - 每序列平均上下文长度
  - 推荐的 `max-num-seqs` 以优化吞吐量
- **vLLM 比对**：以十进制 GB 与二进制 GiB 两种单位显示，便于与 vLLM 输出对照

### 命令生成
自动生成带有适当 flag 的优化 vLLM 启动命令，例如：
- `--tensor-parallel-size`
- `--max-model-len`
- `--max-num-seqs`
- `--max-num-batched-tokens`
- `--gpu-memory-utilization`
- `--enable-chunked-prefill`
- `--enforce-eager`（当禁用 CUDA graphs 时）
- `--disable-custom-all-reduce`（用于多 GPU）

## 使用说明

### 快速开始
1. 打开 [在线演示](https://taylorelley.github.io/vllm-vram-calc/) 或在本地打开 `index.html`
2. 从下拉列表选择 GPU（例如 RTX 5090）
3. 输入 HuggingFace 模型 ID 并点击“Fetch”自动填充配置
   - 或手动配置模型参数
4. 根据需要调整 vLLM 配置（max-model-len、max-num-seqs 等）
5. 查看内存分解与容量分析
6. 检查 vLLM 比对值（以 GiB 显示）以匹配预期输出
7. 复制生成的 vLLM 命令

### 示例工作流程

#### 使用 HuggingFace 集成（推荐）
1. 从下拉选择 GPU：**RTX 5090 (32GB)**
2. 设置 GPU 数量（TP 大小）：**2**
3. 输入模型 ID：`MultiverseComputingCAI/HyperNova-60B`
4. 点击 **"Fetch"**
5. 检查自动填充的配置（layers、KV heads、weights 等）
6. 根据推荐值调整 `max-num-seqs`
7. 检查内存状态（应显示 ✓ Configuration looks good）
8. 使用 GiB 显示与 vLLM 输出对比
9. 复制生成的命令

#### 手动配置
1. 从下拉选择 GPU：**RTX 5090 (32GB)**（或手动输入自定义 VRAM）
2. 设置 GPU 数量（TP 大小）：**2**
3. 输入模型权重：**36.3 GB**
4. 配置层数：**32**、KV heads：**8**、head 维度：**64**
5. 选择量化方法（例如 **MXFP4**）
6. 配置基准参数（base params）：**60** billion
7. 点击 “Apply Estimate” 使用计算出的模型大小
8. 调整 `max-model-len`：**131072** 和 `max-num-seqs`：**8**
9. 查看容量分析与 vLLM 比对值

### 输出说明

**状态指示**：
- ✓ **Configuration looks good**（绿色）：配置安全，有余量
- ⚡ **Tight**（黄色）：在高负载下可能接近 OOM，考虑降低 `max-num-seqs`
- ⚠ **OOM Risk**（红色）：可能失败，需减少上下文/序列数或启用 FP8 KV cache

**关键指标**：
- **Max Tokens in Cache**：KV cache 在所有序列中的总容量
- **Max Context (1 seq)**：单次请求的最大上下文长度
- **Avg Context (all seqs)**：所有并发槽满载时每序列的平均上下文
- **Recommended max-num-seqs**：在约 32K 平均上下文下的并发推荐值

## 技术细节

### 单位：GB 与 GiB
计算器在所有计算中使用 **十进制 GB**（1 GB = 1,000,000,000 字节），以匹配：
- GPU 厂商规格（`nvidia-smi` 报告为 MiB，但宣传常以 GB 为准）
- 常见存储单位约定
- HuggingFace 上的模型大小

**重要**：vLLM 的日志使用 **二进制 GiB**（1 GiB = 1,073,741,824 字节）。

**换算**：1 GB（十进制）≈ 0.931 GiB（二进制）

计算器在状态栏同时显示两种单位以便对比 vLLM 输出：
```
💡 vLLM comparison: Available 28.66 GiB (binary) • KV cache 7.69 GiB
```

### KV Cache 计算
计算器使用以下公式估算 KV cache：
```
bytes_per_token_per_layer = 2 (K+V) × kv_heads_per_gpu × head_dim × dtype_bytes
total_bytes_per_token = bytes_per_token_per_layer × num_layers
```

在存在张量并行时，KV heads 会在 GPU 间分配，从而降低每个 GPU 上的 cache 大小。

### 内存布局
每个 GPU 的内存划分为：
1. **固定开销**：
   - 模型权重（总量 / GPU 数）
   - CUDA graphs（启用时约 ~2.5GB）
   - 框架开销（可配置）
2. **动态 KV Cache**：在剩余空间内填充，直至达到 `gpu_memory_utilization` 限制

### 量化元数据开销
分组量化方法（AWQ、GPTQ、MXFP4）会存储额外元数据：
- **Scale 因子**：每组 2 字节
- **Zero point**：每组 2 字节（用于非对称量化）
- **组数**：`num_parameters / group_size`

## 浏览器兼容性

兼容所有现代浏览器：
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

无需构建步骤或依赖 — 直接打开 `index.html` 即可。

## 许可证

MIT 许可证 — 详见 LICENSE 文件

## 贡献

欢迎贡献！请查看 TODO.md 了解计划中的功能与改进。

## 致谢

为 vLLM 社区构建，旨在简化生产环境下的 LLM 部署。特别感谢 vLLM 团队提供优秀的推理引擎。
