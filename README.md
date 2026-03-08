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
计算器提供简化的权重量化估算，支持以下精度：

| 方法 | 字节/参数 | 说明 |
|------|----------|------|
| BF16/FP16 | 2 | 默认精度（无量化） |
| FP8 | 1 | 8 位浮点 |
| INT8 | 1 | 8 位整数 |
| INT4 | 0.5 | 4 位整数 |
| FP4 | 0.5 | 4 位浮点 |

**默认估算公式**：`模型权重 (GB) = 参数量 (B) × 字节/参数`

**当前实现**：
- 当 HuggingFace 提供真实 checkpoint 元数据时，优先使用真实权重大小而不是纯参数量估算
- 当用户手动切换量化方式时，回退到参数量估算
- 对 INT4/FP4 的参数量估算，会加入少量分组量化元数据开销近似（scale/zero-point）

**HuggingFace 标签映射**：从模型标签自动检测量化类型，映射到对应精度：
- `AWQ`/`GPTQ`/`BNB`/`4BIT` → INT4
- `MXFP4`/`FP4` → FP4
- `FP8`/`8BIT` → FP8/INT8

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
3. 输入模型 ID 或手动配置参数
4. 输入参数量（B）：**36.3**
5. 配置层数：**32**、KV heads：**8**、head 维度：**64**
6. 选择量化方法（例如 **INT4** 或 **FP4**）
7. 调整 `max-model-len`：**131072** 和 `max-num-seqs`：**8**
8. 查看容量分析与 vLLM 比对值

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
*- GPU 厂商规格（`nvidia-smi` 报告为 MiB，但宣传常以 GB 为准）
- 常见存储单位约定
- HuggingFace 上的模型大小
*
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
   - CUDA graphs（启用时按激活缓冲区 × 10 计算）
   - 框架开销（可配置）
2. **动态 KV Cache**：在剩余空间内填充，直至达到 `gpu_memory_utilization` 限制

运行时激活与 CUDA graphs 开销会优先使用模型真实 `hidden_size`；只有在缺失该字段时，才回退到 `attention_heads × head_dim` 的近似。

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

## 代码架构

### 项目结构
```
index.html          # 单文件应用（含内联 CSS 与 JavaScript）
├── <style>         # CSS 样式（CSS 变量、响应式设计、深色主题）
├── <body>          # HTML 结构
│   ├── GPU 配置面板
│   ├── 模型配置面板
│   ├── vLLM 配置面板
│   ├── 内存分析面板
│   └── 命令生成面板
└── <script>        # JavaScript 逻辑
```

### 核心模块

#### 1. GPU 配置模块
- **预设下拉菜单**：15+ 种 GPU 预设，含消费级（RTX 3090/4070 Ti/4080/4090/5090）和专业级（A100/H100/L40S 等）
- **实际可用显存**：使用 `nvidia-smi` 显示值（十进制 GB）
- **张量并行支持**：自动计算每 GPU 权重分配
- **显存利用率**：可配置 `gpu_memory_utilization`（默认 0.90）

#### 2. 模型配置模块
- **内置模型预设**：DeepSeek、Qwen、GLM 等厂商模型快速选择
- **HuggingFace 集成**：
  - 从 `config.json` 自动提取层数、KV heads、head 维度、最大上下文
  - 从 `safetensors` 元数据估算参数量
  - 从模型标签检测量化信息
  - 本地缓存 7 天，支持离线使用
- **注意力架构自动检测**：
  - MHA (Multi-Head Attention)
  - GQA (Grouped Query Attention)
  - MLA (Multi-head Latent Attention) — DeepSeek-V2/V3 等模型
  - 自动提取 `kv_lora_rank` 用于 MLA KV Cache 计算

#### 3. KV Cache 计算引擎
```javascript
// GQA/MHA 模式（传统注意力）
bytes_per_token = 2(K+V) × kv_heads_per_gpu × head_dim × dtype_bytes × num_layers

// MLA 模式（压缩潜在向量）
bytes_per_token = 2(K+V) × kv_lora_rank × dtype_bytes × num_layers
```
- 注意：MLA 的 KV Cache 不随张量并行分割（潜在维度在 GPU 间共享）

#### 4. 内存分析模块
- **内存分解表格**：权重、KV Cache、CUDA Graphs、开销、空闲
- **可视化内存条**：按颜色区分各组件，支持悬停提示
- **KV Cache 对比**：已分配 vs 需求对比图表
- **状态指示**：
  - ✓ 配置正常（绿色）
  - ⚡ 紧张（黄色）— 高负载下可能 OOM
  - ⚠ OOM 风险（红色）— 需调整参数

#### 5. 高级功能
- **CUDA Graphs 开销估算**：按激活缓冲区 × 10 倍计算（多图捕获）
- **激活缓冲计算**：`(prefill_tokens + decode_tokens) × hidden_size × dtype × 2`
- **自动保存/恢复配置**：localStorage 存储，30 天有效
- **命令一键复制**：Toast 通知反馈

### 关键代码位置

| 功能 | 位置（行号） |
|------|-------------|
| GPU 预设数据 | `962-987` |
| 内置模型数据 | `1335-1348` |
| HuggingFace API 调用 | `1535-1584` |
| 模型配置提取 | `1602-1729` |
| MLA 检测逻辑 | `1681-1726` |
| KV Cache 计算 | `2169-2186` |
| 内存分解渲染 | `2216-2253` |
| 命令生成 | `2401-2419` |
| 配置保存/恢复 | `2522-2632` |

### 技术实现要点

1. **单位一致性**：所有计算使用十进制 GB（1 GB = 1,000³ bytes），与 GPU 厂商规格一致
2. **vLLM 对比**：额外显示二进制 GiB（1 GiB = 1,073,741,824 bytes），便于与 vLLM 日志对照
3. **响应式设计**：支持 320px ~ 1440px+ 屏幕宽度
4. **无障碍支持**：ARIA 标签、键盘导航、语义化 HTML
5. **错误处理**：网络超时、私有模型、配置缺失等情况的友好提示

### 开发与调试

- 直接打开 `index.html` 即可运行，无需构建
- 主要调试函数：`calculate()`（内存计算核心）
- 状态查看：`detectedAttentionType`、`extractedKvLoraRank`（MLA 检测状态）
