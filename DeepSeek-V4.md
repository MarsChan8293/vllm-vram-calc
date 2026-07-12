下面按你刚才 **GLM-5.2 同样的口径**，换成 **DeepSeek V4 Flash** 来算：

```json
{
  "prefill": {
    "dp_size": 2,
    "tp_size": 16
  },
  "decode": {
    "dp_size": 8,
    "tp_size": 4
  }
}
```

硬件和精度假设：

```text
4 台 Atlas 800 A3
每台 16 卡，每卡 64GiB

P 侧：2 台 = 32 卡
D 侧：2 台 = 32 卡

权重：DeepSeek-V4-Flash-w8a8 / FP8-like 1 byte/param 口径
KVCache：BF16
Context：100K
专家部分：EP 分布，非 DP 复制
非专家部分：DP 复制，TP 切分
```

DeepSeek-V4-Flash 的公开配置包括：`hidden_size=4096`、`num_hidden_layers=43`、`n_routed_experts=256`、`moe_intermediate_size=2048`、`num_experts_per_tok=6`、`num_key_value_heads=1`、`head_dim=512`、`sliding_window=128`、`compress_ratios` 包含 Compress-4 和 Compress-128 混合注意力；vLLM-Ascend 文档也说明 DeepSeek-V4-Flash 采用 mHC、Hybrid Attention、Compress-4-Attention / Compress-128-Attention，并有 Ascend 量化部署路径。 [\[huggingface.co\]](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json), [\[docs.vllm.ai\]](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/DeepSeek-V4-Flash.html)

***

## 1. 模型权重拆分

DeepSeek-V4-Flash 总参数按 **284B**，激活参数约 **13B** 口径估算。这个口径在多处 DeepSeek V4 Flash 资料中被列为 Flash 版本的规模。 [\[wqh011128.github.io\]](https://wqh011128.github.io/blog/2026/05/08/DeepSeekV4.html), [\[pillumina.github.io\]](https://pillumina.github.io/posts/architecture/deepseek-v4-flash/)

### 1.1 Routed expert 权重

每个 routed expert 是 SwiGLU MLP，参数量近似：

```text
expert_params = 3 × hidden_size × moe_intermediate_size
              = 3 × 4096 × 2048
              = 25,165,824
              ≈ 25.17M params
```

总 routed expert 参数：

```text
num_layers = 43
n_routed_experts = 256

expert_total = 43 × 256 × 25.17M
             ≈ 277.03B params
```

w8a8 / FP8-like 按 1 byte/param：

```text
专家权重 ≈ 277.03 GB
        ≈ 258.0 GiB
```

### 1.2 非专家权重

用总参数 284B 扣除 routed experts：

```text
non_expert_total ≈ 284B - 277.03B
                 ≈ 6.97B params
```

即：

```text
非专家权重 ≈ 6.97 GB
          ≈ 6.50 GiB
```

这里的非专家部分包括 attention、embedding、norm、router、shared expert、MTP 相关公共部分等。DeepSeek-V4-Flash 配置中包含 `n_shared_experts=1`、`num_nextn_predict_layers=1`、`num_hash_layers=3` 等额外结构。 [\[huggingface.co\]](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json), [\[huggingface.co\]](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/config.json)

***

## 2. Prefill 侧权重：dp2 × tp16

Prefill 侧：

```text
dp = 2
tp = 16
总卡数 = 2 × 16 = 32
```

### 专家权重

专家权重 EP 分布在全部 32 卡上，不按 DP 复制：

```text
P 单卡专家权重 = 258.0 GiB / 32
              ≈ 8.06 GiB
```

### 非专家权重

非专家权重在 DP 维度复制，在 TP16 内切分：

```text
P 单卡非专家权重 = 6.50 GiB / 16
                ≈ 0.41 GiB
```

### P 单卡总权重

```text
P 单卡总权重 ≈ 8.06 + 0.41
            ≈ 8.47 GiB
```

***

## 3. Decode 侧权重：dp8 × tp4

Decode 侧：

```text
dp = 8
tp = 4
总卡数 = 8 × 4 = 32
```

### 专家权重

专家权重同样 EP 分布到 32 卡：

```text
D 单卡专家权重 = 258.0 GiB / 32
              ≈ 8.06 GiB
```

### 非专家权重

非专家权重在 DP 维度复制，每个 DP 组只用 TP4 切分：

```text
D 单卡非专家权重 = 6.50 GiB / 4
                ≈ 1.62 GiB
```

### D 单卡总权重

```text
D 单卡总权重 ≈ 8.06 + 1.62
            ≈ 9.69 GiB
```

***

## 4. P/D 权重汇总

| 项目        | Prefill：dp2 × tp16 | Decode：dp8 × tp4 |
| --------- | -----------------: | ---------------: |
| Role 内总卡数 |                 32 |               32 |
| 专家权重切分    |              `/32` |            `/32` |
| 非专家权重切分   |              `/16` |             `/4` |
| 单卡专家权重    |     约 **8.06 GiB** |   约 **8.06 GiB** |
| 单卡非专家权重   |     约 **0.41 GiB** |   约 **1.62 GiB** |
| **单卡总权重** |     **约 8.47 GiB** |   **约 9.69 GiB** |

结论：

```text
DeepSeek V4 Flash 比 GLM-5.2 小很多。
同样 dp/tp/EP 口径下：
P 单卡权重约 8.47 GiB
D 单卡权重约 9.69 GiB
```

***

## 5. 100K BF16 KVCache 估算

DeepSeek V4 Flash 和 GLM-5.2 最大不同在这里：它不是简单 MLA latent KV 口径，而是 **Hybrid Attention + Compress-4 / Compress-128**。官方配置里可见 `head_dim=512`、`num_key_value_heads=1`、`sliding_window=128`，并且 `compress_ratios` 里有 `4` 和 `128` 交替出现；vLLM-Ascend 文档也明确说 V4 Flash 通过 Compress-4-Attention 和 Compress-128-Attention 提升长上下文效率。 [\[huggingface.co\]](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json), [\[docs.vllm.ai\]](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/DeepSeek-V4-Flash.html)

我给两个口径：

***

### 5.1 保守上界：按普通 MQA 全量 KV 算

如果不考虑压缩，按 MQA：

```text
KV = seq_len × num_layers × 2 × head_dim × bytes
```

代入：

```text
seq_len = 100000
num_layers = 43
head_dim = 512
K + V = 2
BF16 = 2 bytes
```

得到：

```text
KV = 100000 × 43 × 2 × 512 × 2
   = 8,806,400,000 bytes
   ≈ 8.81 GB
   ≈ 8.20 GiB
```

这个是 **不考虑 V4 压缩注意力收益的保守上界**。

***

### 5.2 更接近 V4 Flash 的压缩 KV 口径

按配置里的 `compress_ratios` 粗略估算：

```text
前 43 层中：
ratio = 0    约 2 层
ratio = 4    约 21 层
ratio = 128  约 20 层
```

简化估算：

```text
ratio=0 层：按 full 100K 保存
ratio=4 层：远程 KV 约 T/4
ratio=128 层：远程 KV 约 T/128
再加 sliding_window=128 的近邻窗口
```

得到 100K BF16 KVCache 约：

```text
压缩 KV ≈ 1.53 GB
        ≈ 1.42 GiB / request / card
```

所以 DeepSeek V4 Flash 的 100K BF16 KVCache 建议看两个数：

```text
工程规划保守上界：8.20 GiB / request / card
更贴近 Hybrid Attention：约 1.42 GiB / request / card
```

vLLM-Ascend 的 DeepSeek-V4-Flash 示例里，单节点 64G × 8 上可以配置 `max-model-len 135168`、`max-num-seqs 16`、`gpu-memory-utilization 0.92`，也侧面说明 V4 Flash 的实际 KV 远小于普通全量 attention 口径。 [\[docs.vllm.ai\]](https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/tutorials/models/DeepSeek-V4-Flash.html), [\[docs.vllm.ai\]](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/DeepSeek-V4-Flash.html)

***

## 6. Prefill 侧容量估算

按 `gpu_memory_utilization=0.95`：

```text
单卡可用目标显存 = 64 × 0.95
                = 60.8 GiB
```

Prefill 单卡权重：

```text
P 单卡权重 ≈ 8.47 GiB
```

扣权重后：

```text
P 可用于 KV + runtime ≈ 60.8 - 8.47
                    ≈ 52.33 GiB
```

### 按 V4 压缩 KV 口径

```text
100K KV ≈ 1.42 GiB / request
理论每 DP 组可放 ≈ 52.33 / 1.42
                 ≈ 36.8 个 100K 请求
```

Prefill 有 dp2：

```text
P 侧理论聚合 ≈ 2 × 36.8
            ≈ 73 个 100K prefill 峰值
```

但 Prefill 还会有 activation、chunked prefill、workspace、KV 传输 buffer、MoE dispatch buffer 等开销，所以不建议按理论值规划。

建议：

```text
P 侧 100K prefill 安全值：每 DP 组 20～24，总体 40～48
P 侧 100K prefill 常规值：每 DP 组 24～32，总体 48～64
```

### 按普通 MQA 上界口径

```text
100K KV 上界 ≈ 8.20 GiB / request
理论每 DP 组可放 ≈ 52.33 / 8.20
                 ≈ 6.38
```

Prefill 聚合：

```text
P 侧上界口径理论 ≈ 12 个 100K 请求
```

***

## 7. Decode 侧容量估算

Decode 单卡权重：

```text
D 单卡权重 ≈ 9.69 GiB
```

扣权重后：

```text
D 可用于 KV + runtime ≈ 60.8 - 9.69
                    ≈ 51.11 GiB
```

### 按 V4 压缩 KV 口径

```text
100K KV ≈ 1.42 GiB / request
理论每 DP 组可放 ≈ 51.11 / 1.42
                 ≈ 35.9 个 100K 请求
```

Decode 有 dp8：

```text
D 侧理论聚合 ≈ 8 × 35.9
            ≈ 287 个 100K decode 请求
```

但这个是纯容量理论值，实际 decode 还要考虑：

* NPU graph / graph cache
* HCCL buffer
* MoE dispatch / combine buffer
* D 侧调度开销
* P→D KV 接收 buffer
* allocator 碎片
* vLLM block manager 管理损耗
* decode batch 下的实际 token/s 约束

所以建议：

```text
D 侧 100K decode 安全值：每 DP 组 20～24，总体 160～192
D 侧 100K decode 常规值：每 DP 组 24～32，总体 192～256
D 侧 100K decode 激进值：每 DP 组 32～35，总体 256～280
```

### 按普通 MQA 上界口径

```text
100K KV 上界 ≈ 8.20 GiB / request
理论每 DP 组可放 ≈ 51.11 / 8.20
                 ≈ 6.23
```

Decode 聚合：

```text
D 侧上界口径理论 ≈ 8 × 6.23
                ≈ 49 个 100K 请求
```

如果你想保守上线，可以用这个上界作为兜底规划。

***

## 8. 最终汇总表

| 项目                               |                     Prefill |                      Decode |
| -------------------------------- | --------------------------: | --------------------------: |
| 节点数                              |                      2 台 A3 |                      2 台 A3 |
| 卡数                               |                        32 卡 |                        32 卡 |
| 并行配置                             |                  dp2 × tp16 |                   dp8 × tp4 |
| 专家权重切分                           |                       `/32` |                       `/32` |
| 非专家权重切分                          |                       `/16` |                        `/4` |
| 单卡专家权重                           |                **8.06 GiB** |                **8.06 GiB** |
| 单卡非专家权重                          |                **0.41 GiB** |                **1.62 GiB** |
| **单卡总权重**                        |                **8.47 GiB** |                **9.69 GiB** |
| 100K BF16 KV，V4 压缩口径             | **约 1.42 GiB/request/card** | **约 1.42 GiB/request/card** |
| 100K BF16 KV，普通 MQA 上界           | **约 8.20 GiB/request/card** | **约 8.20 GiB/request/card** |
| `gpu_memory_utilization=0.95` 可用 |                    60.8 GiB |                    60.8 GiB |
| 扣权重后余量                           |                   52.33 GiB |                   51.11 GiB |
| V4 压缩口径理论每 DP 组                  |                        约 36 |                        约 35 |
| V4 压缩口径建议每 DP 组                  |                       20～32 |                       20～32 |
| 聚合建议并发                           |                       40～64 |                     160～256 |
| MQA 上界口径理论聚合                     |                        约 12 |                        约 49 |

***

## 9. 和 GLM-5.2 的直观对比

| 项目              |            GLM-5.2 |             DeepSeek V4 Flash |
| --------------- | -----------------: | ----------------------------: |
| 总参数             |             约 744B |                        约 284B |
| 层数              |                 78 |                            43 |
| hidden size     |               6144 |                          4096 |
| routed experts  |                256 |                           256 |
| 每 token 激活专家    |                  8 |                             6 |
| P 单卡权重，dp2/tp16 |         约 22.2 GiB |                    约 8.47 GiB |
| D 单卡权重，dp8/tp4  |         约 25.6 GiB |                    约 9.69 GiB |
| 100K BF16 KV    | GLM MLA 约 8.37 GiB | V4 压缩口径约 1.42 GiB，上界 8.20 GiB |

一句话：

> **DeepSeek V4 Flash 在 A3 64G × 16、P: dp2/tp16、D: dp8/tp4 下，权重压力非常小，真正要重点验证的是 V4 Hybrid Attention 的 KV 实现口径、block 管理开销和 P/D KV 迁移效率。**

***

## 10. 建议起步配置

如果你按 DeepSeek V4 Flash 的压缩 KV 实现来跑，可以先从比较稳的值开始：

```text
Prefill:
  dp_size = 2
  tp_size = 16
  max_num_seqs = 32 ~ 48
  max_num_batched_tokens = 8192 起步，可逐步拉到 16384

Decode:
  dp_size = 8
  tp_size = 4
  max_num_seqs = 128 起步
  稳定后尝试 192 / 256
```

如果你担心当前 vLLM-Ascend 的实现未完全吃到压缩 KV 收益，按保守上界起步：

```text
Prefill:
  max_num_seqs = 8 ~ 12

Decode:
  max_num_seqs = 32 ~ 48
```

最终还是建议以启动日志里的这几个值校准：

```text
available KV cache memory
num_gpu_blocks
maximum concurrency
block_size
实际每个 100K 请求占用 block 数
```

结论版：

```text
DeepSeek V4 Flash，A3 64G×16，2P+2D：

P: dp2 × tp16
  单卡权重 ≈ 8.47 GiB

D: dp8 × tp4
  单卡权重 ≈ 9.69 GiB

100K BF16 KV：
  V4 Hybrid Attention 压缩口径 ≈ 1.42 GiB/request/card
  普通 MQA 保守上界 ≈ 8.20 GiB/request/card

推荐起步：
  P 侧 40～48 个 100K prefill 峰值
  D 侧 160～192 个 100K decode 常驻

压测稳定后：
  P 侧尝试 64
  D 侧尝试 256
```
