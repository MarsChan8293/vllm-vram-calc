我拉到了 GLM-5 的 config、DSA/MLA 架构细节和 vllm-ascend 的部署脚本。先让 Python 帮我把数算准，再解读。# GLM-5 (744B MoE) 在 vllm-ascend 上的显存/并发估算 [\[github.com\]](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md), [\[deepwiki.com\]](https://deepwiki.com/zai-org/GLM-5/1.1-model-architecture), [\[github.com\]](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/models/glm_moe_dsa/configuration_glm_moe_dsa.py), [\[zhuanlan.zhihu.com\]](https://zhuanlan.zhihu.com/p/2026384496501473622)

以下针对官方教程里的示例场景：**`glm-5-w4a8` 部署在 1 台 Atlas 800 A3（64GB × 16 NPU），TP=16 + EP，`gpu-memory-utilization=0.95`**。 [\[docs.vllm.com.cn\]](https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/GLM5.html), [\[github.com\]](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md)

## 关键模型参数（GLM-5 config.json）

| 参数                    | 值              | 说明                                                                                      |
| --------------------- | -------------- | --------------------------------------------------------------------------------------- |
| `num_hidden_layers`   | **80**         | 前 3 层 Dense，后 77 层 MoE+DSA [\[zhuanlan.zhihu.com\]](https://zhuanlan.zhihu.com/p/2026384496501473622)   |
| `hidden_size`         | 6144           | "瘦高"设计 [\[zhuanlan.zhihu.com\]](https://zhuanlan.zhihu.com/p/1933911723917706854)                       |
| `num_attention_heads` | 64             | 相比 GLM-4.5 从 96 减到 64                                                                   |
| QK / V head\_dim      | 192 / 256      | 非对称设计 [\[zhuanlan.zhihu.com\]](https://zhuanlan.zhihu.com/p/2026384496501473622)                        |
| `kv_lora_rank`        | **512**        | MLA 压缩秩 [\[github.com\]](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/models/glm_moe_dsa/configuration_glm_moe_dsa.py)                      |
| `qk_rope_head_dim`    | 64             | RoPE 解耦部分                                                                               |
| `n_routed_experts`    | 256            | 每 token 激活 8 个 + 1 共享 [\[github.com\]](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/models/glm_moe_dsa/configuration_glm_moe_dsa.py)        |
| 总参数 / 激活              | **744B / 40B** | MoE 稀疏率 5.4% [\[deepwiki.com\]](https://deepwiki.com/zai-org/GLM-5/1.1-model-architecture)                 |
| 关键创新                  | **MLA + DSA**  | DeepSeek Sparse Attention 集成 [\[aissential.tech\]](https://aissential.tech/articles/aa342662-dc03-432c-933a-0b8e18715536) |

***

## 一、每卡权重占用

* **w4a8 全模型权重**：≈ **494 GB**（BF16 原始 1488 GB，量化后按 INT4 权重 + FP16 group scale + 部分 BF16 sensitive layers 估算）
* **每卡权重**：≈ **32.4 GB**（TP=16 + EP=16 均分 MoE 专家，考虑约 5% 复制冗余：LayerNorm、router、embedding 等未切分参数）

这也是为什么官方选 64GB × 16 的 A3，而不敢用 A2 系列（32GB × 8）。

***

## 二、MLA KV Cache 单 token 尺寸

这是 MLA 相比传统 GQA 的**核心节省**：

$$
\text{KV/token/layer} = \text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim} = 512 + 64 = 576 \text{ elems}
$$

* **每层每 token**：576 × 2B (BF16) = **1152 B**
* **全 80 层每 token**：**≈ 90 KB**
* **40K 上下文单序列**：40000 × 90KB ≈ **3.69 GB**
* **200K 上下文单序列**：≈ **18.4 GB**

⚠️ **关键陷阱**：MLA + TP（无 DP-Attn）时，**KV cache 在 16 张卡上是完全复制的**——这是你 6/27 提到的老问题。所以上面 3.69 GB 是**每张卡**都要付的代价，不能除以 16。

DSA 的 IndexShare 每 4 层共享一个 indexer，附加约 4.8 KB/token 的索引缓存，占比 \~5%，可以忽略在余量里。 [\[github.com\]](https://github.com/zai-org/GLM-5)

***

## 三、每卡显存分配一览（40K 场景）

| 组件                              |              每卡占用 | 说明                            |
| ------------------------------- | ----------------: | ----------------------------- |
| HBM 总量                          |           64.0 GB | 单张 A3 NPU                     |
| × `gpu-memory-utilization=0.95` |       **60.8 GB** | vLLM 可用上限                     |
| — 权重（w4a8, TP+EP 切分）            |           32.4 GB |                               |
| — HCCL / NPU 上下文                |            2.0 GB | `HCCL_BUFFSIZE=200`           |
| — NPU Graph / 编译缓存              |            1.5 GB | `enable_npugraph_ex`          |
| — MoE All-to-All buffer         |            1.5 GB | EP 通信                         |
| — Chunked prefill buffer        |            1.0 GB | `enable-chunked-prefill`      |
| — 激活峰值（4096 tokens）             |            0.2 GB | `max-num-batched-tokens=4096` |
| — 框架/调度/元数据                     |            1.0 GB |                               |
| **= KV Cache 可用池**              | **≈ 21.2 GB / 卡** |                               |

***

## 四、40K 上下文的最大并发数

$$
N_{\max} = \frac{21.2 \text{ GB}}{3.69 \text{ GB/req}} \approx 5.7
$$

**建议`--max-num-seqs 4`**（留 15–20% 余量应对突发和 prefix cache），最多 5，别超 6。

### 不同上下文长度下的并发上限

| Context |      单请求 KV |  每卡理论并发上限 | 建议 `--max-num-seqs` |
| ------- | ----------: | --------: | ------------------: |
| 8K      |     0.74 GB |      \~28 |                  20 |
| 16K     |     1.47 GB |      \~14 |                  10 |
| **40K** | **3.69 GB** | **\~5.7** |               **4** |
| 128K    |     11.8 GB |     \~1.8 |                   1 |
| 200K    |     18.4 GB |     \~1.1 |                   1 |

***

## 五、对照官方脚本的"矛盾"

官方脚本用的是 `--max-num-seqs 8 --max-model-len 200000`。按上面算法： [\[docs.vllm.com.cn\]](https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/GLM5.html)

$$
8 \times 200K \times 90\text{KB} = 147.5 \text{ GB/卡}
$$

远超 21.2 GB 的池子。官方能跑通的原因有三：

1. **`max_model_len` 是上限，不是实际值**——实际请求多为几千到几万 token，`max_num_seqs=8` 只是并发上限，两者不会同时打满。
2. **Chunked prefill + prefix caching** 会做动态调度，长请求排队而不是并发。
3. **vLLM 的 KV pool 是共享的**——8 个并发的实际长度加和只要不超过池子容量就行。

所以官方配置更像"最坏情况可 8 并发短序列 + 单个 200K 长序列"，不是"8 × 200K 齐飞"。

***

## 六、想提升 40K 并发？三个方向

### 方向 A：改成 DP-Attention（MLA 场景最直接）

将 `--data-parallel-size 16 --tensor-parallel-size 1`（或 DP=2/TP=8 混合）+ `--enable-expert-parallel`：

* **KV cache 按 DP 切分而不是复制** → 每卡 KV 需求降为 1/DP
* 40K 并发上限可从 \~6 提升到 \~90+
* 代价：Attention 无 TP 加速，单请求延迟略升；需要 vllm-ascend ≥ v0.20 且模型支持 DP-Attn

### 方向 B：启用 KV Cache 量化

加 `--kv-cache-dtype int8`（若 vllm-ascend 已支持 MLA 的 INT8 KV）：

* KV 尺寸减半 → 40K 并发从 6 → 12
* 精度损失通常 <1%

### 方向 C：PD 分离部署

Prefiller 专注 40K prompt 的 prefill 吞吐，Decoder 卡 KV 池不再被 prefill 激活挤占：

* vllm-ascend v0.17.0rc1+ 官方支持 GLM-5 的 PD 分离 [\[docs.vllm.ai\]](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html)
* Decoder 侧 KV 池利用率能从 \~50% 提升到 80%+
* 但 GLM-5 的 KV 传输量大（40K × 90KB = 3.6 GB/req），需要 200GbE 参数面网络才够

***

## 汇总一句话

**40K 上下文、TP=16 单机 A3、w4a8 场景下，每卡权重 \~32 GB、每卡 KV 池 \~21 GB、单请求 KV \~3.7 GB、稳态并发建议 4（理论上限 \~6）**。想要显著提升并发，最有效的是切 **DP-Attention**（数量级提升）或叠加 **INT8 KV**（2×）。

要不要我帮你把这套算法写成可复用的 Python 脚本（支持切换 TP/DP/EP、量化方案、Context 长度），或者继续对照 GLM-5.2 的 IndexShare（每 4 层共享 indexer，1M context 下 FLOPs 降 2.9×）算一份 1M 上下文的账？ [\[github.com\]](https://github.com/zai-org/GLM-5)
