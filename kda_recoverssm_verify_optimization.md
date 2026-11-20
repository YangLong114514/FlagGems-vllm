# KDA RecoverSSM Verify 优化记录

本文记录 `kda_recoverssm_verify` 在 Kimi K3 生产 Shape 上已经验证的优化方法、实际效果和失败尝试。性能数据均来自 NVIDIA H800，基线为 vLLM 0.34.0 的 Triton kernel，FlagGems 输入为 BF16 activation + FP32 checkpoint state。

## 1. 算子特征与优化约束

kernel 按 `(V tile, request, head)` 并行，在每个 program 内沿 speculative window 串行更新 `[BV, K]` state。生产 Shape 固定 `K=V=128`，窗口长度分别为：

- Q2：`SPEC_QUERY_LEN=2`，对应 MTP-1。
- Q4：`SPEC_QUERY_LEN=4`，对应 spec-3。
- Q8：`SPEC_QUERY_LEN=8`，对应最大 8 token 验证窗口。

每个 token 需要完成 state decay、`state @ k`、rank-1 state update 和 `state @ q`。Q8 的串行依赖最明显，因此减少窗口循环内的冗余工作更容易产生收益。

优化时必须同时检查寄存器、spill 和目标执行模式。降低指令数不一定降低延迟：寄存器增加可能降低 occupancy，而 eager autotune 选出的配置不一定适合 CUDA Graph。

## 2. 已保留的优化

### 2.1 Q8 的 `dt_bias` 循环外提

原实现为每个 `(request, head, V tile)` program 在每个 token 上重复加载相同的 `dt_bias[K]`。Q8 路径现在只在进入 token 循环前加载一次，Q2/Q4 保留原路径。

代码位置：`src/flaggems_vllm/ops/kda_recoverssm_verify.py` 中 `SPEC_QUERY_LEN == 8` 的编译期分支。

固定候选配置并经过完整生产 benchmark 验证后的效果：

| 模式 | 修改前 Gems 延迟 | 修改后 Gems 延迟 | 延迟改善 | SpeedUp 变化 |
| --- | ---: | ---: | ---: | ---: |
| Kernel | 124.576 µs | 122.688 µs | 1.5% | 0.900 → 0.915 |
| CUDA Graph | 57.963 µs | 56.463 µs | 2.6% | 1.021 → 1.048 |

该改动的 Q8 编译结果没有 register spill。功能测试 92 项全部通过，包括 ragged window、NULL checkpoint、BF16/FP16 activation 和 FP32/BF16 state 组合。

没有全局外提的原因：

- Q4 固定配置下寄存器从 64 增至 79，延迟回退约 4%。
- Q2 固定配置下虽有约 2%–3% 收益，但完整 eager autotune 会误选 BV16，导致 CUDA Graph 性能回退。因此最终只保留收益稳定的 Q8 路径。

### 2.2 V 方向 tiling 与 autotune

当前 kernel 通过 `BV`、`num_warps`、`num_stages` 调节 state tile、并行度和寄存器占用，配置位于 `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml`。vLLM 基线固定 `BV=32, num_warps=4, num_stages=2`，FlagGems 按 `(K, V, SPEC_QUERY_LEN)` 及输入 dtype 选择配置。

在本轮代表性 Shape 上，最终常用配置为：

| 窗口 | BV | num_warps | num_stages |
| --- | ---: | ---: | ---: |
| Q2 | 64 | 8 | 2 |
| Q4 | 32 | 8 | 2 |
| Q8 | 64 | 8 | 2 |

更大的 BV 可以减少不同 V tile 对 `q/k/raw_g` 的重复加载和归一化计算，但并非越大越好。BV128 的编译结果达到 255 registers/thread，只有很低的 occupancy，性能明显下降。调优的目标应是实际延迟，而不是单纯减少 CTA 或重复指令。

## 3. 当前生产 Shape 性能

SpeedUp 定义为 `vLLM latency / FlagGems latency`，数值越大越好。Kernel 模式使用 5 次 warmup、25 次 device-event 采样并取中位数；CUDA Graph 使用 `rep=100` 的 graph replay 中位数。延迟单位为 ms。

| Case | Kernel vLLM | Kernel Gems | Kernel SpeedUp | Graph vLLM | Graph Gems | Graph SpeedUp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k3_tp8_mtp1_bs32 | 0.067680 | 0.081408 | 0.831 | 0.015342 | 0.015409 | 0.996 |
| k3_tp8_mtp1_bs128 | 0.116064 | 0.125344 | 0.926 | 0.063802 | 0.060009 | 1.063 |
| k3_tp8_mtp1_bs256 | 0.168640 | 0.178976 | 0.942 | 0.116957 | 0.113014 | 1.035 |
| k3_tp8_spec3_bs32 | 0.082304 | 0.097952 | 0.840 | 0.029415 | 0.032429 | 0.907 |
| k3_tp8_spec3_bs128 | 0.180448 | 0.188960 | 0.955 | 0.129433 | 0.124164 | 1.042 |
| k3_tp8_win8_bs32 | 0.112256 | 0.122688 | 0.915 | 0.059176 | 0.056463 | 1.048 |
| k3_tp4_mtp1_bs64 | 0.116320 | 0.125920 | 0.924 | 0.065053 | 0.060078 | 1.083 |
| k3_tp2_mtp1_bs64 | 0.171808 | 0.178208 | 0.964 | 0.118586 | 0.112613 | 1.053 |
| k3_tp1_mtp1_bs8 | 0.089760 | 0.097824 | 0.918 | 0.038381 | 0.033392 | 1.149 |
| k3_tp1_spec3_bs16 | 0.181952 | 0.190272 | 0.956 | 0.129974 | 0.124288 | 1.046 |

这张表是当前实现相对 vLLM 的最终结果，不能全部归因于某一项代码改动。Q8 外提的独立增量见第 2.1 节。

## 4. 已验证无效并回退的方案

| 方案 | 结果 | 处理 |
| --- | --- | --- |
| 扩容到 36 组 autotune 候选 | 完整生产集没有稳定收益，Graph 多数变化不足约 1% | 回退到 9 组候选 |
| 全局外提 `dt_bias` | Q4 因寄存器增加回退；Q2 受 autotune 配置切换影响出现 Graph 回退 | 仅保留 Q8 |
| `static_range` 改动态 token 循环 | 慢约 6%–22% | 回退 |
| 交换 batch/head 的 CTA 调度顺序 | 基本无收益，部分 Shape 略慢 | 回退 |
| 将两次 state-vector 归约直接替换为 `tl.dot` | 慢约 3–50 倍；IEEE 路径出现大量 spill | 回退 |
| 使用 BV128 消除跨 V tile 的重复计算 | 255 registers/thread，occupancy 受限，性能下降 | 不作为默认配置 |

这些结果说明，后续优化不应只做局部指令替换或盲目扩大 tile。每次试验都应固定一个主要变量，并分别验证 Kernel 和 CUDA Graph。

## 5. 后续优化方向

Q8 可继续评估窗口级公式重构：利用 cumulative decay 和窗口内的 `k_i·k_j`、`q_i·k_j` 关系，一次求出多个 token 的 correction，再由初始 checkpoint 和 correction 组合计算输出，避免逐 token 物化完整中间 state。

该方向可能减少串行 state 操作，但需要重新处理数值精度、矩阵布局转换和寄存器压力。直接替换为 `tl.dot` 已证明无效，只有完整窗口级实现能够摊薄这些成本时才值得继续。它目前仍是待验证方向，不属于已交付优化。

## 6. 复现命令

当前容器需要把 CUDA compat library 放在 `LD_LIBRARY_PATH` 前面。以下命令使用 GPU 4，并用内存 SQLite 避免持久化调优缓存影响结果。

```bash
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

python -m pytest -q tests/test_kda_recoverssm_verify.py

export FLAGGEMS_DB_URL=sqlite:///:memory:
python -m pytest -q -s benchmark/test_kda_recoverssm_verify.py \
  --level core --mode kernel --iter 1 --warmup 1
python -m pytest -q -s benchmark/test_kda_recoverssm_verify.py \
  --level core --mode cudagraph --iter 100 --warmup 1
```

当前环境没有 `ncu/nsys`。现有结论来自功能对拍、编译后的 register/spill 数据、固定配置对照和完整生产 benchmark；后续若进行更深入的 kernel 重构，应补充硬件 profiler 数据。
