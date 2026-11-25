# KDA RecoverSSM Verify 优化记录

第 1–7 节记录 NVIDIA H800 上的历史实验，基线为 vLLM 0.34.0 Triton kernel，输入为 BF16 activation 和 FP32 checkpoint state。第 3 节的生产性能表最后更新于原分支 `feat/kda-recoverssm-verify` 的 `d6c8e1a`，对应当前分支 rebase 后的 `209e5c7`。第 8 节记录此后 `feat/metax-kda-recoverssm-verify-baseline` 截至 `d99f996` 的开发。两个提交之间的产品 verify kernel 源文件相同，后续改动集中在测试、基线和性能采集。

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

### 2.2 token 有效性改用标量控制流

原实现为窗口内每个 token 执行 masked load、完整 recurrent 计算、masked store 和 `tl.where`，即使 `token_offset >= query_len` 也不会跳过计算。现在保留 `tl.static_range` 静态展开，但用 `if token_offset < query_len` 包围整个 token body：

- 满窗口生产路径不再需要 `token_valid` 对每次加载和存储进行谓词控制，也不再执行 state 选择。
- ragged window 直接跳过剩余 token，不再执行无效 recurrent 计算。
- 不使用已经验证会变慢的动态 token 循环。

固定配置的 CUDA Graph 对照结果：

| Shape | 修改前 | 修改后 | 延迟改善 |
| --- | ---: | ---: | ---: |
| Q2, B32 | 15.436 µs | 14.432 µs | 6.5% |
| Q2, B256 | 113.035 µs | 104.701 µs | 7.4% |
| Q4, B128 | 123.734 µs | 122.148 µs | 1.3% |
| Q8, B32 | 56.009 µs | 49.787 µs | 11.1% |

完整 autotune 和生产 benchmark 中，CUDA Graph 的 10 组生产 Shape 全部提升 5.8%–19.6%，Kernel 模式也全部提升。Q8 固定配置的编译结果从 116 降至 104 registers/thread，无 spill。92 项功能测试全部通过，覆盖 ragged window 和 NULL checkpoint。

### 2.3 Q4/Q8 精确 tile 消除边界掩码

生产 Shape 满足 `K=BK=128`、`V=128`，并且所有 BV 候选都整除 V，因此 K/V 边界掩码恒为真。Q4/Q8 在该条件下将 `mask_k`、`mask_v` 和 `mask_state` 折叠为常量 `True`，删除实际加载和存储的边界判断。Q2 以及非规则维度仍保留 masked general path。

完整 autotune 和生产 benchmark 的 CUDA Graph 效果：

| Shape | 修改前 | 修改后 | 延迟改善 |
| --- | ---: | ---: | ---: |
| Q4, B32 | 26.066 µs | 24.955 µs | 4.3% |
| Q4, B128 | 102.060 µs | 97.507 µs | 4.5% |
| Q4, TP1 B16 | 102.108 µs | 97.621 µs | 4.4% |
| Q8, B32 | 50.123 µs | 48.651 µs | 2.9% |

Q2 的临时无掩码变体回退约 3%–5%，因此没有启用。除 92 项功能测试外，还使用 Q4、`head_dim=96` 的 ragged 输入验证了 masked fallback，结果与 vLLM 一致。

### 2.4 V 方向 tiling 与 autotune

当前 kernel 通过 `BV`、`num_warps`、`num_stages` 调节 state tile、并行度和寄存器占用，配置位于 `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml`。vLLM 基线固定 `BV=32, num_warps=4, num_stages=2`，FlagGems 按 `(K, V, SPEC_QUERY_LEN, WORKLOAD_BUCKET)` 及输入 dtype 选择配置。`WORKLOAD_BUCKET=next_power_of_2(batch × heads)`，用于避免同一窗口的不同并行规模被迫复用首个 Shape 的配置。

更大的 BV 可以减少不同 V tile 对 `q/k/raw_g` 的重复加载和归一化计算，但并非越大越好。guarded token 改动前的 BV128 编译结果曾达到 255 registers/thread，occupancy 和性能明显下降；改动后的最优配置会随 kernel 和并行规模改变，不能为同一 window 类别假定永久固定的配置。调优的目标应是目标执行模式的实际延迟，而不是单纯减少 CTA 或重复指令。

### 2.5 真实布局、启动开销与配置稳定性

已按 vLLM 0.34.0 的实际调用补齐 packed QKV view、二维状态索引表的第一列和预分配 `out`，并在 B256 额外测试打散的 checkpoint slot。测量结果：

| Shape/布局 | Native eager | Gems eager | Eager SpeedUp | Native Graph | Gems Graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2/B32，原布局 | 66.69 µs | 80.32 µs | 0.830 | 15.01 µs | 14.47 µs |
| Q2/B32，完整真实布局 | 63.71 µs | 75.94 µs | 0.839 | 15.42 µs | 14.54 µs |
| Q2/B256，完整真实布局 | 165.22 µs | 168.42 µs | 0.981 | 115.83 µs | 104.93 µs |
| Q2/B256，真实布局并打散 slot | 164.45 µs | 166.85 µs | 0.986 | 115.59 µs | 104.35 µs |
| Q8/B32，完整真实布局 | 106.30 µs | 103.14 µs | 1.031 | 58.03 µs | 41.45 µs |

packed QKV、列索引和 slot 顺序对 Graph 设备延迟没有明显负面影响。预分配 `out` 使 Q2/B32 的 Gems eager 延迟下降约 4.4 µs，但仍未达到 0.9。真实布局下 native/Gems 主机提交中位数约为 45.69/59.10 µs，Gems 仍多约 13.4 µs，因此剩余 eager 差距主要在公共 `LibEntry` 路径。按当前设计要求，没有恢复已经移除的 fast-launch。

Q8 同轮比较全部 9 组候选后，BV32/2 warps/1 stage 以约 41.5 µs 明确优于其他候选；三个独立空缓存进程均选择该配置，Graph 延迟为 41.135、41.140、41.133 µs，不需要额外的 Q8 配置分桶。

Q2 分桶前的首次调优顺序实验：

| 首次调优 Shape | 选中配置 | B32 Graph | B128 Graph | B256 Graph |
| --- | --- | ---: | ---: | ---: |
| B32 | BV64/8 warps/2 stages | 14.110 µs | 55.840 µs | 104.985 µs |
| B256 | BV128/8 warps/1 stage | 14.624 µs | 55.524 µs | 101.392 µs |

不同规模确实需要不同配置，因此保留 workload 分桶，使各桶独立调优，避免 B256 的收益以 B32 回退为代价。完整生产 benchmark 中，分桶后的 Graph 延迟变化为：

- Q2/B256：104.888 → 101.551 µs，改善 3.2%。
- Q4/B128：97.507 → 91.930 µs，改善 5.7%。
- Q4/TP1/B16：97.621 → 91.806 µs，改善 6.0%。
- Q8/B32：48.651 → 41.500 µs，改善 14.7%，与固定最佳配置及三轮稳定性结果一致。
- Q2/B32 和 Q4/B32：分别为 14.108、24.843 µs，基本持平。

92 项功能测试通过。仅应用分桶时，Kernel 模式两轮复测中 Q2/B256 SpeedUp 为 0.999/1.000，Q8 为 1.017/1.000；小 batch eager 仍有波动，Q2/B32 为 0.820/0.825，Q4/B32 为 0.890/0.883。对应 Graph 没有回退，该现象促成了第 2.6 节的主机提交优化。

### 2.6 已回退的主机启动试验

针对 Q2/Q4 小 batch 的 eager 门槛，NCU 显示设备端已经领先：Q2/B32 的 native/Gems kernel 为 24.67/22.14 µs，Q4/B32 为 43.07/34.98 µs，均无 spill。cProfile 确认额外开销集中在 `LibEntry` 的参数归类、特化键和启动封装。

曾试验公共 `LibEntry` Python 优化以及算子内 `_TUNED_CONFIGS + JITFunction` 配置复用，均能降低 eager 延迟；但每次直接经过 `LibTuner + JITFunction` 的方案会因重复查询配置而超过 1.5 ms。考虑公共组件审核和算子启动路径的设计要求，这些启动改动均未保留：

- `src/flaggems_vllm/utils/libentry.py` 恢复原实现。
- `kda_recoverssm_verify.py` 删除 `_TUNED_CONFIGS` 和 JIT 绕行，每次调用统一经过原有 `libentry/libtuner`。
- 删除仅为 JIT 配置复用增加的 stream/lower_bound 回归测试，恢复原有 92 项测试。

当前交付不包含 fast-launch、直接 `CudaLauncher`、调优配置缓存或标准 JIT 绕行。上述启动试验的性能仅作为历史记录，不计入第 3 节的当前性能。

回退后曾用旧固定 5/25 口径验证：92 项功能测试通过；Graph 保持设备端收益，而 Q2/B32、Q4/B32 的 eager wrapper 重新低于 0.9。这确认启动试验的收益来自主机路径。当前 benchmark 已删除该自定义计时，统一使用仓库基类；通用 Kernel/Graph 结果见第 3 节，不能与旧 eager wrapper 数值直接比较。

## 3. H800 生产 Shape 性能（`209e5c7`）

SpeedUp 定义为 `vLLM latency / FlagGems latency`，数值越大越好。当前 benchmark 不再自定义统计口径：Kernel 由通用 `Benchmark.get_latency()` 调用 `triton.testing.do_bench`，使用 `Config.warm_up/Config.repetition` 并返回 median；CUDA Graph 由同一基类调用 `do_bench_cudagraph`。下表命令分别为 `--iter 1 --warmup 1` 和 `--iter 100 --warmup 1`，延迟单位为 ms。旧固定 5 warmup/25 device-event 的数据仅作历史记录，不与本表直接比较。

| Case | Kernel vLLM | Kernel Gems | Kernel SpeedUp | Graph vLLM | Graph Gems | Graph SpeedUp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k3_tp8_mtp1_bs32 | 0.027904 | 0.024896 | 1.121 | 0.015030 | 0.014345 | 1.048 |
| k3_tp8_mtp1_bs128 | 0.071152 | 0.063808 | 1.115 | 0.063542 | 0.055640 | 1.142 |
| k3_tp8_mtp1_bs256 | 0.125600 | 0.109152 | 1.151 | 0.116621 | 0.100240 | 1.163 |
| k3_tp8_spec3_bs32 | 0.044224 | 0.035680 | 1.239 | 0.029132 | 0.024699 | 1.179 |
| k3_tp8_spec3_bs128 | 0.138496 | 0.100512 | 1.378 | 0.128520 | 0.091408 | 1.406 |
| k3_tp8_win8_bs32 | 0.076624 | 0.061248 | 1.251 | 0.057600 | 0.041139 | 1.400 |
| k3_tp4_mtp1_bs64 | 0.073008 | 0.064000 | 1.141 | 0.064611 | 0.055763 | 1.159 |
| k3_tp2_mtp1_bs64 | 0.126608 | 0.110144 | 1.149 | 0.117333 | 0.101465 | 1.156 |
| k3_tp1_mtp1_bs8 | 0.046464 | 0.039504 | 1.176 | 0.038134 | 0.031301 | 1.218 |
| k3_tp1_spec3_bs16 | 0.137488 | 0.100288 | 1.371 | 0.129336 | 0.091429 | 1.415 |

这张表是该提交在 H800 上相对同机 vLLM 的实测结果。Q8 外提、guarded token、精确 tile 和 workload 分桶的增量分别见第 2.1、2.2、2.3 和 2.5 节。

## 4. NCU profile 与瓶颈评估

使用 NCU 2025.4.1 对十组生产用例分别采集 native 和 Gems 的 full profile。每次只采集一个目标 kernel，排除 autotune、参考实现和其他框架 kernel。NCU 默认 flush cache 并通过多 pass replay 收集指标，因此下表延迟用于硬件瓶颈分析，不与 CUDA Graph 延迟直接混用。数据采集于 workload 分桶和算子内启动优化之前；其中 Q2/B32、Q4/B32 的设备配置与后续 eager 分析一致，其他 Shape 的最终性能以第 3 节为准。

NCU SpeedUp 为 `native NCU duration / Gems NCU duration`。Occupancy 为 achieved occupancy，Registers 单位为 register/thread。

| Case | Native | Gems | NCU SpeedUp | Gems SM | Gems DRAM | Occupancy | Registers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k3_tp8_mtp1_bs32 | 24.67 µs | 22.14 µs | 1.114 | 48.1% | 35.6% | 33.5% | 80 |
| k3_tp8_mtp1_bs128 | 73.38 µs | 67.04 µs | 1.095 | 62.6% | 48.4% | 35.6% | 80 |
| k3_tp8_mtp1_bs256 | 134.27 µs | 127.49 µs | 1.053 | 67.5% | 50.6% | 36.0% | 80 |
| k3_tp8_spec3_bs32 | 43.07 µs | 34.98 µs | 1.231 | 53.3% | 23.5% | 33.8% | 80 |
| k3_tp8_spec3_bs128 | 150.05 µs | 116.96 µs | 1.283 | 63.6% | 29.3% | 36.0% | 80 |
| k3_tp8_win8_bs32 | 83.01 µs | 73.50 µs | 1.129 | 38.2% | 12.0% | 18.1% | 168 |
| k3_tp4_mtp1_bs64 | 74.08 µs | 67.42 µs | 1.099 | 63.0% | 48.1% | 35.6% | 80 |
| k3_tp2_mtp1_bs64 | 136.00 µs | 126.02 µs | 1.079 | 67.6% | 51.2% | 36.0% | 80 |
| k3_tp1_mtp1_bs8 | 43.46 µs | 37.73 µs | 1.152 | 56.5% | 43.3% | 34.8% | 80 |
| k3_tp1_spec3_bs16 | 149.79 µs | 116.61 µs | 1.285 | 63.7% | 29.3% | 36.0% | 80 |

瓶颈分类：

- Q2：global load 的 long-scoreboard stall 约占 14%–27%，warp reduction 相关的 short-scoreboard stall 约占 14%–17%。大 batch 的 SM、DRAM 和 L2 利用率同时较高，不再是单一资源瓶颈。
- Q4：DRAM 仅约 23%–29%，short-scoreboard stall 约 18%–20%，主要空间在归约和指令调度。当前 80 registers/thread 对应 37.5% 理论 occupancy，没有 spill。
- Q8：当前 NCU 配置为 BV32、2 warps、168 registers/thread，理论/实际 occupancy 仅 18.75%/18.1%。每个 scheduler 平均只有 0.66 个 eligible warp，约 61% 周期没有可发射 warp，no-instruction stall 约 18.5%。DRAM 仅 12%，主要矛盾是低 occupancy、串行 recurrent 依赖和归约延迟。

基于上述数据执行的后续试验结果如下。延迟均为固定配置的 CUDA Graph 延迟，与 NCU replay 延迟口径不同。

| 试验 | 当前延迟 | 试验延迟 | 编译结果 | 结论 |
| --- | ---: | ---: | --- | --- |
| Q8：BV32/2 warps 改 4 warps | 41.06 µs | 48.33 µs | 168 → 95 registers，无 spill | 回退 |
| Q8：`maxnreg=128` | 41.06 µs | 48.39 µs | 168 → 128 registers，无 spill | 回退 |
| Q8：BV64/8 warps | 41.06 µs | 48.57 µs | 80 registers，无 spill | 回退 |
| Q2 B32：`maxnreg=64` | 14.10 µs | 15.22 µs | 64 registers，2 次 spill | 回退 |
| Q2 B256：`maxnreg=64` | 104.81 µs | 105.42 µs | 64 registers，2 次 spill | 回退 |
| Q4 B32：`maxnreg=64` | 24.86 µs | 28.47 µs | 64 registers，14 次 spill | 回退 |
| Q4 B128：`maxnreg=64` | 97.40 µs | 102.97 µs | 64 registers，14 次 spill | 回退 |

提高理论 occupancy 并不等于降低延迟。Q8 的当前 2-warp 实现虽然 occupancy 低，但归约和指令调度更高效；Q2/Q4 的寄存器限制则直接引入 spill，抵消了 occupancy 收益。

随后实现了 Q8 窗口级公式原型：一次求出窗口内所有 correction，再由初始 state 和 correction 组合计算输出，避免逐 token 物化完整中间 state。最佳原型使用 BV32/2 warps，延迟 104.90 µs，而当前为 40.89 µs；需要 16KB shared memory，达到 255 registers/thread 并出现 40 次 spill。TF32 路径的 correction 最大绝对误差约 4.08e-5，也还需要进一步验证。该原型不保留，现有逐 token kernel 继续作为当前最优实现。

原始报告和汇总数据保存在开发机的 `/workspace/kda_ncu_reports/`，包括 20 份 `.ncu-rep`、`summary.csv` 和 `summary.json`。

## 5. 已验证无效并回退的方案

| 方案 | 结果 | 处理 |
| --- | --- | --- |
| 扩容到 36 组 autotune 候选 | 完整生产集没有稳定收益，Graph 多数变化不足约 1% | 回退到 9 组候选 |
| 全局外提 `dt_bias` | Q4 因寄存器增加回退；Q2 受 autotune 配置切换影响出现 Graph 回退 | 仅保留 Q8 |
| `static_range` 改动态 token 循环 | 慢约 6%–22% | 回退 |
| 交换 batch/head 的 CTA 调度顺序 | 基本无收益，部分 Shape 略慢 | 回退 |
| 将两次 state-vector 归约直接替换为 `tl.dot` | 慢约 3–50 倍；IEEE 路径出现大量 spill | 回退 |
| 改写输出代数关系以打破更新后 state 的依赖 | 增加额外归约，慢约 3%–12% | 回退 |
| state 使用 `.cg` 或 activation 使用 `evict_last` | 基本无收益或略慢 | 回退 |
| 使用 BV128 消除跨 V tile 的重复计算 | 原 kernel 达到 255 registers/thread，occupancy 受限，性能下降 | 不作为固定默认配置 |

这些结果说明，后续优化不应只做局部指令替换或盲目扩大 tile。每次试验都应固定一个主要变量，并分别验证 Kernel 和 CUDA Graph。

## 6. 后续优化方向

本轮 NCU 导向的 occupancy 调整和窗口级公式原型均未通过性能验收。后续不应直接重复这些配置；若重新启动 Q8 深度优化，需要能显著降低窗口内 pairwise 计算、布局转换、shared memory 和寄存器开销的新实现，并先解决 TF32 correction 精度，而不是继续扩大现有原型的 tile 或 warps。

在 workload 分桶和公共启动优化达标后，继续实施了建议文档中的剩余 kernel 实验。满窗口特化使用设备端 `query_len == SPEC_QUERY_LEN` 分支，满窗路径去掉逐 token 判断，ragged 沿用原路径。固定配置 Graph 结果：

| Shape | 当前 | 满窗特化 | 寄存器/spill | 结论 |
| --- | ---: | ---: | --- | --- |
| Q4/B32 | 24.291 µs | 25.051 µs | 80 / 0 | 回退 |
| Q4/B128 | 91.452 µs | 97.467 µs | 128 / 0 | 回退 |
| Q8/B32 | 41.450 µs | 41.575 µs | 168 / 0 | 回退 |

双路径静态展开增加了代码开销，去掉逐 token 判断没有形成收益。归约布局检查显示，Q4/Q8 的 state tile 均为 `threadsPerWarp=[1,32]`、`warpsPerCTA=[N,1]`，已经是一条 warp 完成一行 K 维归约。TTGIR 中剩余的 layout conversion 主要服务于输出和 records 存储，不存在需要强制单 warp 行归约来消除的额外通信，因此不再实施该改写。本轮没有新增生产代码改动，第 3 节性能表保持有效。

针对 Q2/B32 eager 偶尔接近 0.89 的边界波动，补充 12 轮配对复测：默认 `out` 的 SpeedUp 为 0.911–0.939，中位数 0.924；预分配 `out` 为 0.907–0.931，中位数 0.919。两种方式的绝对延迟均下降，但预分配没有显著扩大相对余量。默认 `out` 时 native/Gems 主机提交中位数为 48.72/56.43 µs，Gems 仍多约 7.7 µs，而 NCU 设备 kernel 已领先约 2.5 µs，因此波动主要来自主机路径。

分别重试了设备和主机优化：

| 方向 | 当前 | 试验 | 结论 |
| --- | --- | --- | --- |
| Q2/B32，固定 BV64 外提 `dt_bias` | Graph 14.161 µs，eager 69.056 µs | Graph 14.967 µs，eager 69.856 µs | 回退 |
| Q2/B256，固定 BV64 外提 `dt_bias` | Graph 104.818 µs，eager 162.128 µs | Graph 108.869 µs，eager 166.352 µs | 回退 |
| Q2/B32，批量构造 LibEntry 参数字典 | 主机 56.4 µs，eager 72.5 µs | 主机 57.1 µs，eager 72.6 µs | 无稳定收益，回退 |

这些试验均未修改生产代码。在不引入 fast-launch 的前提下，当前公共主机路径已接近本轮可安全兑现的收益上限；第 3 节性能表继续保持有效。

## 7. H800 原实验复现命令

原 H800 实验容器需要把 CUDA compat library 放在 `LD_LIBRARY_PATH` 前面。以下命令使用 GPU 4，并用内存 SQLite 避免持久化调优缓存影响结果。

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

NCU 2025.4.1 已在开发机的工作区本地解压使用，没有修改系统 CUDA 包。现有结论来自功能对拍、编译后的 register/spill 数据、固定配置对照、完整生产 benchmark 和第 4 节的 NCU full profile。

## 8. 多平台精度和性能基线（2026-11-25）

### 8.1 原生实现与缓存布局

沐曦和摩尔曾在测试中使用仓库内复制的 vLLM 0.34.0 kernel（`df10933`、`6cb47ba`、`fa67218`）。`de9b4c9` 移除了这份副本。现在精度用例和 benchmark 都直接导入机器上安装的 `vllm.models.kimi_k3.nvidia.ops.recoverssm.kda_recoverssm_verify`，按函数签名构造输入。该入口用于独立测量 verify；模型推理时的分派需要另行确认。[基线实现](baseline/kda_recoverssm_verify.py)

| 已核对的版本 | verify 写入位置及测试处理 |
| --- | --- |
| 沐曦 0.29、摩尔 0.32、海光 0.28.1+dtk2604 | native 接收 `correction_cache` 和 `kg_cache`，请求 `b` 写入物理块 `state_indices[b]`；断言前按请求顺序取回有效记录。海光文件与官方 v0.28.1rc0、v0.29.0 及沐曦样本逐字节一致。摩尔样本的差异在 commit 阶段的块边界计算，verify 相同（`afdf148`）。 |
| 官方 v0.33.0、v0.34.0 | native 接收 `records`，请求 `b` 写入第 `b` 行，与 FlagGems 的接口一致。官方 0.33.0 兼容性记录于 `eb9a7fa`。 |

精度测试分别对比 native 与 Torch 参考、FlagGems 与 native，检查输出、有效 token 的 correction 和 key/gate 记录，以及 checkpoint 保持原值。摩尔的 FP64 Torch 参考在 CPU 上计算。海光镜像的 Kimi K3 模型类解析到 AMD 路径，其 speculative 分支调用 `fused_recurrent_kda`；直接测试镜像自带 verify 的结果仅代表这条独立算子路径。[精度用例](tests/test_kda_recoverssm_verify.py)

### 8.2 厂商导入与计时

- **沐曦**（`154320a`）。精度和性能用例使用同一套进程隔离逻辑，在通过 `spawn` 启动的子进程中先导入 native，让 mcoplib 先注册，再加载 FlagGems 测试代码。导入顺序由 KDA 用例控制。[子进程初始化](tests/kda_recoverssm_worker.py)
- **平头哥**（`3f61b93`、`d2b5dca`）。首次导入 Kimi K3 触发 CUTLASS 能力探测，SAIL 继承的 NVML 查询因缺少 `libnvidia-ml.so.1` 失败。最终实现仅在 PPU 导入 native 期间，用 `torch.cuda.get_device_capability(device_id)` 构造 vLLM 的 `DeviceCapability`，退出后恢复平台方法。已做模拟导入验证，仍待在原 PPU 容器的新进程里单跑首例确认。[导入实现](baseline/kda_recoverssm_verify.py)
- **摩尔**。排查时 MUSA stream、Graph capture 和 replay 均通过，报错发生在该镜像的 Triton Graph timer 调用 `torch.cuda.current_stream()`。KDA benchmark 使用仓库通用计时；该分支没有算子专用的 MUSA timer。复核性能时应记录镜像版本及实际计时模式。[性能用例](benchmark/test_kda_recoverssm_verify.py)

### 8.3 同机 SpeedUp 与跨芯片 vs Base

`d99f996` 将采集用例编码为 NCU 子进程可重建的普通 Python 元组，并将 NV native 和目标平台的 `out` 都移到计时前分配。benchmark 通过同一个用例键和 dtype 精确匹配离线 JSON。NV 采集覆盖 10 个生产形状和 3 个边界用例，FP16/BF16 共 26 条；目标平台 benchmark 默认测 10 个 BF16 生产形状。[采集输入与形状键](baseline/kda_recoverssm_verify.py) · [benchmark](benchmark/test_kda_recoverssm_verify.py)

```bash
# 在 NV 参考机采集。默认参考卡为 H800。
python tools/collect_baseline_nvidia.py --ops kda_recoverssm_verify --output tools/kda_nv.json

# 将该 JSON 复制到目标机后，在同一份代码上运行。
python -m pytest -q -s benchmark/test_kda_recoverssm_verify.py --base-data tools/kda_nv.json
```

`SpeedUp` 是目标机 native 延迟除以同机 FlagGems 延迟。`vs Base` 是 NV native 延迟除以目标机 FlagGems 延迟，再除以硬件系数。该系数按 NV 记录的瓶颈资源选择，为目标芯片与参考芯片对应带宽或计算能力的比值。检查日志中的 `factor=`；值为 `None` 时，显示的是未经折算的延迟比。[折算实现](benchmark/base.py) · [采集器](tools/collect_baseline_nvidia.py)

截至 `d99f996`，两套 native 接口各 26 组已用 CPU 和 meta 张量验证 NCU 脚本的输入重建、输出预分配和 JSON 匹配。该适配尚待在 NV 参考机采集实际延迟与 NCU 瓶颈，再在目标机核对硬件系数和 `vs Base`。第 3 节的 H800 实测表保留原测量口径。
