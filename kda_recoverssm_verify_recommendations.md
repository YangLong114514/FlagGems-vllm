# KDA RecoverSSM Verify 优化建议

原始评估基于 `feat/kda-recoverssm-verify` 的 `d934b34`，建议文档首次提交于 `61f3200`。在当前分支的 rebase 历史中，两者分别对应 `f595715` 和 `6fd993a`。原评估引用 NVIDIA H800 相对 vLLM 0.34.0 Triton 实现的测量，输入为 BF16 activation、FP32 checkpoint，生产形状均为 `K=V=128`。Q2、Q4、Q8 分别表示每个请求最多验证 2、4、8 个 token。Ascend 独立后端及 persistent Vector 优化记录见[优化记录第 9 节](kda_recoverssm_verify_optimization.md#9-ascend-独立后端与-persistent-vector-优化2026-11-26)。[原始评估依据](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/kda_recoverssm_verify_optimization.md)

## 建议落实情况（截至 `d99f996` 及本次 Ascend 优化）

| 工作项 | 后续结果 |
| --- | --- |
| 核对真实布局及 Q2 小 batch 启动开销 | 已对照 packed QKV、列视图状态索引、slot 顺序与预分配 `out`。主机启动路径做过配对实验，最终保留通用 `libentry/libtuner`。在 `209e5c7` 的仓库标准计时下，Q2/TP8/B32 的 H800 同机 SpeedUp 为 Kernel 1.121×、Graph 1.048×。下文原始 eager 数据采用另一种计时协议。 |
| 检查 Graph 配置与窗口内核 | 已加入 `batch × heads` 的 workload 分桶。Q2/Q4 全局外提 `dt_bias` 及 Q4/Q8 满窗特化均做过实验；交付代码保留 Q8 外提。各轮配置和回退依据见优化记录第 2、5、6 节。 |
| 后续新增的厂商基线 | 精度和同机性能已按安装版 native 的签名适配两种缓存布局。NV 采集与目标机 benchmark 已对齐 `out` 预分配和用例键。待验证项是 PPU 真机冷启动，以及在 NV、目标机之间跑通实测折算。 |
| Ascend 独立后端与 persistent Vector | 新增独立后端，显式关闭 auto-blockify，并按 40 个 Vector 核实现 program 内多 tile 循环。Q2/B32 使用连续工作区间、固定 BV64 和分桶 `dt_bias` 外提，延迟由首版 132.572 µs 降至 100.656 µs；完整 92 项功能测试及仓库 benchmark 通过。详见优化记录第 9 节。 |

跨平台版本对应、提交号、采集命令及验证范围见[优化记录第 8 节](kda_recoverssm_verify_optimization.md#8-多平台精度和性能基线2026-11-25)，Ascend 宏问题、persistent 设计、profiler 数据和最终性能见[优化记录第 9 节](kda_recoverssm_verify_optimization.md#9-ascend-独立后端与-persistent-vector-优化2026-11-26)。以下保留 `d934b34` 时的判断和实验设计，便于追溯每项建议的依据。

## 原始评估的优先结论

**如果要达到仓库的性能门槛，优先查 Q2、TP8、B32 的 eager 启动开销。如果主要部署在 CUDA Graph 下，先验证 Q8 的配置选择。** 现有 10 组生产形状在 Graph 下均快于 vLLM，SpeedUp 为 1.088–1.331×。eager 只有 Q2、TP8、B32 低于 0.9。该形状的 Gems 耗时从 79.488 µs 降到 73.671 µs 即可达标，差约 5.8 µs。[性能表](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/kda_recoverssm_verify_optimization.md#L79-L96)

| H800 历史测量 | vLLM | Gems | SpeedUp | 对优化方向的含义 |
| --- | ---: | ---: | ---: | --- |
| Q2、TP8、B32，eager event | 66.304 µs | 79.488 µs | 0.834× | 优先分解调用端与设备端耗时 |
| 同一形状，CUDA Graph replay | 15.358 µs | 14.114 µs | 1.088× | 已捕获的执行路径领先 |
| 同一形状，NCU 单 kernel | 24.67 µs | 22.14 µs | 1.114× | 设备端也领先，更支持先查提交路径 |
| Q8、TP8、B32，CUDA Graph replay | 59.171 µs | 48.651 µs | 1.216× | 与同轮固定配置比较后，再决定是否改内核 |

SpeedUp 为 vLLM 延迟除以 Gems 延迟。eager 计时把完整 Python 算子调用放在两个设备 event 之间，Graph 测重放，NCU 使用多轮 replay 并默认刷新缓存。这三种测量应各自比较 native 和 Gems，不能用彼此的差值估算 Python 耗时。[计时实现](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/benchmark/test_kda_recoverssm_verify.py#L29-L48) · [NCU 采集说明](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/kda_recoverssm_verify_optimization.md#L98-L115)

## 1. 补齐 vLLM 实际调用布局

分支基准分别分配连续的 Q、K、V，按顺序构造 checkpoint slot，并省略 `out`。vLLM 0.34 的 Kimi K3 Triton 路径从同一份卷积输出切出 Q、K、V，取二维状态索引表的第一列，并将已经准备好的 `out` 传入算子。因此即使形状相同，token stride、索引 stride 和分配成本也可能不同。[分支输入构造](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/baseline/kda_recoverssm_verify.py#L75-L121) · [vLLM 0.34 调用代码](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/models/kimi_k3/nvidia/kda.py)

保留原基准用于历史回归。新增一组输入，依次切换预分配 `out`、packed QKV view、列视图 `state_indices`，最后组合三项。native 和 Gems 使用相同布局和相同 slot 序列，分别报告 eager 与 Graph 中位数。先覆盖 Q2/B32、Q2/B256、Q8/B32 的 TP8 形状，再在 B256 单独比较顺序和打散的 slot。这样既能得到贴近调用方的结果，也能判断哪项差异影响了延迟。

## 2. 缩短 Q2 小 batch 的 eager 启动路径

历史提交 `f780a25` 曾缓存调优后的 launcher，提交说明记录单次主机提交耗时由 55 µs 降至 21 µs。该路径在 `59d7884` 中被移除，旧提交使用的测试集也与当前不同。这组数字能说明启动路径值得复查，当前可获得的收益仍需重新测量。[历史启动缓存](https://github.com/YangLong114514/FlagGems-vllm/commit/f780a258368e0c7229f6372ae25198106c103a44) · [当前算子入口](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/src/flaggems_vllm/ops/kda_recoverssm_verify.py#L274-L385)

1. 在 Q2、TP8、B32 固定 `out`、形状、stream 和已编译配置。预热后分别记录主机提交时间、profiler 中的 GPU kernel duration，以及原基准的完整 wrapper 延迟。
2. 若额外开销集中在 `LibEntry` 参数归类、构造特化键和计算 grid，先缩短缓存命中路径。首次调用继续通过现有 `libtuner` 编译和选配置。
3. 复用历史实现时修正缓存键。`lower_bound=None` 和 `0.0` 都映射到数值 `0.0`，但 `USE_LOWER_BOUND` 不同。验证两者交替调用，以及切换设备、stream 和 Graph 捕获时的行为。[历史缓存键实现](https://github.com/YangLong114514/FlagGems-vllm/commit/f780a258368e0c7229f6372ae25198106c103a44)

**保留条件**为原基准 Q2/B32 eager SpeedUp 达到 0.9，同轮对照中真实布局的主机提交时间也下降，且生产 Graph 形状没有稳定回退。若现有 wrapper 已达标、真实布局收益又落在测量波动内，可停止该方向。

## 3. 让调优配置匹配 CUDA Graph 工作负载

算子的显式调优 key 为 `K、V、SPEC_QUERY_LEN`，框架会追加 tensor dtype。`batch` 和 `num_heads` 只参与 grid，同一窗口下并行规模不同的请求可能复用首个形状选出的配置。分支已记录 Q2 在 eager 调优时选中 BV16 后 Graph 回退。Q8 的完整 Graph 基准为 48.651 µs，另一轮固定 BV32、2 warps 的参考值约 41 µs。两轮条件不同，不能把约 7.6 µs 计为可兑现收益。[调优入口](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/src/flaggems_vllm/ops/kda_recoverssm_verify.py#L71-L76) · [固定配置对照](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/kda_recoverssm_verify_optimization.md#L117-L135)

- **Q8/B32**。在同一轮、同一布局下比较默认选择与现有 9 组候选，记录实际命中的 `BV / num_warps / num_stages` 和 Graph 中位数。先看 BV32、2 warps 的优势能否复现。
- **Q2/B32、B128、B256**。用两个独立进程和空调优缓存，分别让 B32、B256 首先触发调优，再测全部形状。若顺序改变后配置或延迟明显变化，再用固定配置复核。
- **实现选择**。只有最优配置随并行规模稳定切换时，才将粗粒度 `batch × heads` 桶同时传入启动缓存和调优 key。若各规模的 Graph 胜者一致，按目标执行模式选配置即可。保留全部 10 组的 eager 和 Graph 对照结果，避免一个模式的收益掩盖另一模式的回退。

## 4. 控制流与归约的内核实验

| 候选 | 为什么值得试 | 最小实验与取舍 |
| --- | --- | --- |
| Q4/Q8 满窗口特化 | 10 个生产基准全是满窗。当前循环已静态展开，但每个 token 仍会判断 `token_offset < query_len`。 | 固定当前最佳配置，由每个 request 在设备上判断一次 `query_len == SPEC_QUERY_LEN`。满窗连续执行，ragged 沿用原路径。比较完整与混合窗口的 Graph 延迟，同时检查 Q8 的代码体积和寄存器。Graph replay 的真实长度来自设备元数据。 |
| Q4/Q8 的 K 维归约布局 | Q4 的 short-scoreboard stall 约 18%–20%。Q8 为 168 registers/thread，实际 occupancy 18.1%，每个 scheduler 平均 0.66 个 eligible warp，DRAM 利用率仅 12%。 | 先检查 `state @ k` 和 `state @ q` 的生成代码。若有跨 warp 交换或布局转换，在 `K=128` 下单独试验让一个 warp 完成一行归约。记录 shuffle、spill、eligible warps 和实际延迟。若当前映射已等价，就跳过该试验。 |

每轮只改一个变量。所有变体验证输出、FP32 correction、原始 key/gate 记录以及 checkpoint 只读语义。满窗特化还应检查 ragged 尾部和 NULL 请求的 records 保持原值，避免只比对有效位置而漏掉额外写入。分支已有独立 FP64 oracle 和 native 对拍可复用。[递推和 token 循环](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/src/flaggems_vllm/ops/kda_recoverssm_verify.py#L44-L247) · [功能对拍](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/tests/test_kda_recoverssm_verify.py#L37-L151)

**已有回退记录可直接用于排期。** 调优候选从 9 组扩到 36 组没有稳定收益。Q8 在固定实验中改用 4 warps 或限制 `maxnreg=128`，Graph 延迟由约 41 µs 升至约 48 µs。整窗公式原型由 40.89 µs 升至 104.90 µs，并出现 40 次 spill。除非新设计能具体减少跨 warp 通信、窗口内成对计算或中间量占用，否则这些方向排在上述实验之后。[已回退实验](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/kda_recoverssm_verify_optimization.md#L123-L158)

## 第一轮 H800 排期

| 顺序 | 形状 | 实验结束时应能回答的问题 |
| --- | --- | --- |
| 1 | Q2、TP8、B32 | 换成真实布局后，eager 是否仍低于 0.9？主机提交、输出分配和设备执行各占多少？ |
| 2 | Q8、TP8、B32 | 同轮 Graph 实验中，默认配置与最佳固定配置相差多少？结果能否重复？ |
| 3 | Q2、TP8、B32/B128/B256 | 首次调优顺序是否影响后续延迟？是否有必要按并行规模分桶？ |

每次记录提交号、设备、输入 stride、是否传入 `out`、调优缓存状态、命中配置、native/Gems 中位数及重复运行的波动。计时口径或输入布局改变后，为两侧重建基线。首轮若已解决 eager 门槛且 Graph 配置稳定，再按线上 Q4/Q8 的实际占比决定是否投入内核重写。[仓库性能验收与试验流程](https://github.com/YangLong114514/FlagGems-vllm/blob/d934b3461dbf76f00027e20ffaaf18e1cdd15695/workflow.md)
