# 训练实验日志

本日志只记录已运行、可定位的实验；未通过实战门槛的检查点不作为网页默认 AI。
评测均为「每个初始牌墙让候选轮换四座、其余三座为冻结 Teacher」；标准误以每个
牌墙四座均分为独立样本计算。

## 2026-08-04：监督 v1、DAgger、前瞻 v2

| 检查点 | 训练 | 独立评测种子 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| `rule-policy-classic-mlp`（v1，76 维） | 40 Teacher 局 + 136 游金课程，6 epoch | 20,900,000 起的 100 个牌墙 | 平均净得分 **−8.44 ± 1.59**，胡率 14.25%。 | 明显弱于 Teacher，不上线。 |
| `dagger-classic-run1`（v1） | v1 起点；40 Teacher 局 + 136 课程；3 轮各 20 策略局、每轮 4 epoch | 同上 | 均分 −8.35，胡率 17.25%。 | 点估计改善小于噪声；不宣称更强。 |
| `rule-policy-classic-mlp-v2`（80 维） | 40 Teacher 局 + 136 课程，6 epoch | 同上 | 平均净得分 **−7.54 ± 1.70**，胡率 14.5%。保留集 Teacher 动作一致率 78.5%。 | 比 v1 点估计更接近 Teacher，但仍显著为负；保留为研究候选，不上线。 |

训练种子与独立评测种子不重叠。v1 的早期评测曾与训练种子重叠，已废弃，不能用于
任何比较。下一次候选进入网页前，必须在至少 400 个全新牌墙上重新评测，并要求其
对 Teacher 的净得分置信区间不劣。

## 2026-08-05：DAgger v2、公开余牌 Teacher 与首轮在线 RL

| 检查点 / 对手 | 训练或选择方式 | 独立评测 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| `dagger-classic-v2-run1`（80 维） | 在 v2 上收集策略访问状态并以冻结 Teacher 标注；保留集动作一致率从 74.78% 升至 83.89%。 | 21,000,000 起 100 个牌墙 | 平均净得分 **−3.57 ± 1.65**，胡率 20.5%。 | 是当前最接近 Teacher 的研究候选，但置信区间仍未证明已胜过 Teacher。 |
| `AvailabilityTeacherAgent` | 对听牌按公开可见的剩余张数加权；系数在 20,800,000 起 50 墙上选择。 | 21,000,000 起 100 个牌墙 | 平均净得分 **−0.58 ± 0.79**，胡率 23.5%。 | 选择集上的正向点估计没有在留出集复现；Teacher 保持不变。 |
| 80 维 wide-and-deep 直连通路 | 与 v2 相同 Teacher 训练集。 | Teacher 保留集 | 动作一致率 76.87%，低于 v2 的 78.5%。 | 训练前淘汰，未做昂贵对局评测。 |
| `reinforce-classic-run1` | 从 `dagger-classic-v2-run1` 出发；3 × 40 局，学习率 0.0003，结算净得分 REINFORCE，对手为 3 个冻结 Teacher。 | 21,100,000 起 100 个牌墙；与 DAgger v2 同墙配对 | 自身均分 **−4.21 ± 1.71**；相对 DAgger v2 为 **+0.59 ± 1.13**，95% CI **[−1.64, 2.81]**。 | 方向不确定，不晋升、不上传；该实现只作为在线训练与配对验收基础。 |

REINFORCE 的训练批次平均净得分从 −15.98 升至 −1.90，但这不是独立强度证据。

| 检查点 | 训练 | 独立评测 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| `actor-critic-classic-run1` | 从 `dagger-classic-v2-run1` 出发；600 局（6 × 100，四座均衡），学习率 0.00015；线性状态价值基线每批拟合 4 次。 | 21,200,000 起 100 个牌墙；与 DAgger v2 同墙配对 | 自身均分 **−4.23 ± 1.59**；相对 DAgger v2 **−0.40 ± 1.12**，95% CI **[−2.60, 1.80]**。 | 价值 MSE 每批下降，但未转化为强度；淘汰，不上传。 |

纯 Python 的线性 critic 已证明流程正确，却不足以带来可验证收益。下一项实验不是继续堆叠
该训练，而是准备带共享表征的 policy-value 网络、固定的按牌墙切分数据清单及冻结对手池；
完成后仍以全新牌墙通过配对区间门槛。

## 2026-08-05：安全轨迹 v3 与 deployment-matched policy-value DAgger

轨迹 v3 将训练观测与复现元数据分离：训练 JSONL 默认不含 deal seed、行为 seed、对手
暗牌或牌墙顺序；可选 replay 索引独立保存且不交给训练器。切分按完整物理牌墙进行，
候选四座轮换共享一个不透明切分组，避免同一墙进入不同分区。

| 数据 / 检查点 | 构造或训练 | 结果 | 结论 |
| --- | --- | --- | --- |
| `trajectory-balanced-classic-v3-run1` | 160 Teacher 局、120 随机合法探索局、80 个 Teacher 选择 `pass` 的物理响应课程、136 个规则验证游金课程。 | 496 条完整轨迹、16,402 决策；`discard` 12,470，`pass` 85，`advance_tour` 136。 | 比纯 Teacher 数据覆盖更好，但稀有杠类仍偏少，不能单独作为强度依据。 |
| `policy-value-classic-v1-run3` | 145 维公开候选特征，128 隐层，共享 policy/value 头；按验证集选择 epoch 33。 | 留出集动作准确率 92.79%；在 21,600,000 起 200 个新牌墙上相对旧 DAgger v2 **+1.87 ± 1.36**，95% CI **[−0.79, +4.53]**。 | 方向为正但区间跨 0，不晋升。 |
| `candidate-teacher-dagger-classic-v1-run1` | 用 run3 在 100 副牌墙的四座轮换中对抗 3 个冻结 Teacher；只保留候选实际决策的 Teacher 标签。 | 400 局、4,366 个候选访问决策；训练/验证/测试各按整副牌墙分组。 | 修正了旧 DAgger 全席位模型自博弈与实际部署分布不一致的问题。 |
| `policy-value-classic-v1-run4-dagger` | 从 run3 微调 20 epoch；加入上述在线训练/验证轨迹；验证集选 epoch 20。 | 混合留出集准确率 94.80%，在线验证 96.33%；在 21,800,000 起 200 个新牌墙上相对旧 DAgger v2 **+3.80 ± 1.23**，95% CI **[+1.40, +6.20]**。 | 已显著优于旧 DAgger v2；但与 run3 在同一组牌墙上差 **+0.08 ± 0.90**，95% CI **[−1.68, +1.83]**，故不把 run4 宣称为比 run3 更强，也不替换网页默认 Teacher。 |

policy-value 网络当前每步以单一候选集合推理，GPU 小批量调度会使长对局评测较慢；下一轮
工程优化应为批量 rollout / 事件序列编码。模型升级仍须先在全新牌墙上胜过当前冻结候选，
并保留相对 Teacher 的独立配对报告。

## 2026-08-05：公开事件序列与 neural PPO 的反证结果

| 检查点 / 实验 | 设置 | 验收结果 | 结论 |
| --- | --- | --- | --- |
| `policy-value-public-sequence-v1-run1` | 两层 64 隐层 Transformer，最近 24 个公开事件；暗杠牌面脱敏。 | 混合留出集准确率 93.82%、测试集 90.10%，均低于 candidate MLP。 | 随机初始化的序列模型在当前数据量下不值得实战评测，淘汰。 |
| `policy-value-public-sequence-residual-v1-run1` | 从 run4 精确复制候选 MLP 路径，事件 Transformer 仅学习零初始化残差；继承参数学习率为残差分支的 10%。 | 离线测试准确率 95.13%，但在 21,900,000 起 200 个新牌墙上相对 run4 **−1.15 ± 0.57**，95% CI **[−2.28, −0.03]**。 | 离线一致率提高却显著伤害实战；淘汰。公开事件顺序仍应保留在数据中，但下次须以更大规模或更强目标训练。 |
| `torch-ppo-classic-v1-run1` | 从 run4 起点；3 × 256 局候选一席 vs 三 Teacher，PPO 裁剪策略目标 + 同网络终局净分 value。 | 8,276 个采样决策；在 22,000,000 起 200 个新牌墙上相对 run4 **+0.09 ± 0.58**，95% CI **[−1.04, +1.22]**。 | PPO 管线与合法动作约束正确，但首轮没有可验证收益；不晋升。 |

因此下一轮不应继续在 16k 级别标签集上堆叠更大网络，优先级为：扩大候选对冻结对手的
在线数据、保存多 checkpoint 对手池、批量化 rollout，并以独立牌墙选择 PPO 迭代而非
依据训练回报或 Teacher 动作准确率选择。

## 2026-08-05：批量 rollout、冻结对手池与 PPO 规模验证

`TorchPolicyValueAgent` 新增变长合法动作集合的批量前向；PPO 同时推进独立牌局，并按
冻结 checkpoint 分组批量执行对手决策。规则执行、动作合法性检查和结算仍逐局进行。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `torch-ppo-classic-v1-run2-pool` | run4 起点；3 × 1,024 局，batch 32；75% Teacher、25% 冻结 run3 对手；33,719 个候选采样决策。 | 22,100,000 起 200 个新牌墙，相对 run4 **−0.08 ± 0.29**，95% CI **[−0.64, +0.49]**。 | 规模与对手池都没有产生可验证增益；不晋升。批量化与对手池保留为下一次更大规模联赛的基础设施。 |

## 2026-08-06：单次反事实动作价值的反证与数据质量门槛

新增采集器在候选实际访问的公开状态中，将每个合法动作强制应用到私有内存分支并结算；
JSONL 只保存本家/公开观察、合法动作和对齐的 `action_values`，不保存牌墙、暗牌或种子。
首轮使用 run4 对三 Teacher、每动作一次确定性 continuation：64 副墙四座轮换共 252 个
决策、1,797 个动作分支，动作价值训练/验证/测试按整副墙分组（192/32/28）。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-v1-run1` | run4 起点；Teacher / DAgger 监督加权 1，单次反事实软偏好权重 8，温度 16；以 32 条反事实验证集选择 epoch。 | 第一组 22,200,000 起 200 墙：相对 run4 **+1.41 ± 1.21**，95% CI **[−0.95, +3.78]**；预先隔离的第二组 22,300,000 起 400 墙：**−1.98 ± 0.87**，95% CI **[−3.68, −0.28]**。 | 第二组显著为负，淘汰、不晋升。单次、同一隐藏世界 continuation 的标签噪声/条件偏差不能靠提高权重解决。 |

由此增加两个强制质量控制：动作价值跨度为零或很小的样本按跨度降权；采集器在
`rollouts_per_action > 1` 时导出每个动作的标准误，训练可按标准误降权。下一轮必须混入
冻结对手池、重复 rollout，并仍以全新牌墙配对评测决定去留。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-v1-run2-mixture` | run4 起点；40 副墙四座轮换，75% Teacher / 25% 冻结 run3；每合法动作 4 个共享对手配置的 rollout，共 160 决策、4,456 分支，动作均值标准误 2.70；跨度和标准误降权，反事实权重 6。 | 22,400,000 起 200 墙：**+1.77 ± 1.12**，95% CI **[−0.43, +3.96]**；预先隔离的 22,500,000 起 400 墙：**−1.20 ± 0.72**，95% CI **[−2.61, +0.22]**。 | 离线反事实测试损失从 run4 的 25.30 降至 6.34，但两组实战均未证明显著提升，第二组点估计为负；不晋升。重复 rollout/对手池降低了噪声，却没有解决“特定隐藏世界”分支目标与部署信息集不一致的问题。 |

随后实现了 `--belief-resample`：对每个 rollout 固定本家手牌、公开河/副露、花数量、金牌和
公开动作，重新分配未知墙、对手暗手、非本家花、对手暗杠牌面；若存在无法由当前先验正确
条件化的他家公开天听状态，安全跳过。真实 Torch smoke 在 8 副墙四座轮换中产生 32 决策、
436 分支、64 个 belief worlds，未导出私有状态。它只验证数据构造和安全边界，尚未训练或
评测，不能作为强度证据。下一步是足量 belief 数据、独立分组切分和相同的实战验收。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-belief-v1-run1` | run4 起点；40 副墙四座轮换，公开信息 belief 重采样、75% Teacher / 25% 冻结 run3、每动作 3 个 world，共 160 决策、3,039 分支、480 worlds；温度 24，按价值跨度与标准误（均值 18.93）降权。 | 22,600,000 起 200 墙：**+1.71 ± 0.96**，95% CI **[−0.17, +3.59]**；预先隔离的 22,700,000 起 400 墙：**+0.35 ± 0.65**，95% CI **[−0.92, +1.62]**。 | 不显著，不晋升。相比具体隐藏世界目标没有复现显著负收益，说明信息集重采样是必要的数据修正；但公开先验没有吸收对手历史行为，且数据量仅 160 个状态，尚不足以形成可验证优势。 |

## 2026-08-06：逐动作 Q 头的可辨识性诊断

candidate MLP 新增零初始化的逐合法动作 Q 头，旧 v1/v2 checkpoint 加载时保持 policy 头选牌
不变。训练同时保留 soft preference；Q 目标为 `terminal_score / 80`，报告每动作 Huber/MAE 与
Q-argmax。checkpoint 选择也支持用 Q 的验证 Huber 而非软策略损失，防止训练目标和选择指标错位。

| 检查点 / 实验 | 设置 | 隔离结果 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-q-only-diagnostic-v2` | 从 run4 加载；只使用上表 belief-v1 的 88/52/20 个训练/验证/测试状态，soft preference + 逐动作 Q 回归，按验证 Q Huber 选第 15 epoch。 | 验证 Q Huber 0.1064、Q-argmax 46.15%；测试 Q Huber 0.1249、平均绝对误差 27.29 分、Q-argmax 25.00%。 | 这是头部与指标管线诊断，不是候选策略实验。20 个隔离测试状态太少且 Q 排序接近随机，严禁使用该 Q 头试玩或晋升；下一轮必须扩大独立 belief 状态并混入原 policy 监督。 |

## 2026-08-06：反事实分支批量调度基础设施

collector 新增 `rollout_batch_size`。批量模式只把多个独立的“合法动作 × rollout world”分支的
`scores_batch` 推理合并；每个分支仍保有自己的规则实例、随机数、强制动作和逐步合法性检查。
串行模式（batch=1）保持原实现，作为回归基线。

| 验证 | 设置 | 结果 | 结论 |
| --- | --- | --- | --- |
| 串行—批量等价性单测 | 相同 core 牌墙、相同确定性策略、每动作 2 个 continuation；batch=1 对比 batch=8。 | 所有安全导出的观察、合法动作、chosen index、动作价值、标准误和公开事件完全相同。 | 调度不改变规则推进或目标语义。 |
| 真实 CLI 冒烟 | run4-dagger checkpoint、core、10 墙四座、batch=8、每动作 1 个 continuation；临时数据使用 30/30/40 切分仅为验证三分区输出。 | 40 个决策、321 个分支；manifest 明确 `private_simulator_state_exported=false`。 | 参数、Torch 批量推理和安全导出已跑通；这是基础设施验证，不是模型训练或强度证据。 |
| 实际合并指标 | run4-dagger checkpoint、core、1 墙四座、每动作 4 个 continuation、batch=32；不落盘。 | 160 个分支，116 次 batch 推理、合并 728 个决策，最大单次 32；导出观察未出现 `wall` 或 `opponent_hands`。 | collector 已能在同一决策的多个 rollout world 间实际合并推理；下一步可安全提高 world 数。 |

下一步不直接训练新 checkpoint：先用该调度扩大每状态的 belief worlds，并在缩小规则的 toy profile
上实现、校准顺序粒子过滤，再构造保守分布价值目标。

## 2026-08-06：顺序粒子 belief 的数学与安全原型

新增通用、非序列化的 `SequentialParticleBelief`：它只保存运行时不透明粒子和标准化权重，
每个公开事件由调用方提供粒子 transition 与行为似然；导出层只能读取 ESS、熵、log evidence、
重采样/拒绝计数等聚合诊断。

| 验证 | 设置 | 结果 | 结论 |
| --- | --- | --- | --- |
| 可枚举隐藏类型玩具博弈 | 两个等先验对手风格；观察 `claim` 后 `pass`；与手算贝叶斯后验比较。 | aggressive 后验为精确值 `4/13`；无重采样时权重与枚举一致。 | 数学更新正确。 |
| 粒子退化与失败安全 | 强偏好似然触发 ESS 阈值重采样；另设所有粒子零似然。 | 重采样后保留重采样前 ESS；零总似然抛错且不改写原 belief。 | 不会把崩溃的后验静默伪装成均匀分布。 |

该原型尚未接入厦门 collector，未产生训练数据、模型或强度结论。下一步先构建可回放的本家私有
摸牌轨迹和完整公开事件 replay，再在缩小规则环境中与精确后验校准。

随后 collector 已在**内存**中把本家初始手牌/花、每次私有摸牌（含补花与补杠）和本家动作
绑定到每个候选决策快照；动作记录覆盖公开日志会隐藏牌面的暗杠。定向测试确认庄家的开局摸牌
可定位到首个公开 `draw`，而安全导出的 JSONL/state/source metadata 均没有 trace、墙或他家暗手。
这仍是 replay 前置数据层，不是新的训练数据或模型结果。

### core 公开历史 replay oracle（仍未接入 collector）

在内存中保留开局快照后，新增 core replay：候选自己的出牌/响应取自 private trace；对手按冻结
策略的平滑行为似然处理（若策略提供 action scores 则使用温度 softmax，否则才回退确定性）；每次规则
推进都逐项核对 `draw/discard/chi/pong/ming_kan/an_kan/add_kan/hu/result` 公开事件。固定 `seed=953`
的 13 个候选快照全部重放成功，且庄家后续私有摸牌在回放时被严格核对。新增 audit 仅报告提议/接受
粒子、接受率、平均 log-likelihood 和拒绝类别，不会返回任何暗牌或牌墙。

同时测试“随机未知开局 + 拒绝不一致历史”的朴素 proposal：同一局 5 个公开事件前缀只接受
6/200 个，13 个事件前缀为 0/200。该高拒绝率说明它不是可用的历史 posterior proposal，故没有
修改 `--belief-resample`、没有生成数据、没有训练模型。后续必须采用事件约束的序列 proposal，并先
报告接受率和 ESS，才可进入 collector 实验。

### 最新弃牌局部 SIR：受门槛保护的中间数据构造

先实现最小的可重建事件：response 状态的末尾公开历史必须为同一对手的
`normal draw → discard`。从本家/公开先验 world 逆转该弃牌并按冻结策略行为概率 SIR；神经策略使用
可用 action scores 的温度 softmax，Teacher 使用平滑确定性 fallback。该局部条件化默认关闭，绝不称为
完整历史 posterior。

| 校准（seed 930–939，32 个合格 response 状态，每状态 32 粒子） | 结构一致率 | 平均 ESS 比例 | 最低 ESS 比例 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原始行为似然，power=1 | 1.000 | 0.156 | 0.035 | 权重退化，不可直接采集。 |
| 保守 tempered likelihood，power=0.25 | 1.000 | 0.669 | 0.560 | 可作为 gated 小范围实验的起点。 |

collector 现提供显式 opt-in：32 粒子、power=0.25、最低 ESS 比例 0.5。状态不在受限事件范围、候选
合法动作不一致或 ESS 低于门槛时会跳过；source metadata/summary 只写入配置和聚合 ESS，没有墙、暗手、
粒子或随机种子。端到端 core 导出回归测试已通过；尚未用该数据训练或做强度结论。

经典档增加了显式安全边界：游金、天听和早局 response 不尝试逆转；gold lock 与上一张未被吃碰的弃牌
只从公开 action 序列恢复。seed 930–939 的 34 个合格 classic response 状态中，power=0.25 的平均/最低
ESS 比例为 0.425/0.094；power=0.10 为 0.583/0.094。因此 classic 的初始 gated 实验使用 32 粒子、
power=0.10、最低 ESS 0.20；仍未开始训练或实战评测。

## 2026-08-07：局部行为条件化的独立反证

| 检查点 | 构造或训练 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-local-belief-v1-run1` | run4 起点；40 墙四座轮换，局部 latest draw→discard SIR（32 粒子、power 0.10、ESS ≥ 0.20）、75% Teacher / 25% 冻结 run3、每合法动作 3 个 belief world。采得 124 个 response 决策、822 分支、372 条条件化 world，平均 ESS 比例 0.960；温度 24，价值跨度和标准误降权。 | 22,800,000 起的 200 墙筛选：**+1.04 ± 0.76**，95% CI **[−0.46, +2.53]**；独立 22,810,000 起的 400 墙：**−0.14 ± 0.59**，95% CI **[−1.29, +1.01]**。 | 没有通过 400 墙正向下界门槛；不晋升。局部后验比公开先验更精确的假设尚未获得实战支持。 |

这批数据的平均每动作标准误为 18.55 分，平均价值跨度为 28.47 分，124 个状态也不足以稳定选择
checkpoint。训练器新增逐动作下置信界软标签 `Q - z × stderr`（`--action-value-confidence-z`）：它和既有
整体标准误降权互补，只在明确开启时对高方差动作保守回缩。下一轮先用固定数据进行 z 的小范围筛选，
再决定是否值得扩大数据；任何离线损失改善都不能替代新的同牌墙实战验收。

| 检查点 | 固定数据消融 | 新牌墙筛选 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-local-belief-lcb-v1-run1` | 上表相同 124 状态、run4 起点，只有 soft preference 改为 `Q - 1.0 × stderr`；验证集按同一反事实来源选择 epoch 20。 | 22,820,000 起 200 墙：**+1.04 ± 0.75**，95% CI **[−0.44, +2.51]**。 | 与均值目标筛选几乎相同、仍不显著；不做 400 墙终检。小数据下标签风险厌恶不是主瓶颈。 |

## 2026-08-07：训练期特权 critic 基础设施（尚无强度结论）

参考 Suphx 的 oracle-guiding / global-reward-prediction 方向，PPO 新增默认关闭的
`--privileged-critic`。它在采样时读取完整模拟状态，**仅作为终局净分 advantage baseline**；actor 仍为
本家/公开特征加规则合法动作掩码。critic 特征只保存在 `PpoStep` 的进程内存：不进入 `TeacherDecision`、
轨迹、JSONL、网页、报告或保存的 actor checkpoint，进程退出即丢弃。单测覆盖 batch rollout、PPO 更新、
合法动作以及 actor observation 不含 `wall`/`opponent_hands`。尚未进行 PPO 训练或实战评测，不能据此
声称增益；下一个对照须固定 actor 起点、对手池和 rollout 预算，比较 advantage 方差后再上新牌墙。

16 局 core CLI 冒烟（run4 actor、8 路 rollout batch、32 隐层 critic、1 PPO epoch）产生 143 个候选
决策，actor checkpoint 中没有 critic 参数，报告没有 `wall`、`opponent_hands` 或 `privileged_features`。
该过程仅验证可执行性和隔离边界，不产生可比较强度数字。

## 2026-08-07：配对动作价值误差（采集器升级）

反事实分支已在同一 belief world、同一冻结对手配置中对所有合法动作执行，故新增对每个动作相对于
均值最优动作的配对 gap 标准误，而不仅是各动作边际 Q 标准误。真实 classic 校准（1 墙四座、4 个
response、每动作 6 world）边际标准误为 **18.58** 分，配对 gap 标准误为 **12.13** 分；这支持在策略
软标签中按配对置信界收缩不可靠的排序。该字段没有保存任何私有 world。当前正在采集按物理牌墙分组的
正式数据；尚未训练或做强度判断。

正式数据由两个互不重叠的墙组组成：188 个 response 决策，训练/验证/测试为 106/53/29；每动作 6 个
共享 belief world，平均结构一致率 65.0%、ESS 比例 74.5%、边际标准误 14.33 分、配对 gap 标准误
9.96 分。`policy-value-action-value-classic-local-belief-v3-paired-gap-run1` 从 run4 微调，使用 pairwise
confidence z=1、反事实权重 1，在全新 22,980,000 起的 200 墙筛选为 **+0.84 ± 0.70**，95% CI
**[−0.53, +2.21]**。未通过筛选，不做 400 墙终检、不晋升。结论是配对估计显著降低标签方差，但这批
状态量和离线 policy 更新仍不足以形成可验证胜率；下一轮转向更大规模、方差更低的 on-policy 训练。

## 2026-08-07：训练期特权 critic 的受控 PPO 反证

`torch-ppo-classic-privileged-critic-v1-run1` 固定 run4 起点、经典档、3 × 1,024 局、32 路 rollout batch、
75% Teacher / 25% 冻结 run3 对手；只开启 128 隐层、损失权重 0.25 的训练期特权 critic。共产生 33,688 个
候选决策。三个迭代的 advantage 标准差依次为 0.591、0.578、0.555，critic Huber loss 为 0.164、0.161、
0.146；这是同一 run 内的优化诊断，旧 PPO 报告没有该方差字段，**不能**据此声称相对无 critic 的因果降方差。
三份 actor checkpoint 均不含 critic 参数或 `wall`、`opponent_hands`、`privileged_features` 字段。

在同一组全新 23,000,000 起的 100 个物理牌墙筛选中，相对 run4 的配对净分为：iteration 1 **+0.545 ±
0.583**，95% CI **[−0.597, +1.687]**；iteration 2 **−1.285 ± 0.834**，95% CI **[−2.921, +0.351]**；
iteration 3 **−0.223 ± 0.740**，95% CI **[−1.673, +1.228]**。没有一个 iteration 通过正向下界门槛，故不做
400 墙终检、不晋升。结论是 privacy boundary 和训练管线成立，但当前 critic 配置及 3k 局 PPO 没有给出可验证
强度增益；下一步应先补齐固定策略、同一 rollout 的 critic-on/off 方差 A/B，再决定是否继续投入该路线。

该 A/B 现已完成：用 run4 actor、相同的 run3/Teacher 对手池，先以 512 局独立墙校准 128 隐层 critic，再在
另 512 局（5,704 个候选决策）上以完全相同的 actor 轨迹比较基线。轨迹逐条一致；公开 value 的残差标准差/
均方为 **0.513 / 0.263**，critic 为 **0.591 / 0.350**，残差均方变化为 **−33.0%**（负数表示劣化）。这直接
否定“当前 critic 已降低 PPO 方差”的假设：停止该配置的进一步 self-play，不生成或晋升任何模型。新增
`scripts/measure_ppo_baseline_variance.py` 固化该检验，且仅写入聚合残差；critic、墙与暗手永不落盘。

## 2026-08-07：公开 Q 头校准门槛

为验证是否值得把动作价值接入 PPO，先冻结 policy 偏好（`--action-value-weight 0`）并只训练可部署的
`action_value_head`。过程中发现并修复训练器耦合：此前 Q 回归错误地复用了 policy source weight，因而
`--action-value-weight 0` 也把 Q 标签全置零；现新增独立的 `--action-value-regression-sample-weight`，并有
单测保证 Q-only 校准确实更新 Q 头而不依赖 policy 偏好权重。

修复后的 `policy-value-public-q-calibration-v1-run2` 在同一 188 个、按物理牌墙隔离的 local-belief paired
数据上选择 epoch 31。相对零初始化 Q 头，隔离测试的 Q Huber/MAE 从 **0.122 / 30.85 分**降至
**0.053 / 22.76 分**；但最关键的动作 argmax 排序从 **55.2%** 降至 **48.3%**（仅 29 个状态）。这说明小数据
足以拟合平均数值，却不足以可靠决定吃/碰/过；不接入 PPO、VRPO 或 Q 选牌，不做实战评测，也不晋升。后续
动作价值路线必须先大幅提高按墙隔离的有效状态数与排序精度，而不能只降低 MAE。

为扩大排序检验，新增独立 `local-belief-v4-scale-a`：120 副物理牌墙、356 个 response 决策、每动作 6 个
共享 belief world。结构一致率 57.6%（预设下限 55%）、ESS 96.1%、配对误差 10.56 分；质量没有改善，故只作为
独立加量数据，且不与任何已有测试墙重叠。加入后 Q-only 校准的验证/测试状态增至 111/103。隔离测试上，公开 Q
的 MAE 从零初始化的 **29.79** 分降至 **19.85** 分，argmax 从 **41.7%** 到 **50.5%**；但逐状态配对准确率差
仅 **+8.7** 个百分点，95% CI **[−7.5, +25.0]**，仍跨零。不接入 Q 选牌、PPO 或 VRPO；继续增加独立墙组，直到
排序下界通过或该路线被反证。

第二独立墙组 `local-belief-v4-scale-b` 追加 369 个 response 状态（一致率 62.9%、ESS 96.0%、配对误差
10.80 分）。联合五组墙后，公开 Q 在 146 个隔离测试状态上相对零初始化从 **39.0%** 到 **52.7%**，MAE 从
**29.96** 到 **18.92** 分；配对排序差 **+13.7pp**，95% CI **[+0.6, +26.8]**，首次满足离线 Q 门槛。因此只为
已受监督的 response 阶段实现 `response_action_value`：摸牌后仍严格使用 policy，避免将未训练 Q 用于弃牌。

但全新 23,300,000 起的 200 墙四座轮换筛选相对 run4 为 **−1.88 ± 1.28**，95% CI **[−4.39, +0.63]**。不做
400 墙终检、不晋升。结论不是 Q 头无法拟合 local-SIR 标签，而是这类局部 posterior 反事实排序仍未能外推为
完整对局强度；停止继续扩大这条 local-SIR/Q 数据路线，不能将离线排序当作网页 AI 改进证据。

## 2026-08-07：公开后状态 outcome 数据契约（诊断阶段）

对福州麻将工程的只读调研带来一个可迁移但须重新验证的方向：其文档把成功候选描述为
`policy prior + lookahead / rule + expected score + win probability - risk` 的**保守混合**；同时明确把单隐藏
世界的动作反事实 Q 与原始 PPO 视为不可靠路线。福州与厦门的规则、计番和可见信息并不相同，因而没有复制
权重、规则代码或强度结论；这里只采用其数据纪律：先把实际执行动作的长程结果记录清楚，再讨论混合选牌。

为此 `TeacherDecision` 新增可选 `executed_index`。`chosen_index` 始终是 Teacher 的监督建议，
`executed_index` 则是生成该轨迹的行为策略真正执行的合法动作；旧语料保持 `null` 并可正常读取。Teacher、随机
探索和 DAgger collector 均已写入这一字段；JSONL 仍不含牌墙、他家暗手或随机种子。这样终局得分才只监督
它所属的那个动作，绝不把同一条隐藏 continuation 伪装成所有合法动作的 Q 标签。

首轮数据为 run4 对三名冻结 Teacher 的 120 个物理牌墙、四座轮换：480 局、5,282 个候选决策，按墙组隔离为
3,432/919/931 个 train/validation/test 决策。约 95% 行为动作与 Teacher 建议相同（训练/验证/测试的不同动作数
为 169/43/51），因此这不是足以支持全候选 argmax 的探索数据，而是先校验长程标签是否可学习的 on-policy
数据。`policy-value-afterstate-outcome-classic-v1-run1` 从冻结 run4 编码器训练独立的 score、己方获胜和对手获胜
头；它没有改动 policy logits，报告状态固定为 `diagnostic_only_not_authorized_for_action_selection`。

| 独立集 | 终局净分 MAE | 零预测 MAE | 己方 / 对手胜率 Brier | 实际己方 / 预测己方胜率 | 实际对手 / 预测对手胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation（919） | 27.25 分 | 32.09 分 | 0.154 / 0.180 | 24.7% / 25.8% | 70.4% / 74.2% |
| test（931） | 34.78 分 | 39.66 分 | 0.172 / 0.172 | 28.4% / 27.8% | 71.6% / 72.2% |

这只证明公开特征可预测**已执行动作**在固定后续策略下的软终局结果；它没有观察未执行动作的结果，不能授权
afterstate head、Q head 或网页 agent 对全体候选直接取 argmax。下一阶段必须先以已知 propensity 的温和随机
行为混合增加动作覆盖，并在按墙隔离的动作集上做 propensity/支持度审计；只有 score 与两类概率继续通过校准、
且保守 policy-prior 混合在新牌墙的 200 墙筛选为正，才允许进行 400 墙终检。当时冻结的 run4 仅作为
历史实验起点；网页默认始终是 Teacher。

### 单动作随机干预：使终局标签匹配目标策略后缀

持续 epsilon 行为会让某个动作之后的整局策略也偏离 run4，因此不能为 run4 的动作价值提供干净标签。collector
现改为每一局候选座位只随机选一个决策序号：在该点按 `0.4 × uniform + 0.6 × run4` 选合法动作，此前与此后均
严格执行 run4；每条记录写入实际 `executed_probability`，而随机种子仍从训练 JSONL 剔除。由此，干预点的
终局结果可解释为“已执行动作 + run4 后续”，而不是局部 SIR 的隐藏世界分支。

`afterstate-intervention-classic-v1-run1` 使用 240 个物理牌墙四座轮换（960 局、10,745 个候选决策）、
按墙组隔离为 7,338/1,779/1,628 个一般决策；随机干预点为 **419/95/90**，最小已知 propensity 为约 **0.025**。
训练只取 `executed_probability < 1` 的干预点，并冻结 run4 的 encoder 和 policy head。

| 检查点 | 训练／验证／测试干预点 | 独立测试终局净分 MAE | 零预测 MAE | 己方/对手 Brier | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `policy-value-afterstate-intervention-classic-v1-run1` | 419 / 95 / 90 | **32.38** 分 | 40.47 分 | 0.195 / 0.195 | 标签构造和公开预测均通过最小校准检查，但样本小、尚无未执行候选排序或实战证据；保持 diagnostic-only。 |

该结果与之前的 local-SIR Q 不同：它没有把一个隐藏 continuation 共享给所有动作，并且干预后立即回到目标
策略。不过这仍是稀疏 logged-bandit 数据，不能凭 head 对未执行动作的外推直接 argmax。下一步是扩大按墙隔离的
单点干预覆盖、按 action kind／概率分桶审计支持度，并仅在 policy-prior 锚定的小幅混合通过新的 200 墙筛选后，
才有资格进行 400 墙终检。

### response 定向干预：补齐吃／碰／过覆盖

首轮全阶段干预的 90 个测试点有 64 个弃牌，response 覆盖不足。因此 collector 增加
`--intervention-phase response`：只在本家 response 决策中计数并选择唯一干预点，弃牌和后续 action 都保持
冻结 run4。专项数据用另一组 240 个物理牌墙（960 局），获得 train/validation/test **437/56/116** 个干预；
全部在 response。测试动作组成是吃 54、碰 34、过 23、胡 2、明杠 3，最小 propensity 为 0.133。

`policy-value-afterstate-response-classic-v1-run1` 仍冻结 run4 policy、只训练 outcome head，且仅在
response 干预数据上选择 epoch。测试的终局净分 MAE 为 **30.84** 分，优于零预测 **34.28** 分；己方／对手胜率
Brier 为 **0.182 / 0.183**（测试常数基线为约 0.188）。这说明公开 afterstate 特征在吃／碰／过的目标策略后缀
中有可学习信号。

但 56 个 validation、116 个 test 干预仍不足以校准融合系数、量化候选间排序错误或支持任何 response argmax。
它不替代此前 response-only Q 的失败结论，也不进入 200 墙实战；下一步先训练多 seed outcome ensemble、报告每个
候选的均值/离散度和 policy-prior 偏移率，再预注册一个小幅保守混合及其独立 200 墙筛选。

### response outcome ensemble：校准通过、实战否决

为避免 outcome 训练改变已验证的 run4 policy，网络升级为 v5：新增独立的 `afterstate_encoder`，它从 run4
candidate encoder 初始化，但默认训练时只有它和 score／己方胜率／对手胜率三头可更新；policy encoder、policy
logits、Q 和 value 均冻结。v1 的 240 墙与另一组 v2 的 240 墙 response 单点随机干预按原有物理墙分区追加，得到
**844 / 148 / 220** 个 train / validation / test 干预点。五个独立随机种子训练为 ensemble；每个 checkpoint 都
保留 `diagnostic_only_not_authorized_for_action_selection` 状态。

联合独立测试集上，ensemble 对已执行动作的 score MAE 为 **28.92** 分（零预测 **32.94**），己方／对手 Brier 为
**0.165 / 0.172**；平均 score ensemble 标准差为 **2.77** 分。它说明第二批数据提高了 *logged-action* outcome
校准，却不表示未执行候选的排序正确。审计还显示：在 validation 的 run4 policy top-2 候选内，score LCB 会改动
**36.5%** 的 response 决策，足以构成可检验、但仍受 policy-prior 限制的干预。

预注册候选只在 response 阶段使用该五模型的 `mean(score) - 1.0 × std(score)` 重排 run4 policy top-2；所有
摸牌／弃牌仍逐位复用冻结 run4 policy。它在从训练、验证、测试均隔离的 **23,400,000** 起 200 个物理牌墙、四座
轮换中相对 run4 的配对净分为 **−3.821 ± 1.319**，95% CI **[−6.406, −1.236]**。候选胡率为 21.75%，run4 为
25.25%。负向下界明确，故**不进行 400 墙复核、不晋升、不接入网页**。

结论：执行动作的终局校准不是 response 反事实排序的充分条件；有限策略先验集合和 ensemble LCB 也不能弥补该
数据识别缺口。停止继续扩展这一 selector 或调融合系数。下一轮改为构造规模化、同一环境样本中逐合法 response
动作分支到结算的直接 Monte-Carlo Q 数据；私有状态只留在 collector 内存，导出／输入仍严格限于本家手牌和公开
信息。它将先以小型按墙隔离 pilot 验证覆盖、方差和离线排序，再决定是否值得训练或实战筛选。

## 2026-08-07：direct-response Q pilot 与冻结路径门槛

counterfactual collector 新增显式 `--decision-phase response`；与旧的混合抽样不同，它只导出 response 决策，
没有 response 的牌局直接跳过。每个选中的公开／本家私有信息集在 collector 内存中克隆为所有合法动作的分支，并
在冻结 run4 后缀与 Teacher 对手下结算。分支所用的暗手和牌墙既不进入 action feature，也不进入 JSONL 或 manifest；
跨大量牌墙时，它们是信息集条件终局分布的 Monte-Carlo 样本，而非部署时可见输入。

60 个全新 classic 物理牌墙的 pilot（每座最多两个 response，单分支）得到 **416** 个决策／**920** 条分支，
动作为 pass 255、chi 84、pong 73、hu 3、ming_kan 1，平均动作回报跨度 **34.40** 分。按墙隔离为
267 / 64 / 85 train / validation / test，所有 416 个导出决策均为 response，证明定向采集及边界检查可用；但单个
hidden world 的标签方差仍需要靠扩大墙组而非把同一 world 重复当作独立 rollout。

为保证 Q-only 诊断不影响当前 best policy，网络升级为 v6：Q head 使用独立的 `action_value_encoder`，从 run4
candidate encoder 初始化；`--freeze-policy-path-for-q-only` 进一步冻结所有非 Q 参数，避免即使损失系数为零时
AdamW 的 weight decay 仍轻微移动共享 policy 路径。回归测试和实际 pilot 均确认 run4 与训练后 checkpoint 的
policy logits 最大绝对差为 **0.0**；v1–v5 checkpoint 仍可加载并以 candidate encoder 初始化缺失的独立编码器。

pilot 的冻结 Q 在 85 个独立测试决策上 MAE 为 **32.32** 分（零 Q 36.30 分），但 MAE 不是选择证据。修正
“多个动作同为最优”后的 tie-aware 指标中，Q 最优动作率 60.0%，run4 policy 为 62.35%，29.4% 目标有最优并列。
`audit_response_q_policy.py` 给出的配对 Q−policy 最优率差为 **−2.35pp**，95% CI **[−15.87pp, +11.16pp]**；
policy−Q 后悔改善为 **+1.74** 分，95% CI **[−7.09, +10.57]**。两项都不通过，故 pilot 不进入真实 200 墙评测、
不晋升。

后续 direct-response Q 只有同时满足以下离线门槛才允许真实筛选：policy identity 与 run4 完全一致；按未见物理
牌墙的 Q−policy 最优率差和 policy−Q 后悔改善的 95% 下界都大于零；随后仍须在全新 200 墙、四座轮换中通过配对
净分正下界，才有资格做 400 墙复核。下一步只扩大互不重叠的 direct-response 墙组，不改 selector 或网页默认 AI。

### direct-response Q scale-a：三种排序目标的共同反证

独立 scale-a 使用 240 个全新 classic 物理牌墙，收集 **1,666** 个 response 决策与 **3,699** 条逐合法动作分支；
按墙隔离为 1,265 / 202 / 199 train / validation / test。未见测试包含 pass 131、chi 37、pong 27、hu 2、
ming_kan 2，平均动作回报跨度 26.53 分。所有 rollout 均为同一信息集的动作分支，且未导出牌墙或他家暗手。

先以四个独立 seed 的绝对 Q ensemble 在 run4 policy top-2 内重排：测试平均后悔改善 **+2.23** 分，但 95% CI
**[−1.78, +6.23]**；最优动作率差 **−1.51pp**，95% CI **[−8.13pp, +5.11pp]**。随后只用 validation 墙组尝试
0／1／2／4／8／12／16 分 Q 优势阈值；没有任何阈值使最优率和后悔改善的下界同时为正，故不对 test 做阈值挑选。

为检验绝对终局分数的共同噪声是否是瓶颈，训练器又加入按信息集中心化的 Q advantage regression，以及独立 Q
listwise soft-ranking loss；两者均使用 v6 独立 Q encoder/head，并启用 `--freeze-policy-path-for-q-only`。中心化模型
在 top-2 中只覆盖 15.1% 决策，测试最优率持平、后悔改善 **+0.21** 分，95% CI **[−1.53, +1.96]**；validation
阈值同样没有正下界。listwise 单模型虽有正均值（最优率 +1.01pp、后悔 +2.67 分），但四 seed ensemble 回落为最优率
**−1.51pp**，95% CI **[−8.83pp, +5.81pp]**，后悔改善 **+0.77** 分，95% CI **[−3.34, +4.88]**。

结论：在当前 run4 访问分布、完整隐藏 world 的单样本分支与约 1.7k response 信息集下，direct-response Q 没有可靠
超过冻结 policy 的排序证据。停止继续扩大该数据或调 Q loss；不做 200/400 墙、不晋升。下一阶段回到更根本的信息
集问题：先实现可验证、全公开历史条件化的 sequential belief proposal（必须先报告可接受率、ESS、与 toy exact
posterior 的校准），再考虑用其多 world 目标重新构造长程行动价值。网页默认维持 Teacher；run4 保留为历史
实验 checkpoint，不能据此宣称强度。

### 全公开历史约束修复 proposal：接受率恢复，但权重 ESS 淘汰

为区分“随机暗手下公开行动不合法”的 rejection 问题和行为似然本身的问题，新增 core 专用、audit-only 的
`core_public_history_constraint_repair_v0`。当他家的已观察弃牌/吃/碰/明杠缺少所需暗牌时，它只在未知墙和
非本家暗手间做物理牌守恒交换；不触碰本家手牌、私有摸牌 reservation、公开副露或导出数据，也没有接入 collector。

固定 seed 953 的 9-event 单元夹具，100 个 proposal 中朴素重放接受 **0** 个；修复后接受 **94** 个，平均
3.38 次交换。但以冻结 Teacher 行为概率加权的 ESS 仅 **3.55/94（3.8%）**，长前缀探针约 1%。此外交换 proposal
的密度尚未显式计算，不能把当前权重当作严格 posterior 修正。故该方案只证明错误来源，**不通过** belief 数据、
Q 训练或实战评测门槛；网页默认 Teacher 不变。下一步应先在可枚举小牌墙上实现带 proposal-density 的逐事件
条件化，再决定是否重启多 world 行动价值采集。

### 重要性修正的微型物理牌墙校准

为给下一种 proposal 建立不可绕过的数学门槛，`SequentialParticleBelief` 增加运行时 `initial_weights`，
用于输入 `p/q` 的显式重要性修正。新单测枚举含重复牌的微型牌墙：未知 `0,0,1,1`、对手两张暗手、观察弃 1；
真实后验中 `P(01|弃1)=16/19`。把只生成合法暗手的刻意偏置 proposal（`q(01)=3/4,q(11)=1/4`）连同每粒子
`p/q` 输入后，过滤器精确恢复 `16/19`。这仅是 proposal-density API 的校准，不是厦门麻将 belief 或训练样本；
真实全历史方案必须先满足同类枚举测试，才有资格替代当前 audit-only repair。

对同一 core seed 953、9 个公开 action、200 个固定 proposal 粒子的复核显示，单靠放宽确定性
Teacher fallback 的均匀平滑也不能挽救全历史权重：`uniform_mixture=0.02/0.10/0.25/0.50` 时，接受数均为
192，行为权重 ESS 比例仅为 **1.09% / 1.33% / 1.94% / 4.16%**。故不能用“更大 uniform noise”伪装出
健康 posterior；下一代方案需要可校准的行为能量模型与显式 proposal density，两者缺一不可。

为排除“整段 rejection 没有逐步重采样”的解释，又实现了 `core_public_history_sequential_smc_v0`：setup prior
保持 `p/q=1`、不交换暗牌，公开日志按玩家动作与其自动 draw/result 分组；每组按增量行为似然更新、低 ESS
时 systematic resample。seed 953 的 9-event fixture 中，**200** 粒子在完成 4 个公开 event 后耗尽；**1024**
粒子走完，但最小预重采样 ESS 为 **1/1024**，重采样后最终均匀权重不能掩盖该退化。该严格基线同样不通过
collector/Q 数据门槛；其作用是将下一步明确限定为“可计算密度且经小牌墙 posterior 校准的条件 proposal”。

### 精确 setup-claim density 的最窄正向验证（不进入训练）

新增 `core_initial_response_claim_exact_density_v0`：仅限 core 开局中“本家弃牌后另一家立即吃／碰／明杠”。
该家尚未摸牌，因此可对其初始暗手以多元超几何分布直接抽样，条件为包含公开副露消耗的牌。组合动态规划精确
给出条件概率，完整 setup 的修正为 `p/q=P(初始暗手包含所需牌)`；不是通过手工交换暗牌伪造一个 world。

固定 `seed=2` 的首吃前缀、256 粒子审计：普通 setup prior 重放为 **40/256** 接受；该 exact-density proposal
为 **256/256** 接受，`p/q=0.13146149`，加冻结 Teacher 平滑行为似然后的 ESS 是 **190.04/256（74.2%）**。
数学 micro-deck 单测也验证 `{0,0,1,1}` 抽 2 张且要求含 `1` 的条件概率为 `5/6`，条件后 `11` 的概率为 `1/5`。
这只是一个极窄 setup 事件的**构造正确性**验证：后续摸牌、弃牌、暗杠与多个事件尚未有 density，故不改变 run4、
不接入 collector/Q/网页，也不能解释为完整 history posterior 或模型强度提升。

为扩展 draw→discard，普通摸牌的 `public_actions` 已新增 `draw.tiles`：它按顺序记录该次摸牌公开补到的花牌，
不记录随后拿到的可打牌。重放先校验对手最终花数能由这些事件解释；旧记录或补杠／明杠后的未定位 replacement draw
仍拒绝为 `opponent_flower_history_unsupported`。小牌墙先验证 wall-only 条件 proposal：固定公开花序列、依剩余物理
张数抽取隐藏底牌，`p/q` 等于该观察的解析先验概率；其底牌条件分布也有独立回归测试。随后已将未知对手 setup
hand／flower-slot、花序列与公开弃牌的结构可行性联立；微型枚举验证联合概率。固定 core、dealer 0、seed 271、首弃
后下家补 `(40,)` 并弃牌的 256 粒子审计达到 **256/256** 结构接受，reference `p/q=0.0012275911710061692`，但
冻结 Teacher 行为权重 ESS 仅 **17.76/256（6.9%）**，不通过数据门槛。

随后开局翻金 transition 已补齐：将引擎骰位环形扫描／跳花规则抽为共享 `gold_indicator_index`，固定墙的
`P(indicator face)` 条件 sampler 与小牌墙完整枚举一致；“翻金 + 庄家已知首摸”还验证了联合结构概率。两个 core
重建夹具（seed 271、seed 32，其中后者首摸补花）均保持本家手/花、公开金指示牌/骰子、牌墙长度与 144 张物理牌
多重集。随后 opening 与首位对手 draw→discard（含公开弃牌可行性）合并为层次 proposal；标号微型牌墙的枚举先验
为 **1/450**，proposal 的平均 per-particle `p/q` 与之相符。真实 core seed 271、256 粒子审计已达 **256/256**
结构接受，但 `p/q` 随隐藏分配为 `2.5967952709322814e-07` 到 `1.0387181083729125e-06`，Teacher 行为加权 ESS 仅
**9.55/256（3.7%）**，比旧 reference 更低。因此继续拒绝 collector/Q/网页；普通重放仍以
`opponent_flower_transition_unsupported` 安全拒绝。下一步先校准行为条件 proposal，而不是继续延长该前缀。

## 2026-08-07：人类对局评测与数据采集入口（默认关闭）

“胜过人类”不能由 Teacher 配对分数替代。网页现在支持显式 `--human-log local_human_data/*.jsonl`：只在
玩家完成一局且至少行动一次后，追加玩家可见状态、合法动作、实际人类选择、公开 action 历史和本局分差。
牌墙顺序、AI 暗手、随机种子、账号、网络标识和时间戳都不写入；本地目录被 Git 忽略。记录来源固定为
`local_human_opt_in`，默认不进入 Teacher/DAgger 训练，须单独质量审查与按完整牌局留出评测。显式 checkpoint
对手只以文件 SHA-256 记录，不保存本地路径；`audit_human_trajectories.py` 会在人工评审前拒绝混合规则/对手、
重复手牌、重放种子、私有字段和非本局分差记录。

审计报告另对**结构合格**的完整局计算人类 seat 的本局平均分差、样本标准误、正态近似 95% 区间、胡率和
流局率。候选对手的 `agent_profiles` 也明确记为 `explicit_policy_value_checkpoint`，不会伪装成 Teacher；对手
的可比身份仍以 SHA-256 为准。这只是未来真人 A/B 的描述性汇总，既不检验记录者水平，也不构成“模型胜过
人类”的结论。

为进行受控试玩，`serve_web_game.py --ai-checkpoint <policy-value.pt>` 会将三名 AI 显式替换为该 checkpoint，
网页标识为 `EXPLICIT EXPERIMENTAL CHECKPOINT`；没有参数时仍是 Teacher。显式 checkpoint 已通过“加载、响应、
人类一手、AI 自动推进”的规则引擎冒烟检查。该路径仅提供未来人类 A/B 与数据采集能力，不构成任何 checkpoint
战胜人类的结论。

## 2026-08-07：信息集 rollout 规则基线（初筛未通过）

新增 `InformationSetRolloutAgent` 作为与训练数据无关的在线诊断：对当前行动者的每个合法动作，从本家可见
事实重采样暗手/牌墙，在同一 world 内强制动作、其后回退 Teacher 结算；live game 的真实墙与对手手牌不读且
不修改。单测确认动作合法、live hidden state 不变，以及只替换 live 未见牌不会改变固定 RNG 下的选择。

全回合 mode 每副物理墙四座约需 20 秒，且首个单墙净分为 **−14.75**，不具可用性。收缩到默认的 response-only
mode 后，1 belief world 在预注册 10 墙、经典换座筛选相对 Teacher 为 **−1.775 ± 6.692**（95% CI
**[−14.892, +11.342]**）；2 worlds 在另一 5 墙筛选为 **−9.30 ± 5.704**（**[−20.480, +1.880]**）。均无正向
证据，故不接入网页、不开更大筛选，也不产生训练数据。该组件仅保留为日后有经校准 belief/value 时的安全
信息集搜索基线。

## 2026-08-07：独立回归评测——淘汰 run4，冻结 Teacher 默认

为检验“DAgger 混合验证集更高一致率”是否真的提高实战强度，使用此前未参与训练或 checkpoint 选择的
经典档 **80** 个物理牌墙（seed `202608081` 起），每墙四座轮换；推理固定为 CUDA、policy head。评测 CLI
现在同时记录 checkpoint SHA-256，避免同名参数文件被替换后让历史结果失去身份。

| 候选 | 对三名 Teacher 的均分（每局） | 胡率 | 结论 |
| --- | ---: | ---: | --- |
| run3 | **+1.3375 ± 1.3770**，95% CI **[−1.3615, +4.0365]** | 26.25% | 方向为正，但区间跨 0；只可作显式实验候选。 |
| run4-dagger | **−2.3813 ± 1.1274**，95% CI **[−4.5911, −0.1714]** | 22.19% | 相对 Teacher 已显著为负，不可试玩默认或再作为训练起点。 |

同一批牌墙的配对 run4 − run3 为 **−3.7188 ± 1.5691**，95% CI **[−6.7942, −0.6433]**，因此 run4 的退化不是
牌墙难度差异。可复核评测产物为 `artifacts/evaluation-run3-vs-teacher-independent-202608081.json` 和
`artifacts/evaluation-run4-vs-teacher-independent-202608081.json`；两个参数 SHA-256 分别为
`01616a38ff34af71f1774995d5d3f9e08c4408fd85757ad563d9dfa339b19f70` 与
`e8a23464717762421ff6616c95d4b290188864d8d0363a0dac19f80500e39909`。

同日对 `InformationSetRolloutAgent` 做了另一组经典 10 墙 response-only 诊断（1 个 belief world、seed
`202608071`）：**−4.975 ± 4.862**，95% CI **[−14.504, +4.554]**。这与先前的初筛一样没有正向证据；不扩大
搜索、不让它产生训练标签。下一轮优化应重新构造可验证的强度来源（经审计的人类数据、或通过联赛门槛的
自博弈），而不是继续在 Teacher/DAgger 轨迹上微调。

### paired-LCB 信息集搜索：控制选择偏差后仍未通过筛选

原始 rollout 对每个动作取少量 determinization 的最高均值，存在典型的“在噪声中选最大值”偏差。为使这条
规则搜索基线本身更保守，`InformationSetRolloutAgent` 新增实验性 `selection_mode="paired_lcb"`：每个候选
和 Teacher 回退动作必须在**同一批** world 中结算，只在候选的 paired mean delta − `z × stderr`（本轮 `z=1`）
严格为正时才覆盖 Teacher；不足两个 world 时必定回退。该模式不改变既有默认 `mean`，也没有接入网页。

经典档 response-only、2 个 belief world 的预先隔离筛选使用 seed `202608121`–`202608160` 共 **40** 个物理
牌墙、四座轮换，分为八个 5 墙可复核分片执行。候选对三名 Teacher 的均分为 **−3.5688 ± 2.3135** 分/局，
95% CI **[−8.1032, +0.9657]**。下界不为正，且点估计为负，故淘汰：不扩大至终检、不生成训练标签、不替换
Teacher。保留的只是经过单元测试的保守选择机制，日后只有在校准 belief/value 能降低 world 方差时才可重审。

### Teacher 单参数探针：没有实际决策覆盖，不进入代码

为排除把静态启发式系数误当作优化，三类临时 probe 都直接以四座轮换对抗默认 Teacher：弃金惩罚
`0/14/21`、吃碰允许的 hand-quality 损失 `0/1/2`，各在同一经典 10 墙中均得到严格 **0** 总分差；听牌张数
权重 `9/27/36` 也在 10 墙中严格为 0，权重 36 在另一 40 墙仍严格为 0。检查初始 300 个随机局面，只有把
权重夸大到 0 或 1000 才出现 1 个不同弃牌。结论是这些邻域参数并不覆盖 Teacher 的实际排序边界，临时参数化
已撤回；后续只研究会产生足够行为覆盖、并能独立评测的状态/价值改进。

## 2026-08-07：固定对手阵容联赛基线（只作筛选，不晋升）

单独“对三名 Teacher”的结果不能检验候选是否只适应了 Teacher。评测器现接受恰好三名命名对手；对每个物理
牌墙，候选轮换四座，三名对手被赋到其余绝对座位，因此每名对手会覆盖候选周围的三个相对位置。配对比较额外
拒绝不同的对手标签，CLI 会记录所有 checkpoint 的 SHA-256。

首个经典档筛选阵容为 `[run3, run4-dagger, Teacher]`，seed `202608271` 起 **20** 个物理牌墙、四座轮换、
CUDA policy head。绝对均分依次是：Teacher **+0.6875 ± 2.2509**、run3 **−0.4000 ± 1.8075**、run4
**−1.7250 ± 2.2914** 分/局。相对同墙 Teacher，run3 为 **−1.0875 ± 2.8592**（95% CI
**[−6.6914, +4.5164]**），run4 为 **−2.4125 ± 3.1959**（**[−8.6765, +3.8515]**）。

这只是 20 墙筛选，所有区间均跨 0，既不建立模型强度排序，也不授权更多 Teacher/DAgger 微调。可复核产物为
`artifacts/evaluation-league-{teacher,run3,run4}-run3-run4-screen-202608271.json`。今后 PPO 或其他候选须先
在 Teacher 基准和固定异质阵容两者中通过筛选，才允许进入 200/400 墙终检；这仍不等同于胜过人类。

## 2026-08-07：当前策略快照 PPO（小规模消融：拒绝）

参考四人不完全信息牌类的近期自博弈工作，PPO 增加“本轮更新前 actor 的冻结快照”对手。快照在每轮采样前
复制，不共享参数或梯度，更新后才刷新；所有对手抽样、源代码 revision 和各类型计数写入 report。该消融从
run3（SHA-256 `01616a38…b19f70`）出发，连续三个独立 stage，每个经典档 128 局、32 路 batch、默认 PPO
超参数，Teacher/current-snapshot 概率各 0.5。三段共 **384** 局、**4,273** 个候选决策；各段的实际对手数为
Teacher/current snapshot `178/206`、`190/194`、`196/188`。训练时的回报不作为强度指标。

冻结 stage3 参数（SHA-256 `c39d3a63f9ab7b6c4f3220161162261dd82df68e5697ee5711a0c1fddcc47d23`）在全新
Teacher 筛选的两段 40 墙（合计 **80** 墙）为 **+0.7406 ± 1.7495** 分/局，95% CI **[−2.6885, +4.1698]**；
同墙 run3 为 **−0.5031 ± 1.5810**。候选 − run3 的配对差为 **+1.2438 ± 1.4029**，95% CI
**[−1.5060, +3.9935]**，不通过第一道正向下界。

固定 `[run3, run4-dagger, Teacher]` 阵容的另一组全新 20 墙筛选中，候选绝对均分 **+1.1375 ± 2.2310**；
相对 Teacher 为 **+0.8375 ± 2.9100**（95% CI **[−4.8661, +6.5411]**），相对 run3 为
**−1.1875 ± 1.7310**（**[−4.5802, +2.2052]**）。双门槛均不通过，故不做 200/400 墙终检、不晋升、不作为
下一轮训练起点或网页 AI。产物保留在 `artifacts/torch-ppo-classic-current-selfplay-v1-stage{1,2,3}/` 及
`artifacts/evaluation-{torch-ppo-classic-current-selfplay-v1,run3,league-*-current-selfplay}-*.json`，用于复核而非部署。

## 2026-08-07：公开信息一步前瞻规则候选（初筛拒绝）

为测试“静态弃牌后牌形”是否是 Teacher 的主要局限，新增 `OnePlyLookaheadTeacherAgent`。它只看本家手牌、
四家河/副露和翻金：对每个弃牌枚举下一次可能的可打牌，并按公开可见后的剩余张数加权；若下一摸成和，则用与
引擎同尺度的自摸结算值，否则取下一步合法弃牌的廉价牌形值。单测通过“交换对手暗手与牌墙中的未见牌，动作
不变”验证信息边界；它不读取墙或任何对手暗手，也没有接入网页或数据采集。

经典档 seed `202608401` 起的 **20** 个独立物理牌墙（80 个四座换位对局）已足够否定该启发式：一步前瞻候选
对三名 Teacher 的均分为 **−21.4000 ± 2.5405**，同墙相对 Teacher 为 **−21.4000 ± 2.5405**，95% CI
**[−26.3794, −16.4206]**，胡率仅 2.5%。这是强负结果，不以调权重挽救：不扩大终检、不生成训练标签、不作为
任何模型的 Teacher、不替换网页默认。完整逐局分数和配对统计见
`artifacts/evaluation-one-ply-lookahead-teacher-pilot-202608401.json`；CLI 仅保留
`--one-ply-lookahead-teacher-candidate` 以便复核。

## 2026-08-07：人类数据训练闸门（基础设施，不是强度实验）

本地工作区当前没有 `local_human_data/` 对局，因而没有任何真人强度或人类监督结论。审查训练入口发现：尽管网页
记录带有 `training_default=excluded_until_separate_quality_review`，手工把该 JSONL 传入旧训练 CLI 时仍会被当作
普通 hard-label 样本读取。现已修正为 fail-closed：检测到 `collector=local_human_opt_in` 默认报错；操作者必须显式
传 `--allow-local-human-data`，并让所有人类输入共同通过现有隐私/重复/规则档位/对手身份/最小局数审计。混合人类和
其他来源的同一文件也会拒绝，checkpoint report 仅留下无路径的聚合审计和人工授权标志。

这只建立将来经过人工复核的人类**行为**安全地用于 policy imitation 的路径；人类终局分数尚未用于 value regression，
也没有任何模型因该基础设施而变强或获得“击败人类”主张。下一步仍需要独立、足量且多来源的真人留出对局，或在
不依赖伪人类标签的自博弈联赛中取得可复现优势。

## 2026-08-07：run3 的 400 墙 Teacher 终检（未通过）

run3 曾在独立 80 墙上得到正向但跨零的点估计，不能据此作为“当前最强”继续累积训练。为一次性解决这个不确定性，
冻结参数 SHA-256 `01616a38ff34af71f1774995d5d3f9e08c4408fd85757ad563d9dfa339b19f70` 在完全未参与训练、选择或
此前评测的经典 seed `202608431` 起 **400** 个物理牌墙上对三名 Teacher 四座轮换（共 1,600 局、CUDA policy
head）终检。结果为 **−0.8550 ± 0.6448** 分/局，95% CI **[−2.1189, +0.4089]**，胡率 **23.8125%**，另有 5 局流局。

下界不为正、点估计也转负，因此 run3 不升级、不替换网页 Teacher、也不再被描述为下一轮优化的最强冻结起点。该
结果不宣称 run3 已显著弱于 Teacher（区间仍跨 0），但足以否定其具有已验证优势。完整可复核产物为
`artifacts/evaluation-run3-vs-teacher-terminal-202608431.json`。

## 2026-08-07：Teacher 单点 response 干预收集器（基础设施）

既然 run3 与 run4 都不再是可验证的最强基线，扩展 run4 后状态 selector 或从其继续 PPO 都不符合证据。为获得
直接对网页默认 Teacher 的因果数据，`collect_candidate_teacher_dagger_trajectories.py` 现允许 `--teacher-base`。
它以 Teacher 作为干预前、干预后和三名对手的冻结策略；只在一个预选 response 位置按已知
`ε × uniform + (1−ε) × Teacher` 行为采样，并导出真实执行动作和其 propensity。manifest 显式写入
`behavior_base=heuristic_teacher`、无 checkpoint／无推理设备，避免将规则 Teacher 伪装成神经参数来源。

core 小规模 smoke（3 个物理墙、12 个换座手）确认该路径可完成结算、生成安全 split 轨迹，并保持规则 Teacher
身份。尚未收集经典训练墙、训练 outcome/Q 头或测试 selector，因此这不是模型或强度实验。后续先做独立 OPE/支持度
审计，再决定是否值得生成足量经典数据。

## 2026-08-07：Teacher response 单点 OPE 审计器（基础设施）

新增 `xiamen_mahjong.off_policy` 与 `audit_teacher_response_intervention_ope.py`。每个有效观察必须有完整的
Teacher epsilon-mixture propensity、受保护的 `split_group_id` 和一个真实终局分数；IPS 与 doubly-robust delta 均按
物理牌墙组先平均、再计算标准误，避免把同一牌墙的四座轮换误当独立样本。DR 的 direct value 只允许来自不接触
test 墙的 outcome ensemble；候选 action 用跨成员终局分数 LCB 与 Teacher 标签比较，从而优先保持 Teacher。

输出状态固定为 `audit_only_one_response_teacher_override`。即使 IPS、DR 两项 95% 下界为正且两侧 ESS 达标，也只可
进入「一局至多一次 response override，随后 Teacher 后缀」的 200 墙筛选，不能作为完整策略、网页部署或打败人类的
证据。当前只有脚本与单测，尚无经典 Teacher 干预数据、outcome checkpoint 或 OPE 数值结果。

## 2026-08-07：Teacher response 干预数据与首轮 OPE（否决）

收集 `teacher-response-intervention-classic-v1`：800 个独立经典物理墙、四座轮换共 3,200 局，按墙组为
553/134/113 个 train/validation/test。安全 JSONL 不含随机种子、墙或隐藏手牌；三分区 `split_group_id` 交集均为
零。随机 response 记录为 **1,341/332/282**，测试动作覆盖吃 125、碰 88、过 53、胡 8、明杠 8，propensity 范围
0.1–0.8。原始 JSONL 保留在本机训练盘（train 文件 106 MB，超过 GitHub 单文件上限）；manifest 与 SHA-256 为
`c5dcc1d20697b8213dc5d961d74f9a069e3e1d786485083c81213965b7cc2034`（train）、
`97d4b72c0875323f08e6ad784bbe58398c562d53f8136cbaddfc36e3c88dadb8`（validation）、
`6dd701fc4e04340a491adbe446173f0a90c20606bd5bcec7ce6d90ef1efcdb77`（test）。

为避免任何历史 checkpoint 重新成为训练起点，以相同的**从未训练** policy anchor（seed 202608471）训练五个
afterstate outcome 成员，分别只更新其独立的 afterstate encoder/score/win heads。独立 test 的 score MAE 为
30.70–32.22 分，零预测为 32.85 分；这是 logged-action 校准改善，仍非选牌许可。

在 108 个有有效干预的 test 墙组（282 条随机 response）上，预先固定的 `LCB(z=1) 最大 score，否则 Teacher`
一次 override 候选产生 IPS **−1.6428 ± 4.4087**（95% CI **[−10.2839, +6.9983]**）与 DR
**−0.7897 ± 4.6188**（**[−9.8425, +8.2632]**）。target/base ESS 为 **66.1/218.7**，支持度足够但两个下界均不为正。
因此明确拒绝，不进行 200 墙、网页试用或任何强度主张；这个 test 墙组已冻结，不能用于事后调 LCB 阈值或选择成员。

## 2026-08-07：四层盲态选择协议与第二轮 LCB 网格（否决）

为避免把一份 test 既用于阈值搜索又用于强度结论，新增 `split_heldout_trajectory_groups.py`：强制按
`split_group_id` 将已隔离的 held-out 池分成 `selection` 与 `terminal`，terminal manifest 不导出动作或终局统计。
新增 `select_teacher_response_override.py`：它只在 selection 比较预注册网格；没有任何候选通过时，程序不读取
terminal 文件并写入 `selection_rejected_terminal_unread`。训练器同时加入 `--skip-test`，使 outcome 模型的 epoch
选择只接触 train/validation，report 显式保存 `terminal_test_read=false`。

第二批 `teacher-response-intervention-classic-v2` 使用 1,600 个全新经典物理墙；训练/验证含 2,332/583 条随机
response，selector 含 522 条、200 个墙组，且 Teacher 基线／propensity 契约均为零违规。五个全新随机 anchor
outcome 成员在 validation 的 score MAE 为 31.52–32.40 分，零预测为 34.97 分。此改善仍只针对 logged action。

在**打开 selector 结果前**固定 `{0, 12, 24, 36}` 分 LCB 优势网格。0 分候选 override 35.63%，IPS/DR 为
−2.5208/−2.8229，95% 下界 **−7.7110/−7.5706**；12 分候选 override 3.45%，下界 **−1.5236/−1.1817**；24 和
36 分不再偏离 Teacher，估计恰为 0，仍不满足严格正下界。所有候选均失败；terminal 209 个墙组（文件 SHA-256
`12ec74325811ac96457fbcc4ac03ec7f6b055d1ff3563ea57f7bfd11614a144f`）未读取、不提交结果、不允许改阈值后重跑。
结论是当前静态 afterstate score-LCB 候选族没有正向因果证据，而不是 Teacher 变弱或模型已经具备人类强度。

## 2026-08-07：Teacher 行为似然代理的 SMC smoke（否决）

为检验全历史 SMC 的低 ESS 是否只是确定性 Teacher likelihood 太尖锐，独立训练 core Teacher 行为 MLP（100 训练局、
40 独立验证局、16 隐层、4 epoch）。验证集 top-1 为 **76.84%**、交叉熵 **0.7254**，因此它足以作为“软概率是否缓解
退化”的 audit-only 候选，但不是游戏策略。

固定 core seed 202608503 的同一私有回放 snapshot、64 个 actor-visible setup 粒子、相同温度 1 与 uniform mixture
0.02 下，确定性 Teacher sequential SMC 在 8 个请求公开事件中只条件到 3 个，最终 ESS **3.01**；行为代理同样只到
3 个，最终 ESS **2.86**。两条路径都因后续事件在 proposal world 中结构不一致而全粒子失败。故软化行为 likelihood
没有改善、也不能修复提议分布；不提交到 SMC collector/Q/网页。下一阶段必须先改进带已知密度、可枚举 toy posterior
校准的结构 proposal，而不是继续训练代理或调温度。

## 2026-08-07：opening claim 精确 proposal 与行为相容性诊断

加强了 opening-aware structural proposal 的证据门槛：微型标记物理牌墙不只验证 `P(事件)`，还穷举目标 posterior，
并确认 proposal 的逐粒子 `p/q` 加权后能恢复暗手和翻金后墙位的边缘分布。该检查覆盖首个 claim 所需初始暗手，
以及 claim 后立即弃牌所需的联合初始手牌多重集；所有测试只在内存运行，不导出暗牌或墙。

新的 audit 将 structural ESS 与包含行为似然的 total ESS 分开。core `seed=271`、256 粒子、RNG `202608504` 的
opening→首次对手摸打为 256/256 接受，structural ESS **248.76/256**，total ESS **15.28/256**；旧总 ESS 的低值不应
再误读为该 `p/q` 的退化。core `seed=2`、256 粒子、RNG `202608505` 的 opening→claim 为 255/256 接受，structural
ESS **246.33/255**、total ESS **187.88/255**，但覆盖范围仅两条动作。

最初将同一 core `seed=2` 延长到 claimant 下一弃牌（RNG `202608506`）所见的 32/256 replay 接受已撤销为无效诊断：
该弃牌无人可响应时，引擎会在同一 transition 自动产生下一 normal draw，旧代码在 draw 前停止导致 224 个
`public_event_mismatch`。recognizer 现只接受弃牌后仍有 actor response option 的原子边界；不具备该条件者会在
未建模下一 draw 的 density 前安全跳过。

修正后，独立 core `seed=67`、256 粒子、RNG `202608507` 的 claim→discard 前缀为 **256/256** 接受，structural ESS
**250.74/256**、total ESS **33.30/256**。因此这条更长 prefix 的低 total ESS 确为行为 likelihood 退化，而非事件边界
错误。该 audit 仍没有进入 SMC、collector、动作价值、训练或网页；对确定性 Teacher 动作做未归一化 rejection 仍不是
合法的下一步。

为探索同时保留归一化的行为导向 proposal，新增 tile-factor 条件多元超几何 sampler：每种面额的正权重用 count-DP
精确归一化，返回条件 event 概率和 conditional `p/q`；toy 后验测试确认重要性加权可还原原始超几何分布。它只是一个
数学 primitive，不是行为模型。

在打开结果前固定 factor `{1, 1.5, 2, 3}`，只提高 claim 后公开弃牌面额的额外副本权重；墙组为 core `67,139,201,275`，
每个 factor/墙 256 粒子。平均 structural ESS fraction 为 **0.9787、0.9655、0.9441、0.8801**；平均 total ESS
fraction 为 **0.1334、0.1399、0.1252、0.0979**。所有候选都有远低于 20% 的单墙 total ESS，故此 tile-only family
整体失败；1.5 的小幅同组差异不用于选择或二次调参。没有生成训练数据、模型参数或网页行为。

## 2026-08-07：独立线性行为能量 proposal（终检否决）

以独立 core Teacher 数据（seed `202608520`，300 训练局 10,476 决策；独立 100 局 3,558 决策）训练
`RulePolicyModel`。其验证 top-1 为 **67.09%**、loss **1.0179**。对固定公开弃牌，模型的 `target × hand_tile`
线性系数被转换为 clipped `exp(scale × coefficient)` tile factors；DP 精确返回条件 hand proposal 的归一化和 `p/q`。
公共特征不会进入 factor，原始训练 JSONL 保持 Git 忽略；仅模型参数、报告和 SHA-256
`80587fcbc108888b884a39b59b4462040c6d54e618e88aa69093042adc0edfd3` 可复核。

selection 墙 `390,492,499,585` 与训练及旧 audit 墙不重叠。预先固定 scale `{0,0.25,0.5,1}`、clip=2、每墙 256：
平均 total ESS fraction 分别为 **0.2358、0.3187、0.3319、0.2395**，据此唯一选择 0.5。全新 terminal core `seed=707`
以 1,024 粒子复核，接受 1,023，total ESS **226.17/1,023 (22.1%)**，但 structural ESS 仅
**158.26/1,023 (15.5%)**。joint structural+total ESS ≥20% 门槛失败，故整条线性能量 proposal 路线拒绝；不导出
belief world、不生成训练数据、不训练/部署任何选牌模型。

## 2026-08-07：Progressive hiding 字段隔离（基础设施）

训练脚本新增固定维度的 `oracle → hide_wall → visible` 遮蔽向量，作用域仅为训练进程；visible stage 把三家暗手及
墙组成的片段置零。对同一 actor-visible core 局面重采样不同未知世界：oracle 向量不同、visible 向量严格一致，且
unknown stage 直接报错。actor `TeacherDecision`、轨迹、checkpoint、网页均未新增该向量；legacy privileged critic
仍保持 oracle-only。没有进行 progressive-hiding 模型训练、PPO、数据导出或强度评测。

## 2026-08-07：Progressive hiding critic 校准 smoke（拒绝，未训练策略）

字段隔离本身不足以说明课程有用，因而先执行一个可证伪的最小校准实验，而不是恢复 privileged-critic PPO。
core 固定 seed `202608560` 下，一个**全新随机**、未更新的合法动作 actor 对三名 Teacher 产生 64 个训练局
（554 个本家决策）与不同牌墙的 32 个验证局（269 个本家决策）。隐藏向量仅驻留进程内；没有写出暗牌向量，
没有 actor 或 critic checkpoint。两个 32 隐层 critic 从逐字节相同的初值出发、使用相同训练决策及相同按 epoch
打乱序列：direct baseline 全程以 visible 输入训练 9 epoch；课程候选按 `oracle → hide_wall → visible` 各 3 epoch。

唯一的预注册门槛是独立验证上的**最终 visible** Huber 与 MAE 均严格低于同预算 direct baseline。结果相反：
direct 为 Huber **0.1707421**、MAE **0.4069980**；课程为 **0.1715056**、**0.4072743**，差值（课程−直接）为
**+0.0007634**、**+0.0002763**。两项都变差，状态
`rejected_before_policy_training`。完整无私有特征报告为
`artifacts/progressive-hiding-core-smoke-v1.json`，生成脚本为
`scripts/audit_progressive_hiding_critic.py`（source `b025af5`）。

这否决的是该预注册的 core/9-epoch 课程作为进入策略训练的依据，并不把小样本差异解释为 progressive hiding 的一般
定理。若日后重访，必须先建立真正缩小规则的独立基准和新的预注册 schedule／墙组；不得在本报告的训练或验证墙上
调节阶段比例、学习率后声称通过，也不得以此报告为依据训练 PPO、导出模型或作任何强度主张。

## 2026-08-07：Teacher 单点弃牌干预 v1（terminal 否决）

此前 response-only 干预覆盖过窄，故新建严格 phase-isolated 的 `discard`（摸后决策阶段，包含可用的胡／杠）
单点因果路径。预注册先于数据生成提交：经典档连续 1,600 个物理墙 seed `202608700` 起、四座轮换共 6,400 局，
三名对手和前后缀均为 `HeuristicTeacherAgent`；每局仅在前 8 个该阶段决策随机选一处，以
`0.4 × uniform + 0.6 × Teacher` 执行并记录 exact propensity。按物理墙分出 train/validation/held-out，后者再按墙组
拆成 selection/terminal，均无组重叠。原始 JSONL（296 MB）只保留本地；GitHub 只存 manifest、无标签覆盖统计和最终审计。

随机干预数为 train **3,003**（968 墙）、validation **727**（228 墙）、selection **579**（191 墙）、terminal
**581**（190 墙），均超过预注册最少 2,500/500/500/500。五个相同 fresh policy anchor、不同训练 seed 的 128 隐层
afterstate outcome 成员只训练独立 outcome encoder/heads，均传 `--skip-test`；validation score MAE 为
**29.03–29.36** 分，优于零预测 **33.99** 分，因而允许打开 selection。所有五组参数与报告已提交，均明确
`terminal_test_read=false`。

selection 仅比较预注册 LCB 优势阈值 `{0,8,16,24}`（5 成员均值减 1 倍成员标准差），要求 grouped IPS 和 DR 的
95% 下界均正、两侧 ESS ≥50。0 与 8 分候选下界为负或 target ESS 不足；24 分同样失败。16 分阈值仅以
**1.73%** 的 override rate 通过 selection（IPS/DR 下界 **+0.049/+0.099**，ESS **372.6/378.6**），因此按既定规则
唯一进入 terminal。

独立 terminal 的同一阈值立即失败：IPS **−0.0148 ± 0.1665** 分/局，95% CI **[−0.3411,+0.3115]**；DR
**+0.1392 ± 0.1788**，**[−0.2113,+0.4897]**；target/base ESS **372.7/376.7**。支持度充足而双下界为负，
故拒绝该静态 score-LCB 单点弃牌候选，不做 200 墙实战筛选、PPO、网页接入或强度主张。terminal 墙组已消耗，
不得在此数据上更换阈值、成员、LCB z 值或模型后重跑。完整 protocol、覆盖与选择报告分别为
`knowledge_base/teacher_discard_intervention_v1_protocol.md`、
`artifacts/teacher-discard-intervention-classic-v1/coverage.json` 与
`artifacts/teacher-discard-intervention-classic-v1-heldout/selection-result.json`。

## 2026-08-07：原始 DR 相对优势诊断（拒绝直接训练）

为给下一候选族建立正确统计基元，新增每个合法动作相对 Teacher 的 DR 伪优势：直接模型差
`q(a)−q(Teacher)`，再对 logged action 与 Teacher action各加一次已知 propensity 的 residual。五个 outcome 模型均只
见过 train 墙，因此在 v1 validation 的 727 个随机弃牌干预／228 个墙组上可以安全做 aggregate-only 审计；没有读取
selection/terminal、没有训练 advantage head 或选择动作。

结果否决了朴素的未缩减伪标签：7,135 个非 Teacher 动作的直接模型 gap 标准差仅 **11.33** 分（p99 **14.40**），
而 DR 伪优势标准差为 **206.66** 分，范围 **[−1,737.25, +3,251.66]**；logged action 伪优势标准差更为 **628.66**。
相应非 Teacher propensity 的中位数为 **0.0333**，范围 **0.0235–0.2**，重要性残差尾部放大与此一致。故不以原始 DR
向量训练 action head、不做 AWR/PPO 或网页选择。完整无私有特征报告为
`artifacts/teacher-discard-intervention-classic-v1/dr-advantage-validation-audit.json`。

未来若再研究相对优势，必须使用按物理墙 cross-fitting 的 direct model，并在**全新**四层墙组上预注册有限的
shrinkage／SWITCH 类方差控制与偏差审计；不能在已消耗 v1 selection/terminal 上调 clipping 或 threshold。
