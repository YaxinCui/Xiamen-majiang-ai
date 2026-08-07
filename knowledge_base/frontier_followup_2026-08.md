# 前沿复盘：从论文到厦门麻将的下一步

调研日期：2026-08-06。本文补充已有的《麻将 AI 前沿与开源实现》，重点回答：当前项目该借鉴什么、
不该过早做什么，以及下一轮工程投入的顺序。

## 结论先行

当前不应直接上 Deep CFR、全局 MCTS 或更大的 Transformer。最有价值的工作依次是：

1. 让反事实 rollout 能批量、高吞吐且精确复放；
2. 以**完整公开历史**做顺序粒子过滤，构造可审计的信息集价值目标；
3. 保留动作价值的分布/不确定性，再以保守的 AWR 离线更新策略；
4. 在数据规模足够后，再做事件序列 Transformer、动作分层和有限深度的 belief search。

这是因为现有候选网络的 Q 头在留出集上的排序仍不稳定；在这种前提下直接用 `argmax Q`
选牌，或用其驱动 PPO，只会放大目标噪声。当前更缺的是可信标签与样本量，而非表达能力。

## 2026-08-07：联赛与当前快照自博弈补充

新增调研的 [Big 2 四人不完全信息自博弈论文](https://arxiv.org/abs/2605.28863) 报告，在其受控环境和有限
预算中，当前策略自博弈优于 checkpoint 自博弈或固定对手，并观察到适度熵正则可避免过早确定化。这是与本项目
四人、隐藏信息、变长合法动作最相近的新**工程信号**，但仍是另一种牌类、预印本和其自身奖励设定，不能外推为
厦门麻将结论。因此项目只采纳一个可证伪消融：每轮 PPO 采样前冻结当前 actor 快照，按显式概率与 Teacher/旧
checkpoint 混合；不得共享梯度，也不得因为训练回报提升而晋升。

同时参考 [RiichiEnv](https://github.com/smly/RiichiEnv) 的 Gym 风格、Rust 核心和事件观察接口，确认“规则
状态转移与观察/策略接口分离”仍是规模化路线。它是日麻项目，不移植规则、代码或权重；厦门引擎短期保持 Python
权威实现，只先把训练和评测建立为可复核的固定对手阵容。新 `evaluate_policy.py` 已支持该阵容：候选和每名命名
对手在一副墙的四座轮换中都覆盖相对座位，配对比较拒绝混合阵容。

## 外部工作及可迁移部分

| 工作 | 已验证的方法 | 对本项目的采用判断 |
| --- | --- | --- |
| [Suphx](https://arxiv.org/abs/2003.13590) | 海量监督预训练后自博弈策略梯度；以完整暗牌/牌墙构造 oracle guiding，并逐步去除特权信息；还使用候选动作的前瞻特征。 | 采用“可见 actor + 训练期特权 critic/辅助目标”的课程思想；不将暗牌特征、oracle 动作或单个真实牌墙的结果当作部署标签。先扩充现有可见前瞻特征与反事实数据，再考虑小步 RL。 |
| [MahJax](https://arxiv.org/abs/2605.20577) / [源码](https://github.com/nissymori/mahjax) | JAX 向量化 `reset/observe/step` 环境，支持 BC 与 PPO；其论文以大规模 GPU 并行训练为重点。 | 借鉴批量环境接口、合法动作掩码和吞吐优先的架构；不移植日麻规则或权重。先在既有厦门引擎外建立批量 rollout 调度层，而非重写规则引擎。 |
| [Mortal](https://github.com/Equim-chan/Mortal) / [训练文档](https://mortal.ekyu.moe/) | Rust 模拟器、批量推理、offline/online 训练分段。 | 仅研究部署与训练编排。项目采用 AGPL-3.0，不能复制代码或混入源码；若未来想复用，必须先由项目方接受相应许可证义务。 |
| [Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) | 离线 AWR（可选 BPPO）后，再用带重要性校正、PPO clip 的 online policy gradient。 | 借鉴“先保守离线、后在线”的顺序。对本项目，AWR 必须以不确定性合格的动作价值为权重，并保留 Teacher/DAgger 的行为克隆锚点；不能从当前低质量 Q 头直接切换。该项目同为 AGPL。 |
| [Tjong](https://www.researchwithnj.com/en/publications/tjong-a-transformer-based-mahjong-ai-via-hierarchical-decision-ma/) | 将动作类别和牌目标分层，并以 Transformer 表示决策历史。 | 以后在候选动作排序器上增加“动作类别/目标牌”的辅助头；数据到达至少数十万条高质量决策前，不重试大 Transformer。规则引擎仍是最终合法性约束。 |
| [History Filtering in Imperfect Information Games](https://arxiv.org/abs/2311.14651) | 用历史上的策略到达概率更新隐藏世界；一般情形下由公开状态直接重建历史很困难，建议采样/过滤。 | 这直接否定了“只按最近弃牌重洗未知牌”即可得到真实后验的想法。应先实现从发牌到当前的顺序粒子过滤，并在小玩具规则上做精确后验校验。 |
| [ReBeL](https://arxiv.org/abs/2007.13544) / [Student of Games](https://arxiv.org/abs/2112.03178) | public belief、价值网络与搜索结合。 | 公共历史/信念状态的表示值得采用；两人零和的理论保证不能外推到四人一般和厦门麻将。不能据此宣称纳什收敛，也不应现在照搬 CFR。 |
| [Progressive Hiding](https://arxiv.org/abs/2409.03875) | 先在更多信息可见的阶段学习游戏机制，再逐步增加信息约束；论文给出与 CFR/非完全记忆相关的理论与小型数值验证。 | 它提供了“不把尚未校准的完整 posterior 直接塞进 actor”的替代课程思路。仅可作为隔离的 core toy→训练期特权→可见 student 消融；不能把论文的小型交易博弈结果外推为四人厦门麻将保证。 |
| [Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) / [Mortal](https://github.com/Equim-chan/Mortal) | 日麻工程采用 offline AWR、再 online policy-gradient/PPO 的分阶段流程；公开仓库为 AGPL。 | 再次说明“保守 offline→online”是可行编排，不提供厦门规则或可复用权重。当前本项目的 Q/ESS 门槛未满足，不能照搬其训练脚本或混入代码。 |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | 小型完全/不完全信息博弈与 CFR/Deep CFR 的验证平台。 | 可作为粒子过滤、信息集价值和评价器的玩具基准，不作为厦门规则引擎的替代。 |

## 推荐实施路径

### 0. 先把吞吐做成可验证的基础设施

反事实数据的每个“候选动作 × belief world × rollout”应成为独立 job；相同需要网络
推理的 job 聚合为 batch。保留同一候选动作各分支的 common random numbers，以降低动作差的
估计方差。第一版不要求 JAX 化：只要求 CPU 引擎的规则状态独立，Torch 推理可以跨局批量。

验收：批量/串行在相同 seed、动作和对手快照下得到相同终局；零非法动作；可记录每秒
rollout、每秒网络推理和等待时间。吞吐没有达到可稳定增加 belief world 数之前，不扩大网络。

### 1. 构造真正按历史条件化的 belief

现有 `--belief-resample` 已避免把某一真实暗世界泄漏进训练，但只是
`p(hidden | 公开牌面, 本家手牌)` 的先验。下一版 collector 在**内存中**保存以下内容：

- 每一步公开事件和其发生时的行动者；
- 本家在整个过程中的私有摸牌观察；
- 冻结对手策略版本及温度/动作概率定义。

粒子从发牌开始重演；每遇到对手已发生的公开动作，按该对手策略在该粒子私有状态下选择
该动作的概率更新权重；在有效样本量（ESS）过低时重采样。导出数据只保存 actor 可见特征、
合法候选、聚合价值和安全诊断（ESS、权重熵、样本数），绝不保存粒子的暗手、墙、随机种子
或可逆信息。

先在缩小牌墙的 toy profile 上枚举全部隐藏世界，检验粒子后验与精确后验的差异；通过后才
启用到经典规则。没有这一步，不能把“按对手行为加权”称为完整 belief。

### 2. 从平均 Q 升级为保守的分布价值

单个平均值掩盖了“高均值但偶发大放铳”的动作。每个动作至少跨多个 belief world 保留
收益分位数（例如 10/50/90%）或均值、标准差和有效样本量；这些均为不可逆的聚合数值，
可安全导出。网络学习分位数 Q，而非只学均值 Q。

离线策略更新采用有上限的 advantage-weighted regression：

```text
A(s, a) = Q_lower(s, a) - V_visible(s)
w(s, a) = clip(exp(A / beta), w_min, w_max)
loss = weighted_behavior_cloning + lambda * value_losses
```

其中 `Q_lower` 是保守分位数或置信下界。低 ESS、高标准误或低动作价值跨度的样本应降权；
Teacher/DAgger 的交叉熵必须保留，避免策略走出已有对手/规则数据的支持范围。先比较
`policy` 选牌与保守 Q 选牌；当前均值 Q 的 `argmax` 未获独立验证，不能晋升。

### 3. 使用特权信息时只做训练期课程

可把真实 simulation world 中的完整信息提供给一个 oracle critic，训练更低方差的价值或
风险辅助头；visible student 只接收本家手牌和公开信息。训练时逐步增加特权特征的 dropout，
并只以跨 world 聚合的目标蒸馏给 visible student。Suphx 报告直接 oracle distillation 并不
总是有效，因此本项目将它作为消融实验，而不是把 oracle 的最佳动作直接当标签。

硬验收：对部署网络导出、网页调用、离线数据扫描都做“无他家暗牌/无墙顺序”测试；特权路径
不得能被 `TorchPolicyValueAgent` 加载或调用。

### 4. 数据量足够后再提升表征和搜索

当存在至少约 20 万条由上述 belief/Q 管线产生的、按物理牌墙隔离的决策，才进行以下消融：

- 候选排序 MLP 加动作类别、目标牌、多任务风险头；
- 公开事件序列 Transformer，显式编码谁做了什么与时间间隔；
- 以冻结 value/风险网络为叶节点的 1--2 层 belief rollout 或有限深度搜索；
- 通过后，才尝试 league + KL anchor 的小步 PPO。

所有门槛是工程决策，不是从论文得出的普适阈值；数据质量优先于层数和参数量。

## 2026-08-07：belief proposal 连续反证后的路线更新

本项目已经完成并否决三类短前缀 proposal：纯结构 tile factor、按公开弃牌同面额加权、以及用独立线性
Teacher action-energy 转换出的精确 tile factor。最后一类在 selection 墙上提升 total ESS，但全新 terminal 的
structural ESS 只有 15.5%，因此没有 collector 授权。这是有价值的反证：**在当前规则 Teacher 下，试图仅以
初始暗手的按面额独立因子解释后续决策，既不够表达行为，也会制造过高的 p/q 方差。** 不应继续增加 factor、
温度、clip 或同一组墙的调参。

因此 public-belief search 暂时不再是下一项训练前置条件。下一条与既有实验实质不同、但仍有安全边界的研究路线是
**progressive hiding curriculum**：

1. 先建立仅在进程内使用的 full-state oracle observation，和现有 actor observation 做字段级差集审计；任何导出、
   checkpoint actor、网页 payload 必须继续只含可见 observation。
2. 在 core 的缩小 toy 规则上，训练/搜索时逐阶段遮蔽 oracle 字段，并验证最终全遮蔽 student 与从开始全遮蔽的
   student 在同样数据预算下的离线值校准差异；这一步不进行经典实战、也不称为强度证据。
3. 只有 privacy、字段遮蔽、玩具规则精确性和独立 core paired screen 都通过，才考虑 classic 与 league；任何阶段
   的 full-state actor 都不得被网页加载。

这不是重新开启已失败的“privileged critic PPO”：新实验的决策变量是**信息可见性课程与 student 蒸馏误差**，需要
独立数据、独立初始化和新的评测墙。若 toy curriculum 不能改善最终可见 student 的校准或独立 core 筛选，则将其
整体否决，而不是把特权信息留在部署模型中。

## 反面清单

- 不复制日麻系统的源码、权重或未明确授权的牌谱；特别是 AGPL 项目仅作研究。
- 不把单个真实暗牌墙的 rollout 分数用作部署动作标签。
- 不因为训练损失、模仿率或单局胜率而提升模型；只以未见物理牌墙上的四座轮换配对评测决定。
- 不把四人一般和的经验方法描述成纳什/最优性保证。
- 不在没有历史粒子过滤和吞吐基础设施前投入全局 CFR、MuZero 或深度 MCTS。

## 下一次实验的最小规格

在实施新网络前，先做一个最小实验：冻结 `Teacher` 对手池，串行与批量 collector 对同 32
个物理牌墙产生相同的可复放反事实结果；每动作至少 8 个 belief world，报告吞吐、ESS、
均值/方差、分位数，以及物理牌墙切分的 Q 排序。若 Q 排序和配对实战均无改善，停止 AWR/PPO，
优先排查过滤器校准和 Teacher/对手混合，而非继续调温度或网络宽度。
