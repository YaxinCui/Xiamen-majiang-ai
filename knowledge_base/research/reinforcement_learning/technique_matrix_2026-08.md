# 棋牌 AI 技术选择矩阵（2026-08）

此矩阵将大量论文收敛成可执行决策。`状态`针对当前厦门项目，不评价论文质量；所有部署策略均不得读取墙牌、对手暗手或未来信息。

| 方法族 | 代表资料 | 解决的问题 | 当前状态 | 最小前置/否决条件 |
| --- | --- | --- | --- | --- |
| 行为克隆（BC） | [Kanachan](https://github.com/Cryolite/kanachan) | 先复制稳定 Teacher 行为。 | 现在可验证 | 按物理墙切分；低 loss 但实战未提升即不晋级。 |
| DAgger/反事实标注 | [Suphx](https://arxiv.org/abs/2003.13590) 的训练启发 | 覆盖候选实际会访问的状态。 | 现在可验证 | Teacher/搜索标签的可见字段和版本必须存档。 |
| Teacher-anchored residual | [Suphx](https://arxiv.org/abs/2003.13590) | 使候选从可解释基线保守偏移。 | 现在可验证 | 零残差严格等价 Teacher；每次偏移可解释、可回放。 |
| 多任务辅助头 | [Suphx](https://arxiv.org/abs/2003.13590) | 把稀疏终局回报拆成牌效/风险/规则事件标签。 | 现在可验证 | 每头有 held-out 指标；辅助升而实战降则删/降权。 |
| PTIE/特权蒸馏 | [PerfectDou](https://arxiv.org/abs/2203.16406) | 训练期以完美信息帮助不完全信息 actor。 | 准备后验证 | 训练/部署特征表、泄露单测、模型导出审计齐全。 |
| 深度 Monte Carlo + 动作编码 | [DouZero](https://arxiv.org/abs/2106.06135) | 大而变化的合法动作集。 | 准备后验证 | 厦门动作候选稳定编码；摸打/响应/特殊规则分头评测。 |
| 局部 afterstate value | [DeepStack](https://arxiv.org/abs/1701.01724) | 将长局价值压缩到短边界。 | 准备后验证 | afterstate 定义、终止/收益、同墙对照实验；不可直连真墙。 |
| public belief + 局部搜索 | [ReBeL](https://arxiv.org/abs/2007.13544) | 将公开历史转为未知牌的概率信念。 | 准备后验证 | 未知牌池重建、归一化、零概率、搜索可见性单测。 |
| 对手池 / PSRO-lite | [PSRO](https://arxiv.org/abs/2403.02227) | 防止只击败单一 Teacher。 | 现在可设计 | 冻结版本、四人阵容组合、矩阵协议和最差表现均要报告。 |
| 对手建模 | [多人对手建模](https://arxiv.org/abs/2212.06027) | 利用公开历史区分对手风格。 | 准备后验证 | 仅用公开历史；先验证跨会话稳健性，不能引入身份泄漏。 |
| PBT | [PBT](https://arxiv.org/abs/1711.09846) | 在线调 loss/学习率等时间表。 | 后期 | 至少已有可靠 100 墙筛选和独立确认，避免选择噪声。 |
| IMPALA/V-trace | [IMPALA](https://arxiv.org/abs/1802.01561) | 高吞吐 actor-learner 与滞后校正。 | 后期 | 单机数据管道/复现实验已稳定，吞吐确为瓶颈。 |
| 离线 DR OPE | [DR OPE](https://proceedings.mlr.press/v48/jiang16.html) | 用行为日志快速诊断候选价值。 | 准备后验证 | 实际 propensity、支持集、行为版本和奖励定义缺一不可。 |
| FQE | [FQE](https://proceedings.mlr.press/v48/thomasa16.html) | 训练目标策略的离线 value 模型。 | 后期 | 与 DR/实战排序对照；数据外动作多则否决。 |
| CQL/IQL | [CQL](https://arxiv.org/abs/2006.04779)、[IQL](https://arxiv.org/abs/2110.06169) | 控制离线学习的分布外误差。 | 后期 | 大量多行为策略数据、稳定动作表示和保守基线齐备。 |
| AIVAT/配对降方差 | [AIVAT](https://arxiv.org/abs/1612.06915) | 降低随机牌墙造成的对局估计方差。 | 准备后验证 | 已知策略概率或合格 value 控制变量；不替代实战金标准。 |
| CFR/MCCFR/Deep CFR | [Deep CFR](https://arxiv.org/abs/1811.00164) | 在信息集上做遗憾最小化。 | 仅局部研究 | 先在明确定义的两方 response 子博弈验证；不可宣称四人全局收敛。 |
| Student of Games | [Student of Games](https://arxiv.org/abs/2112.03178) | 搜索、学习与博弈论推理的统一。 | 长期参照 | 需要成熟 belief、value、局部搜索和大量计算。 |
| R-NaD/DeepNash | [DeepNash](https://arxiv.org/abs/2206.15378) | 两人零和隐信息自博弈的稳定学习。 | 长期参照 | 厦门为四人一般和，不能直接照搬优化/收敛叙述。 |
| MADDPG/QMIX/MAPPO | [MADDPG](https://arxiv.org/abs/1706.02275)、[QMIX](https://arxiv.org/abs/1803.11485) | 协作型 CTDE 和价值分解。 | 不作为主线 | 四人麻将无统一团队 reward；中央 critic 的特权信息与部署隔离风险高。 |
| AlphaZero/MuZero | [AlphaZero](https://arxiv.org/abs/1712.01815)、[MuZero](https://arxiv.org/abs/1911.08265) | 搜索引导 self-play。 | 不作为主线 | 完美信息偏置；在无 belief 与局部收益校准前禁止套用。 |

## 固定推进顺序

```text
规则正确 + 可见性审计
  → Teacher/搜索标签质量与覆盖
  → BC + 一项辅助头 + residual
  → 100 墙四座筛选与失败归档
  → opponent league / 真实 propensity 日志
  → DR/AIVAT 诊断与局部 afterstate
  → public belief + 有限深度搜索
```

其中任意一步若出现非法动作、隐藏信息泄露、固定墙不可复现，立即回退到规则/数据层，而不是增加网络层数或训练轮数。
