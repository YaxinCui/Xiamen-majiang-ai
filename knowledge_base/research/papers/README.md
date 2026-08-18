# 论文区

## 阅读与落地索引

- [`method_cards_2026-08.md`](method_cards_2026-08.md)：七篇/组一手资料的方法卡。每张卡片都拆分为论文事实、
  厦门项目的可迁移动作、前置条件与禁止外推的边界。
- [`landscape_2026-08.md`](landscape_2026-08.md)：跨麻将、斗地主、扑克、一般不完全信息博弈、多人训练和离线评测的
  研究地图（35+ 一手来源）；用于检索和选题，不把书目数量误当成已验证能力。
- 当前优先顺序：**Suphx 的辅助目标与部署隔离 → 可审计行为日志 → DR 离线评测 → 对手池/PSRO → 局部信息集搜索**。
  Deep CFR、Single Deep CFR、ReBeL 是中长期架构参照，尚不是完整四人厦门麻将的直接实现方案。

## 使用原则

论文中的“有效”只在其问题设定与假设下成立。厦门麻将当前为四人、一般和、含地方特殊规则的环境；因此本区不会把
两人零和扑克的收敛结论写成厦门麻将的强度结论。所有迁移动作需先写入实验协议，并在冻结 Teacher 的四座轮换评测中验证。

## 1. Suphx：麻将专用深度强化学习

来源： [Suphx: Mastering Mahjong with Deep Reinforcement Learning](https://arxiv.org/abs/2003.13590)（A，访问：2026-08-07）。

论文将麻将视为多人不完全信息、规则复杂且回报稀疏的环境，报告了 global reward prediction、oracle guiding 和
run-time policy adaptation 等组件。对本项目最有价值的不是“直接使用 PPO”，而是三点：

1. 训练目标要拆成可学习的中间信号，不能只依赖终局分数。
2. 训练期可以使用比部署时更完整的 oracle/privileged 信息，但这些特征必须在部署 actor 中隔离。
3. 运行时策略适应必须经过独立评测，不能把训练辅助信息泄露进网页 AI。

厦门边界：Suphx 面向日麻/天凤生态，不能直接移植规则、牌效、得分或特征；只能迁移“辅助目标 + 部署信息隔离”的方法论。

## 2. Deep CFR 与 Single Deep CFR

来源： [Deep CFR](https://arxiv.org/abs/1811.00164)、[Single Deep CFR](https://arxiv.org/abs/1901.07621)（A）。

CFR 的核心是对信息集累计反事实遗憾；Deep CFR 用网络近似大规模信息集，Single Deep CFR 进一步减少平均策略网络的误差来源。
它们适合研究“不完全信息策略”和可利用性，而不是直接替换四人厦门麻将的手牌启发式。

可执行启发：

- 用信息集 key（本家手牌、公开河牌/副露、阶段、规则事件）定义训练样本，绝不能用墙顺序或对手暗牌作为 actor 输入。
- 若采用 CFR，先在缩小的 claim/response 子博弈做可解释实验，再扩展到完整摸打循环。
- 把 exploitability、对固定对手的分差和四座轮换作为不同指标，不能用一个胜率替代全部指标。

## 3. NFSP、DQN 与自博弈

来源： [RLCard 算法文档](https://rlcard.org/algorithms.html)和 [RLCard 论文](https://arxiv.org/abs/1910.04376)（A/B）。

NFSP 的思路是 best-response 学习与平均策略监督学习并行；DQN 则适合离散动作的局部价值学习。对厦门麻将，直接从零
自博弈的问题是终局净分方差高、多人非零和、特殊规则触发稀疏。当前更合理的顺序是：先用 Teacher/DAgger 让 actor 学会
合法且基本合理的行动，再把 NFSP/DQN 用作局部价值或反事实排序分支。

## 4. ReBeL / 搜索与学习的组合

来源： [ReBeL: Self-Play RL in Imperfect-Information Games](https://arxiv.org/abs/2007.13544)（A）。

ReBeL 代表“公共状态 + belief + 子博弈搜索 + 学习 value”的组合路线。它提醒本项目：仅增加一个静态牌效分数不等于
真正建模了隐藏信息；如果要上搜索，应先建立公开信息 belief 和可审计的局部子博弈边界。

## 论文区的暂不采用项

AlphaZero 式纯自博弈、从零 PPO、纯终局 Monte Carlo Q 暂不作为当前主线。原因不是这些方法无效，而是当前厦门数据量、
规则触发稀疏度和评测方差还不足以支撑它们；必须先通过小规模、独立、可复核的中间信号实验。
