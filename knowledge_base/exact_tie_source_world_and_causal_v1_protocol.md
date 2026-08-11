# 完全并列：source-world 配对标签、结构化小模型与因果确认

日期：2026-08-08
状态：三版小模型及高支持二元确认均拒绝；final test 未读；网页 Teacher 不变

## 研究问题与边界

`HeuristicTeacherAgent` 的普通弃牌按 `score` 后再按牌号排序；实测约一半弃牌决策的最高分完全并列。这看似是最安全的
neural residual 入口：网络不必推翻 Teacher 牌形分，只替换任意的牌号 tie-break。但此前公开向听／有效进张 Pareto
消歧在 100 墙为 −0.08，说明牌效代理不能直接作真值。

本轮固定：只处理 classic 普通弃牌最高分完全并列；模型输入只有本家和公开信息；其他状态严格回退 Teacher；只用
hidden-64 小 MLP；按物理墙隔离 train/validation/test；validation 不通过不解析 final test；离线通过后仍须真实随机
干预与新墙整局筛选。

## A：一个自然隐藏世界的配对终局标签

Teacher 自博弈自然到达公开状态后，在私有内存中复制真实模拟状态，把每个最高分并列动作分别强制执行，随后四家恢复
frozen Teacher。导出只含 actor-visible state、并列动作和终局目标，不保存墙、暗手、RNG 或物理 seed。

这属于训练期 privileged supervision：实际隐藏状态是自然访问过程给出的一个 `H ~ P(H|I)` 样本，模型从未接收
`H`。但单样本不是信息集价值，必须靠独立墙泛化证明其可学习。

100 墙 pilot 得到 398 个状态、100 groups、1,311 个终局分支；Teacher 最优 121、替代最优 144、全部相同 133；
数据 SHA-256 `f255b51dc6c7f2e09fb73b6f5e0fa0f3871be2a94a67a94146080bfa04cee957`。

扩大至 900 个新墙后得到 3,550 状态／897 groups／11,501 分支，固定 80/10/10 切分：

| split | 状态 | groups | pairwise examples | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 2,841 | 718 | 7,981 | `ab4f1815ac5f500ccd2af47460cea803f824988659c9cf7701548ffa8afbc18c` |
| validation | 342 | 86 | 1,032 | `95dd51191d1ce3fbdfcbe2169d669dc5de0b7e036249eb7f710780f51235ce05` |
| test | 367 | 93 | 1,302 | `6121839681ad08816a0491954fbc4d63e94c5030fae5f71510c8cf7e782e290a` |

v1 是 feature-v3、五个 hidden-64 MLP；只有五模型一致才偏离，否则回退 Teacher。外层 validation 覆盖 154/342，
paired delta **−1.418**，95% CI **[−3.550,+0.715]**，拒绝。

## B：显式结构 feature-v4

小数据不应迫使 MLP 重新发现规则。feature-v4 在 v3 后增加 59 维：Teacher 结构分量、精确 regular-hand shanten、
34 种牌的完整公开剩余比例，以及等待／活张／目标牌／阶段摘要；总维度 204，hidden-64 单模型仍低于 5 万参数。

用原 train 加已消费 validation 训练，再采全新 100 墙／391 状态／99 groups 验证。五模型一致覆盖 98/391，paired
delta **+0.075**，95% CI **[−2.222,+2.371]**。结构特征消除了 v1 的明显负方向，但仍无正下界。

## C：四个未来牌序平均

为分离“当前暗手不确定性”和“以后摸牌顺序方差”，固定同一个自然到达的对手暗手，只把剩余墙随机重排四次；每个
牌序内所有并列动作共享该牌序，训练前按公开决策平均终局分数。重复牌序不是四个独立 belief world，仍按原物理墙
整体切分。

500 新墙得到 1,972 个公开决策、7,888 行、25,340 个终局分支，固定 70/20/10：

| split | 行 | groups | 平均后 pairwise examples | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 5,436 | 344 | 5,180 | `b4e8eff5162ba01b218da96b11fae954a2f85757fcf8193378b45d23be243cbd` |
| validation | 1,532 | 97 | 1,688 | `f4ad8c1000fb0c59004d361a667cd81a9361f188b869dad75c8d01735c0c4ce4` |
| test | 920 | 59 | 837 | `51ddda4a1136148b5adede81c98c0dd5c9a44309d1a5cb6a8055c9c2fa2cc467` |

同一 feature-v4 ensemble 的内部排序准确率仅 50.6%–52.2%；外层 validation 覆盖 143/383 个公开决策，paired
delta **−0.091**，95% CI **[−1.380,+1.198]**。降低未来牌序方差仍未使公开特征上的动作顺序可学习；test 未解析。

## D：真实随机单点干预

已有 `single_intervention_epsilon_uniform` 数据每局只随机一个弃牌位置，随后恢复 Teacher。在 1,469 个随机弃牌点中，
915 个为最高分完全并列：

- v2 固定 ensemble 偏离 227 次；Horvitz–Thompson **+1.920**，95% CI **[−1.456,+5.295]**；
- 四未来牌序 v3 偏离 295 次；HT **−0.448**，95% CI **[−2.954,+2.058]**。

v2 只有 81 行实际抽中“目标或 Teacher”，所以又预注册高支持确认：每条候选轨迹只在第一次五模型一致分歧处以
0.5/0.5 执行 v2 或 Teacher，之后恢复 Teacher。

全新 `seed=202644000` 起 100 墙／四座轮换得到：

- 229 次合格干预，低于预注册 250；
- candidate/Teacher=113/116，分配 49.34%，结构问题 0；
- HT candidate − Teacher **−5.210**，95% CI **[−11.311,+0.891]**；
- 安全数据 SHA-256 `6a4f9aceafba450a0958c99d7c0964d1fd65c8d417e4c10f250a3cd6160bf53e`。

覆盖未达标且方向明显为负；旧 OPE 正点估计没有复现。状态固定为 `selection_rejected_no_deployment`：不实现重复
override 部署 agent、不跑整局 100/400 墙、不打开 final test。

## 论文联系与适用边界

- [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) 提倡数据不足处回退行为基线；五模型不一致即 Teacher 采用
  同一保守原则，但四人部分可观察麻将不满足其有限 MDP 保证，不能声称理论安全。
- [Localized Interventions](https://proceedings.mlr.press/v238/marmarelis24a.html) 强调只在观测行为附近学习小幅干预；
  完全同分动作是很局部的动作邻域，但本实验说明“局部”不等于“可辨识”。
- [Robust Offline Policy Learning from Multiple Sources](https://proceedings.mlr.press/v258/carranza25a.html) 提醒不同来源应考虑
  最坏混合 regret；source-world、未来墙平均和随机干预方向不一致，不能挑正点估计上线。
- [CANDOR](https://proceedings.mlr.press/v333/mandyam26a.html) 研究结合反事实注释和真实 bandit 反馈。它支持未来的融合方向，
  但当前证据不一致，尚不满足继续融合的前提。

## 固定结论

1. Teacher 完全并列不是“免费改进区”；公开信息不一定足以辨别长期最优动作。
2. source-world 标签可用于训练期 privileged supervision，但当前规模不可泛化；四未来牌序也没有修复。
3. feature-v4 的规则分量可留给未来不同标签源，但自身不是改进信号。
4. exact-tie 自动神经修正族冻结；不再增加同族墙数、hidden size、ensemble 阈值或未来牌序数。
5. 当前可接受的新信号仍是匿名人工二选一、强真人牌谱，或与本族不同且通过高支持因果确认的局部错误类别。

主要实现：`xiamen_mahjong/exact_tie_rollout.py`、`scripts/collect_exact_tie_rollout_v1.py`、
`scripts/train_exact_tie_rollout_ranker_v1.py`、`scripts/train_exact_tie_rollout_ranker_v2.py`、
`scripts/train_exact_tie_future_average_ranker_v3.py`、`scripts/audit_exact_tie_randomized_ope.py`、
`xiamen_mahjong/exact_tie_intervention.py`、`scripts/select_targeted_exact_tie_ensemble_v1.py`。
