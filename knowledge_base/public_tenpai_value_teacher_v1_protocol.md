# 公开听牌价值 Teacher v1：预注册协议

日期：2026-08-08
状态：100 墙 selection 未通过；400 墙 terminal 未读取；不得接入网页或训练标签

## 假设来源

对安全轨迹 v4 中前 1,000 副 natural Teacher self-play 做聚合审计，共见到 33,043 次普通非游金弃牌。
冻结 `HeuristicTeacherAgent` 已经选择直接听牌时，有 946 次存在 Teacher 分差不超过 2 的其他听牌弃牌；其中
229 次（近分听牌局面的 24.2%，全部普通弃牌的 0.693%）存在更高的
`公开剩余张数 × 可见立即自摸结算`。224/229 次主要差异来自更多公开剩余张数。

这只是结构性盲区，不是动作价值真值：公开剩余牌可能仍在对手暗手，立即自摸结算也没有覆盖后续攻防与整局收益。
因此只把它实现为窄门控候选，并要求全新牌墙上的实战证据。

## 唯一候选

`PublicTenpaiValueTeacherAgent(score_margin=2.0, minimum_weighted_value_advantage=1.0)`。

它完整继承 frozen Teacher，只在以下条件同时满足时改变普通弃牌：

1. 当前无任何游金状态，本家也不处于金牌锁；
2. frozen Teacher 的弃牌已经直接听牌；
3. 另一个直接听牌弃牌与 frozen 分差不超过 2；
4. 使用本家手牌、公开牌河、公开副露和金牌指示计算时，候选的
   `sum(公开剩余张数 × 普通自摸可见结算)` 至少严格增加 1。

对手暗手、牌墙、未来随机数和其他玩家暗杠牌面一律不读取。没有阈值网格，也不允许在 selection 结果出来后修改
上述参数重跑同一牌墙。

## 强度筛选

- 规则：classic。
- selection：从 `seed=202616000` 起连续 100 副全新物理牌墙；每墙候选四座轮换，其他三座固定为
  `HeuristicTeacherAgent`，共 400 局。
- 同墙 reference：候选座也使用 `HeuristicTeacherAgent`，其余配置完全相同。
- 独立统计单位是物理牌墙：先平均同一墙的四个座位，再计算 paired score delta 及 95% CI。
- selection 唯一门槛：`paired_seed_score_delta_95pct_low > 0`。
- 未通过立即淘汰；从 `seed=202616200` 起连续 400 墙的 terminal 保持未读。
- 只有 selection 通过才读取 terminal；terminal 使用相同门槛，且结果仍只进入人工晋级复核，不自动替换网页 Teacher。

100 墙按项目当前约定只是低成本筛选，不能把正点估计或胜率单独解释成“更强”。

## 结果

selection 已按预注册配置运行。候选在 400 个四座轮换对局中 97 胜，均分 **−0.7275**；相对同墙 frozen
Teacher 的 paired score delta 95% CI 为 **[−1.8200, +0.3650]**。点估计为负且下界不大于 0，故状态为
`selection_rejected_terminal_unread`。`seed=202616200` 起的 400 墙 terminal 没有读取。

结论不是“公开余牌没有信息”，而是当前固定的“余牌数 × 立即自摸可见分”不足以代表整局动作价值。该规则候选
保留作失败对照，不接入网页、不生成训练标签，也不在同一 selection 墙上调整 margin 或优势阈值。
