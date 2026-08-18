# 32-world 局部 paired advantage 标签试验 v1

日期：2026-08-08
状态：聚合标签门槛未通过；不训练、不导出逐状态标签

## 目的

Teacher 模仿 MLP 的高 logit gap 已在独立 100 墙中失败，因此 logit 不能再充当 advantage。过去的信息集 rollout
和 action-Q 数据通常只有每状态 1–3 个 belief worlds，方差太大。本试验不部署搜索，也不立即生成训练集；它只问：
对已经失败的 2% gate 状态，使用 **32 个共享 private-world 样本**后，能否把一部分动作稳定分成“替代项更好”和
“Teacher 更好”，从而证明局部 advantage 标签值得继续采集。

## 固定协议

- classic；`seed=202620000` 起连续 50 副新物理墙，四座依次作为观察者。
- 基础对局与所有 continuation 都使用 frozen `HeuristicTeacherAgent`。
- 状态筛选完全复用失败候选的 checkpoint SHA、普通弃牌范围和固定 margin
  `2.4152190685272217`；不重新挑状态公式。
- 最多处理 40 个 override 状态；每状态从行动者公开信息集独立重采样 32 个 private worlds。
- 同一个 world 内分别强制模型替代弃牌和 Teacher 弃牌，其后使用相同 frozen continuation；标签统计单位是
  `score(alternative) - score(Teacher)` 的 paired difference。
- 每状态报告均值、标准误和 95% 区间，但产物只保存聚合计数，不保存手牌、历史、动作 trace、任何重采样 world、
  牌墙、对手暗手或 RNG 状态。

## 继续采集门槛

只有以下条件同时满足，下一轮才允许导出 actor-visible 状态和 paired 标签供 linear/GBDT/小 MLP 比较：

1. 至少 15 个可用状态；
2. belief world 可用率至少 90%；
3. 至少 5 个状态的替代动作 95% 下界严格大于 0；
4. 至少 5 个状态的替代动作 95% 上界严格小于 0。

正负两类同时存在是训练 gate/discriminator 的最低可识别条件；若大部分状态仍不确定，说明 32 worlds 仍不足或当前
公开 belief 不校准，本路线停止，不能用 noisy argmax 生成模型标签。

## 结果

50 副物理墙共扫描 1,388 次候选座普通弃牌，命中 35 个固定 run3 gate override 状态。每状态 32 个 world，
1,120/1,120 个 world 全部可用，没有合法集合不一致或重采样失败。在逐状态 paired 95% 区间下：

- 替代项下界严格大于 0：1 个；
- 替代项上界严格小于 0：4 个；
- 区间跨 0：30 个。

跨 35 个状态均值为 **−1.9143**，标准误 2.5612，95% CI **[−6.9343, +3.1058]**。正负显著状态都未达到
各 5 个的预注册门槛，故状态为 `paired_label_pilot_not_ready`。不导出安全状态、不训练 gate/discriminator，
也不通过增加 epoch、改变置信系数或挑选单个正例挽救此 run3 标签源。

该结果把瓶颈进一步定位到**优势来源**而非模型容量：32-world paired 计算已经完全可运行，但 run3 高 logit gap
仍不能稳定提供正优势。后续只能换独立候选来源（优先检查曾保持正点估计的精确一摸候选）或收集经授权真人纠错。
