# 精确一摸 public-fix v2：32-world paired 标签试验

日期：2026-08-08
状态：已完成，标签门槛未通过；不训练、不导出逐状态标签

## 候选来源与 v1 边界修正

`ExactOneDrawTenpaiTieBreakTeacherAgent` 是目前唯一在两组实战中都保持正点估计的规则候选：旧 v1 在 160 墙为
+2.1203，400 墙复核为 +0.5119，但两个 95% 区间都跨 0，不能晋级。复核代码时发现 v1 的余牌计数会读取他家
暗杠牌面，而 v4 公开契约明确隐藏该牌面。v2 已改为只知道“他家完成暗杠”，不计其具体 face，并加入暗杠牌面
替换不变性测试。

因此旧 160/400 墙只作为选择这个候选来源的方向性先验，不能证明 public-fix v2 更强。v2 必须使用全新数据重新
建立证据。

## 固定标签协议

- classic；`seed=202622000` 起连续 50 副全新物理墙，四座依次作为观察者。
- 基础对局和所有后续行动均为 frozen `HeuristicTeacherAgent`。
- v2 参数固定 `score_margin=2.0`、`minimum_live_advantage=1`；只处理其与 Teacher 不同的普通弃牌。
- 最多 40 个 override 状态，每状态从行动者公开信息集采样 32 个 private worlds。
- 在同一 world 上分别强制 v2 弃牌和 Teacher 弃牌，之后都由 frozen Teacher 完成；统计
  `score(v2)-score(Teacher)` 的 paired mean、stderr 与 95% CI。
- 聚合产物不保存状态、手牌、历史、动作 trace、private world、牌墙、对手暗手或 RNG。

## 数据继续门槛

只有以下四项同时满足，才允许下一轮导出 actor-visible 状态与分组 paired 标签：

1. 至少 15 个可用状态；
2. world 可用率至少 90%；
3. 至少 5 个状态的 v2 95% 下界严格大于 0；
4. 至少 5 个状态的 v2 95% 上界严格小于 0。

若通过，下一轮按物理墙切分 train/validation/test，只比较 linear、GBDT 和小 MLP gate；若未通过，不增加训练
epoch、不从不确定状态取 argmax，也不把旧 v1 实战点估计当标签。

## 执行结果：标签源拒绝

固定协议已在 `seed=202622000` 起的 50 副物理墙上完成。共扫描 705 个候选普通弃牌决策，收集 40 个
v2/Teacher 分歧状态；每状态 32 个共享 belief worlds，1,280/1,280 个 world 可用（100%）。40 个状态中只有
**2** 个 v2 动作的 95% 下界严格为正、**2** 个 95% 上界严格为负，其余 **36** 个跨零。跨状态 paired mean
为 **−0.9000 ± 1.7625**，95% CI 为 **[−4.3546, +2.5546]**。

正负可辨识状态均少于预注册的 5 个，因此状态为 `paired_label_pilot_not_ready`。没有导出逐状态训练行，
没有训练 linear、GBDT 或小 MLP，也不把 40 个状态中的 noisy argmax 当成监督。继续把 world 数从 32 增至
128 只会更精确地估计当前未按完整历史校准的近似 belief，不能修复分布本身；该候选来源在得到新的历史条件化
后验之前停止。
