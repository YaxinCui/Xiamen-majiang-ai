# 精确一摸听牌平分裁决 Teacher v1：预注册协议

日期：2026-08-07。此前全局 `OnePlyLookaheadTeacherAgent` 已在独立经典档筛选中显著落后默认 Teacher，
因此本实验**不**重试全局一步前瞻，也不改变默认的牌形主评分。

只读结构诊断（classic、40 墙、seed `202614100` 起）覆盖 1,245 次 Teacher 弃牌；其中 1,060 次为
“原 Teacher 评分相差不超过 2 且当前没有直接听牌”的可比较决策。仅 47/1,060（4.43%）存在更多的纯结构
一摸听牌路线。这个诊断不是强度评测、没有导出逐局状态，也没有授权策略替换；它只证明一个严格限域候选值得
被独立证伪。

## 唯一固定候选

`ExactOneDrawTenpaiTieBreakTeacherAgent` 首先完整保留 frozen `HeuristicTeacherAgent` 的弃牌排序。只有同时满足：

1. frozen 第一选择及比较对象都不是直接听牌；
2. 两者 frozen `hand_quality + 18 × 等待牌种数 − 打金惩罚` 分差不超过 `2.0`；
3. 枚举下一张可摸牌、按经典档跟打规则枚举下一次合法弃牌后，替代选择的公开“可摸张数 × 后续活听张数”总和
   至少严格多 `1`；

才把该替代选择排到第一。公开可见张数只包括本家手牌、四家河牌和副露、翻金；后续活听会扣除该假设摸入的
一张。同一分值仍保持 frozen Teacher 分数、牌面顺序的确定性并列规则。候选不读取牌墙、对手暗手、未来随机数，
也不改变胡、杠、游金、吃碰或跟打以外的任何规则。

`score_margin=2.0` 和 `minimum_live_advantage=1` 是本协议唯一参数，没有网格调参。它不能直接接入网页、
Teacher 数据采集或神经网络标签。

## 独立选择和终检

- selection：classic 物理墙 seed `202614500` 起连续 160 个；每墙候选四座轮换，其余三座 frozen
  `HeuristicTeacherAgent`。
- 若候选相对同墙 Teacher 的 paired score delta 95% 下界不严格大于 0，终检墙完全不读取。
- terminal：只有 selection 通过，才在互不重叠的 seed `202614800` 起连续 400 个物理墙、同一四座换位协议下
  终检；同样要求下界严格大于 0。
- 即使通过 terminal，也只能解锁额外 1,000 墙和异质对手阵容筛选；不能自动替换网页默认 Teacher，不能产生
  “超过人类”结论。

选择脚本只写聚合对局和配对统计。慢速网页与四人手动模式是规则观察工具，不参与本候选的数据构造或选模。

## 执行结果：selection 拒绝，terminal 未读

selection 已在固定 classic `seed=202614500` 起 160 个物理墙完成（640 个四座换位对局）。候选均分为
**+2.1203 ± 1.1525** 分/局，175/640 胜；但相对同墙 frozen Teacher 的 paired score delta 95% CI 为
**[−0.1386, +4.3793]**。下界不严格大于 0，故状态为 `selection_rejected_terminal_unread`：`seed=202614800`
起的 400 墙终检未读取，候选不进入网页、训练标签或默认 Teacher。selection 墙已消耗，禁止改变本协议参数后重跑。
