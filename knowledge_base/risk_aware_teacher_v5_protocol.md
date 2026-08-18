# 公开信息防守 Teacher v5：预注册协议

日期：2026-08-07。v3/v4 的 direct-score relative-advantage 路线在两个独立 selection 墙组均被拒绝，不能以
更多 clipping、threshold、beta、temperature 或同类 neural residual 重试。v5 改为一个可解释的、**不依赖
反事实标签**的公开信息 rule candidate，并直接用规则引擎的四座轮换对局评测。

## 固定候选族

候选继承 `AvailabilityTeacherAgent` 的自家牌形与公开余牌听口评分，在每一项合法弃牌上额外减去
`risk_weight × public_danger(tile)`。`public_danger` 只读取：本家手牌、四家河、副露、翻金，以及当前公开
轮次；对每家对手，未公开副本越多、对手副露越多、牌局越晚，危险贡献越大；该牌若曾被该对手弃过则按固定
0.2 折扣。算法不得读取牌墙、任何对手暗手、随机种子或未来响应。

唯一可变参数为 `risk_weight∈{0,1,2,4,8}`；副露/进程系数、折扣与基础 wait-copy value 4.5 固定在源码。
response、胡、杠和游金决策全部继续使用 frozen `HeuristicTeacherAgent` 的规则。`risk_weight=0` 是公开余牌
Teacher 对照，不是网页默认 Teacher。

## 独立选择和终检

- selection：classic 物理墙 seed `202611000` 起 160 个，每墙候选四座轮换、其余三座 frozen Teacher。
- 仅当候选相对 Teacher 的按墙 paired score delta 95% 下界 >0，才可参与选择；多个通过则下界最大优先，平局选更小
  `risk_weight`。若无通过者，terminal 不运行。
- terminal：唯一 winner 在不同的 classic seed `202611500` 起 400 个物理墙、同样四座轮换和 frozen Teacher；同样
  需 paired 95% 下界 >0。通过后也只允许一项全新 1,000 墙筛选，不自动接入网页或声称超过人类。

无论单局胜率、训练/selection 点估计或启发式解释多好，都不允许改变该网格、规则系数、种子范围或在 terminal 上
选择参数。所有报告只存聚合与配对统计；候选不得把隐藏状态导出为训练数据或网页输入。
