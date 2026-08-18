# 低分差 top-2 可解释因果错误地图 v1

状态：**45 个预冻结类别均未通过稳定性与全族下界；无确认候选，不读取 terminal，低分差 top-2 数据路线完全冻结。**

## 目的

低分差 residual v1/v2 已证明整体神经 gate 没有正下界，但两轮高支持随机试验留下 8,000 个物理墙。与其继续扩大同一个
网络，本轮只问一个新问题：Teacher top-1/top-2 的差异是否集中在一个 actor-visible、可解释且跨历史分区方向一致的
错误类别。若不存在，这批弃牌随机试验到此完全冻结；若存在，只允许把唯一类别带到全新墙确认。

## 可读取与不可读取数据

只读取：

- v1 train：4,000 墙；
- v1 已消耗 validation：2,000 墙；
- v2 已消耗 validation：2,000 墙。

三个分区分别报告，不把旧 validation 重新称为 validation。禁止读取 v1 terminal `202656000..202657999`、v2 terminal
`202660000..202661999`、人工标签或真实墙／其他玩家暗手。若未来产生候选，确认必须使用上述范围之外的全新 seed。

## 冻结类别词典

所有谓词只使用安全记录中的 actor hand、公开计数、阶段、庄家、摸牌和两张动作：

- 阶段：`early(wall>=54)`、`middle(36..53)`、`late(<36)`；
- Teacher margin：`exact_tie`、`positive_margin`；
- 庄家：dealer／nondealer；
- 动作牌类转换：suit→suit、suit→honor、honor→suit、honor→honor；
- 手牌面数：alternative 的同牌张数低于／等于／高于 Teacher；
- 局部连接：同花色 ±1/±2 邻牌的手中张数，alternative 低于／等于／高于 Teacher；
- 摸切关系：alternative 摸切、Teacher 摸切、两者都不是摸切；
- 公开剩余同面张数：alternative 高于／等于／低于 Teacher；
- 数牌端张：alternative 端张而 Teacher 非端张、反向、两者均端张、两者均非端张；
- 同花色牌号：alternative 更低／更高；以及不同花色。

另固定 `exact_tie × {阶段、牌类转换、面数、连接、摸切}` 的交叉项。不得在结果后新增向听、有效进张、风险、牌号阈值或
三重交叉项。

## 冻结估计与选择规则

每个类别定义一个策略：仅当谓词为真时把 top-1 换成 top-2，其余保持 Teacher。使用已知 0.5 propensity、四座墙聚合。
同墙另外三局均分 control 固定使用 `c=-2.5`；它在本轮不按类别或分区重新拟合。每个分区至少 60 次 override、60 个墙组，
日志 alternative/Teacher 各至少 20。

类别只有同时满足以下条件才可成为唯一确认候选：

1. 三个历史分区的 control-adjusted 点估计都严格为正；
2. 三个分区机械支持都通过；
3. 合并 8,000 墙后，对整个冻结类别词典做双侧 family alpha=0.05 Bonferroni，其下界严格为正。

多个类别通过时，选择三个分区中最小 `mean/SE` 最大者；再以 pooled 校正下界、类别名作确定性破同分。全部失败则不产生
候选、不收集新墙。即使通过，也只解锁一个另行预注册的随机二元确认，不直接修改 Teacher、训练网络或部署网页。

## 已执行结果

三个历史分区共 8,000 个物理墙、32,000 条四座记录，group 两两不重合。45 个冻结类别中，36 个在三个分区都达到
机械支持，22 个在三个分区的 control-adjusted 点估计均为正；但没有一个类别的 pooled family-corrected 下界严格为正。

按预注册“最小分区 z”排序最接近的类别为：

| 类别 | 三分区点估计 | pooled 均值 | family-corrected 下界 |
| --- | --- | ---: | ---: |
| alternative 与 Teacher 公开剩余同面张数相同 | +0.766 / +2.274 / +0.970 | +1.194 | -0.393 |
| 两动作都是数牌但花色不同 | +0.458 / +1.357 / +1.313 | +0.896 | -0.179 |
| 两动作均为非端张数牌 | +0.335 / +1.067 / +1.043 | +0.695 | -0.105 |

这些方向虽跨分区同号，但最弱分区都只有约 1 个标准误，且已经从 45 个类别中筛选，不能当作确认候选。状态固定为
`no_stable_interpretable_category_top2_family_frozen`：不新增交叉项、不选择上述任一类别、不采集新墙、不训练规则 gate。
报告 SHA-256 为 `736e58abaf534a214a7e55a8da64cf973be01be4a598521cafa86f0441d7b1d2`。
