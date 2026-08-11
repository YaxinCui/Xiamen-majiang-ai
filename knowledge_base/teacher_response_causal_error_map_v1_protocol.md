# Teacher 响应因果错误地图 v1 协议

日期：2026-08-08
状态：validation 未通过；不收集新干预、不部署

## 目的与边界

已有三类响应替代都没有通过独立筛选：吃碰后强制弃牌形分、精确向听词典序，以及 afterstate outcome-LCB。
本轮不训练更大的网络，也不把局部牌效再次当作终局价值。它只使用已有的
`teacher-response-intervention-classic-v2` 单点随机干预，建立 Teacher 响应的因果错误地图：行为策略在每局前四个
response 位置中预选一个，以 `0.4 × uniform + 0.6 × Teacher` 执行动作，随后立即恢复冻结 Teacher。

地图只报告物理墙分组的 aggregate IPS，不导出手牌、动作实例、赛果或牌墙。估计对象始终是“一处响应替换后恢复
Teacher”，不是一套会在整局多次改动作的策略。

## train-only 错误类别

互斥类别预先固定为：

- `chi_{early,middle,late}`；
- `pong_{early,middle,late}_{suit,honor}`。

公开余墙 `>=54` 为 early，`36..53` 为 middle，`<36` 为 late。每类只估计 `pass − Teacher claim`。Teacher 的
`hu`、`ming_kan` 和 `pass` 不提出替代。该地图为多重探索，区间不作强度结论；只允许读取 v2 的 train 分区。

train 探索显示 `pong→pass` 汇总为负，尤其荣牌碰不应削弱；`chi→pass` 汇总点估计为正但区间很宽。因此冻结唯一
筛选候选：**只在随机选中的响应位置把 Teacher 的 `chi` 改成 `pass`，其他位置与后缀全部保持 Teacher**。候选没有
stage、向听、金牌数、阈值或网络参数，结果出来后不得添加子类重跑同一 validation。

## 固定 validation 门槛

- 数据：仅 `teacher-response-intervention-classic-v2/validation.trajectories.jsonl`；不读取旧 selection 或仍封存的
  terminal。
- 同时报告全部随机响应位置的目标策略 IPS，以及仅 Teacher-chi 支持行的 conditional IPS。
- 至少 100 个 chi override 观察、75 个物理墙组；conditional target/base ESS 都至少 30。
- 全部位置 IPS 与 conditional IPS 的 95% 下界都必须严格大于 0。

全部条件通过也只允许在全新物理墙上收集“首个合格 chi 的 0.5/0.5 二元随机干预”；它不允许网页部署、训练 residual、
读取旧 terminal 或声称强于人类。任一条件失败则候选冻结，不追加 stage/向听/牌类门控，不收集新数据。

## 已执行结果

train 错误地图读取 2,332 次随机 response／939 个物理墙组。`pong→pass` 的汇总点估计为 −8.913，95% CI
`[-16.091,-1.735]`；其中荣牌碰为 −13.614，区间 `[-25.309,-1.920]`，说明“统一减少碰牌”有直接反证。
`chi→pass` 汇总为 +2.450，但区间 `[-5.017,+9.917]`，只能提出筛选假设。train 文件 SHA-256 为
`e6d5117523dcaec29c45bd72bb7afd7b2827a7ea93e9b8d21f7a8a12cc7e84f5`。

冻结候选随后只读 v2 validation：583 次随机 response／239 个输入墙组，其中 338 次 Teacher-chi override、194 个
有效 conditional 墙组。conditional target/base ESS 为 64.69/262.49，覆盖门槛均通过；但 `pass−chi` IPS 为
**+7.406**，95% CI **[−6.905,+21.718]**。计入未改响应位置的策略估计为 **+5.511**，95% CI
**[−4.519,+15.540]**。两个下界均未严格为正，状态
`validation_rejected_no_targeted_collection`。validation 文件 SHA-256 为
`e77deb8d428f8e4d912d481ea070182fcfb3cc8b0589b4aab2a861b96d9eaa5a`。

因此不收集新的首-chi 二元干预，不读取 response-v2 terminal，不把“少吃牌”写进 Teacher，也不训练该 gate。这个结果
不是证明吃牌必然更好，而是说明当前已有随机样本不足以把正点估计与高方差区分开；按协议必须换数据／训练路线，而不是
在 validation 上追加巡目、向听或金牌子类。
