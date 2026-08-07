# Teacher 单点弃牌干预 v1：预注册数据与筛选协议

日期：2026-08-07。目标是得到**一个摸后/弃牌阶段动作替换、其余全程为
`HeuristicTeacherAgent`**的可识别因果估计。该协议不训练完整策略，不能直接
用于网页或人类强度主张。

## 冻结的数据生成

- 规则：`classic`；物理牌墙 seed 为 `202608700` 起的 1,600 个连续整数；每墙四座
  轮换，因而共有 6,400 局候选席记录。
- 行为基线与三名对手：网页默认 `HeuristicTeacherAgent`。
- 每局候选席仅在 `discard` phase 计数；从前 8 个这类决策中均匀选一个位置。仅该位置
  执行 `0.4 × Uniform(legal) + 0.6 × Teacher`，其他决策及该位置之后的后缀仍严格执行
  Teacher。每条随机动作须记录其 exact `executed_probability`。
- 按完整物理牌墙用 salt `teacher-discard-v1-main` 分为 train/validation/held-out =
  60%/15%/25%。不得跨座位、跨 partition 复用同一墙。held-out 再按新 salt
  `teacher-discard-v1-heldout` 均分 selection/terminal；outcome 训练和 epoch 选择不得读取
  terminal。

## 数据与校准闸门

只有当 train、validation、selection、terminal 的随机 `discard` 干预数分别至少为
2,500、500、500、500，才进入下一步。若不足，记录覆盖失败并重新设计**新的**收集协议，
不在此墙组事后改 `intervention-max-decisions`。

使用同一个新鲜、固定 policy anchor（feature v3、hidden=128）训练 5 个独立 seed 的
afterstate outcome 成员；所有成员仅更新独立 afterstate encoder 与 score/own-win/opponent-win
heads，只接收各自 partition 中 `executed_probability < 1` 的 `discard` 动作，且训练均传
`--skip-test`。每个成员的 validation score MAE 必须严格低于 validation 的零分预测 MAE；
任何成员失败则不产生 ensemble selector、不读取 selection/terminal 的终局标签。

## 选择与终检

仅在 selection 上比较预先固定的 `minimum_lcb_advantage ∈ {0, 8, 16, 24}` 分，score
LCB 为 5 成员均值减 1 倍成员总体标准差。候选为“LCB 最高动作；若未比 Teacher 弃牌
高过阈值则仍用 Teacher”。每一候选由墙组聚合 IPS 与 DR 估计；两项 95% 下界必须都
大于 0，且 target/base ESS 均至少 50。通过者取最小下界最大者，同分取更高阈值。

selection 无通过者时程序必须保持 `terminal_read=false`。若有唯一通过候选，才在 terminal
上以固定阈值读取一次并重复同一严格门槛。即使 terminal 通过，也只授权一个后续的 200
物理墙、每局至多一次弃牌 override + Teacher 后缀的实战筛选；它仍不是完整策略、更不是
人类强度证据。

## 已完成的 smoke（不属于 v1 数据）

独立 core seed `202608601` 的 48 墙/192 局 smoke 证明 `discard` phase 的行为 metadata、
propensity 和 phase-isolated trainer 可连通：train/validation/test 随机干预为 102/22/57。
一名 16 隐层 outcome 头的 validation score MAE 为 33.18 分，而零预测为 33.09 分，未过
校准门槛；未训练第二成员、未做 selector 或 OPE。该 smoke 的 test 仅在预检计数时被读取，
因此永不作为 v1 的 selection 或 terminal 数据。
