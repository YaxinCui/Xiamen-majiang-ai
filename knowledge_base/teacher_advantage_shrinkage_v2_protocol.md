# Teacher 相对优势 shrinkage v2：预注册协议

日期：2026-08-07。v1 的静态 absolute-score LCB 已在独立 terminal 失败；同一数据上的
未缩减 DR 相对优势又有极端重要性残差，因而禁止把 v1 的 selection/terminal 用于任何
shrinkage 调参。本协议定义一个新的、明确**有偏但有限影响**的训练标签候选；最终策略许可仍
只能来自未缩减 IPS/DR 的新墙组评估。

## 冻结的数据契约

- classic 规则，连续 2,000 个全新物理墙 seed `202608900` 起，四座轮换。
- 四家基线均为 `HeuristicTeacherAgent`。每局只在前 8 个 `discard` phase 决策中随机取一处；
  该点按 `0.8 × Uniform(legal) + 0.2 × Teacher` 执行，随后无条件恢复 Teacher。
- `0.8` 是预注册的 propensitiy 提升，不是根据 v1 terminal 结果调得；它使非 Teacher 动作的
  最低行为概率约为旧 `epsilon=0.4` 的两倍。每条记录保存 exact executed propensity。
- 按完整物理墙分 train/validation/held-out = 60%/15%/25%，held-out 再按墙组均分 selection/
  terminal。所有 raw JSONL 本地保留；Git 只提交无私有特征的 manifest、覆盖、参数、报告。

覆盖门槛为随机弃牌干预 train ≥3,500、validation ≥800、selection ≥700、terminal ≥700。
任何分区不足即整体停止，不更改本墙组的 epsilon 或 intervention index 上限。

## Cross-fitting 与有界伪优势

在 train 墙组上按完整墙 K=3 折训练 direct outcome model。每个动作的 direct prediction必须
来自未接触它所属折的模型。相对 Teacher 的 action label 取 direct score gap 加两个 DR residual，
但每个 ``residual / propensity`` 分量在点数单位 winsorize 到预注册
`C ∈ {20, 40, 80}`。这不是无偏估计；只是一组待检验的有限影响训练标签。

每个 C 训练独立的、**不复用旧 synthetic action-Q 语义**的 relative-advantage learner；只可看
actor-visible 状态。validation 仅报告 direct calibration、raw/trimmed residual 分位数、Teacher action
恒等于零的检查，以及候选预测的 gap/override rate；不以 validation 选 C 或阈值。

## Selection 与终检

selection 中固定比较 9 个候选：`C ∈ {20,40,80}` 与预测优势门槛
`T ∈ {0,8,16}`。每局仍至多替换一次弃牌、随后 Teacher；只要候选改变动作，就以 logged
epsilon-mixture 的 exact propensity 做按墙组 IPS 与**未缩减** DR。每个候选须满足 IPS、DR 的 95%
下界都正，target/base ESS 都 ≥75；通过者按最小下界最大、再按更高 T、再按更低 C 选择。

只有唯一 selection winner 才能读取 terminal 一次。terminal 同样要求两项正下界和 ESS 门槛；通过后也只能进入
200 个全新物理墙的单点 override 实战筛选。任何成功前不得导出网页策略、混入 PPO，或声称强于人类。
