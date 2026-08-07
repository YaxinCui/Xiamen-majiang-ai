# Teacher 相对优势 shrinkage v3：预注册协议

日期：2026-08-07。v2 在 selection 随机干预覆盖 `693 < 700` 时被训练前拒绝，不能降低
门槛、改 split 或复用其墙组。v3 只改变**全新的独立物理墙数量**，其余 data、cross-fitting、
shrinkage 与 OPE 契约完全沿用 v2。

## 冻结收集参数

- classic 规则；连续 2,400 个全新物理墙 seed `202609200` 起；四座轮换。
- 四家均为 `HeuristicTeacherAgent`；每局在前 8 个 `discard` phase 决策中最多干预一次。
- 干预行为固定为 `0.8 × Uniform(legal) + 0.2 × Teacher`，随后严格恢复 Teacher；每条
  执行记录保存 exact propensity。
- 整墙按 train/validation/held-out = 60%/15%/25% 切分；held-out 用新 salt
  `teacher-advantage-shrinkage-v3-heldout` 均分 selection/terminal。

随机弃牌干预覆盖门槛不变：train ≥3,500、validation ≥800、selection ≥700、terminal ≥700。
任一失败即不训练、不开标签汇总、不调整本墙组的任何参数。

## 条件通过后的固定训练与选择

若且唯若四项覆盖均通过：train 按完整物理墙 K=3 cross-fit direct outcome models；每行
direct prediction 必须来自未训练其 fold 的模型。相对 Teacher 标签以 direct gap 加 DR residual，
每个 influence 在 `{20,40,80}` 分之一 winsorize。每个 C 使用独立 relative-advantage learner，
不复用旧 synthetic action-Q 语义或网页选择入口。

### Direct model 的泄漏防护（训练前补充固定）

v3 采集运行期间尚未读取任一分区的动作或终局标签；为使 outer-fold 的 direct prediction 不会通过
early stopping 间接读取其预测墙组，direct model 的训练程序固定如下：对第 `i` 个 outer fold，只用
其余两个完整墙 fold 的随机 `discard` 干预记录训练 32 epochs（batch 256、AdamW learning rate
0.001、weight decay 0.0001、score/win/opponent loss weights 1/.25/.25），不读取 validation、selection
或 terminal，并保留第 32 轮权重。三项模型共享从未训练的 policy anchor seed `202609201`（hidden
size 128），训练 shuffle seeds 固定为 `202609310+i`。每个 outer-fold 行只接收对应排除该 fold 的单一
direct model；validation 的 calibration/tail/gap 审计使用这三个模型的等权预测均值，且不反向选择
epoch 或 hyperparameter。训练器需在报告中显式记为 `fixed_pre_registered_epoch`。

validation 只做 calibration/tail/gap 审计。selection 固定比较 `(C,T)` 的九个组合，
`C∈{20,40,80}`、预测优势门槛 `T∈{0,8,16}`；最终仍以未缩减 grouped IPS 与 DR 的双正 95%
下界及 target/base ESS ≥75 筛选。terminal 只在唯一 winner 后读取一次，随后仍必须通过相同门槛，
才可进入独立 200 墙单点 override 实战筛选。无阶段可直接推出网页部署或人类强度。
