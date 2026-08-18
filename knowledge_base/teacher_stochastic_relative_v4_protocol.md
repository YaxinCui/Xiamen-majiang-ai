# Teacher 随机相对优势 v4：预注册协议

日期：2026-08-07。v3 的全部九个确定性 `(C,T)` 相对优势 override 在 selection OPE 失败，terminal 未读。
不得在 v3 selection 墙上重调 C、阈值、网络、direct ensemble 或聚合方式。v4 是**不同的策略族**：不取
相对优势 argmax，而是在 Teacher 和一个温度受限的随机动作分布之间作小概率混合。它仍只表示一次 `discard`
替换后恢复 Teacher，不代表完整对局策略或对人强度。

这一构造使用已知行为 propensity 的 contextual-bandit OPE：IPS/DR 可评价任意在合法动作上有概率质量的目标
分布；训练标签中的 winsorization 仍明确有偏，只能生成候选，最终 gate 必须使用未缩减 IPS/DR。相关的 policy
learning/OPE 背景见 [Counterfactual Risk Minimization](https://proceedings.mlr.press/v37/swaminathan15.html)、
[Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601) 与
[Optimal and Adaptive Off-policy Evaluation](https://proceedings.mlr.press/v70/wang17a.html)；这些是 contextual-bandit
结果，不提供四人麻将或多次改策保证。

## 冻结收集与墙组

- classic 规则；连续 **3,000** 个全新物理墙，seed `202610000` 起；四座轮换。
- 四家均为 `HeuristicTeacherAgent`。每局仅从前 8 个 `discard` phase 决策随机选一个位置；该点按
  `0.8 × Uniform(legal) + 0.2 × Teacher` 执行，之后严格恢复 Teacher，并保存 exact propensity。
- 整墙 train/validation/held-out = 60%/15%/25%，salt `teacher-stochastic-relative-v4-main`；held-out 以
  `teacher-stochastic-relative-v4-heldout` 均分 selection/terminal。
- 随机弃牌干预覆盖门槛为 train ≥5,000、validation ≥1,200、selection ≥1,000、terminal ≥1,000。任一项失败：
  不训练、不读取动作/得分汇总、不改本墙组参数。

## 训练固定项

若且唯若覆盖通过：train 以完整物理墙 K=3 cross-fit direct outcome models。第 i 个模型只用其余两折的
随机弃牌行动，fresh policy anchor `202610101`（128 hidden），固定 32 epoch、batch 256、AdamW lr 0.001、
weight decay 0.0001、score/win/opponent weights 1/.25/.25，shuffle seed `202610110+i`；不读取 validation、
selection 或 terminal。每行的 direct prediction 仅来自未训练该行 fold 的模型。

相对 Teacher DR 标签仍使用 C∈{20,40,80} 的 winsorized residual，且在单独 JSONL 中删除终局结果。每个 C 训练
一个独立 `145→128→128→1` centered ReLU MLP，Teacher 动作按构造为 0；Huber delta 16、batch 256、AdamW
lr 0.001、weight decay 0.0001、固定 32 epoch，shuffle seeds 202610120/121/122。validation 只报告 direct
校准与 pseudo-label tail/gap；不允许据此改变 C、模型、epoch 或下列 policy grid。

## 新的随机策略与 OPE

给定 learner 预测 `A_C(s,a)`，先截到 `[-32,32]` 作为 policy logit（这不是 OPE residual clipping），令

```text
rho_C,tau(a|s) = softmax(clip(A_C(s,a), -32, 32) / tau)
pi_C,beta,tau(a|s) = (1-beta) * 1[a=Teacher] + beta * rho_C,tau(a|s)
```

只比较固定 12 个组合：C∈{20,40,80}，beta∈{0.05,0.10}，tau∈{8,16}。不另设 argmax/LCB 阈值。选择时每行的
IPS 增量为 `R * (pi(A|s)-1[A=Teacher]) / mu(A|s)`；DR 增量为
`sum_a (pi(a|s)-1[a=Teacher]) q(a,s) + (pi(A|s)-1[A=Teacher])*(R-q(A,s))/mu(A|s)`，其中 q 来自不接触
该评测墙组的三个 direct outcome 模型等权均值。所有墙组先求均值再计算 95% 正态下界；target/base ESS 分别由
`pi(A|s)/mu(A|s)` 与 `1[A=Teacher]/mu(A|s)` 的非负权重计算。

selection 的每个候选必须同时满足未缩减 grouped IPS 与 DR 95% 下界 >0、target/base ESS≥100。多个通过时，选
`min(IPS_LCB,DR_LCB)` 最大者；并列时依次选择更小 beta、更大 tau、更小 C。仅 winner 可读取 terminal 一次，且
必须通过相同门槛，才允许进入全新 200 墙随机一次 override 实战筛选。任一阶段不得据此直接接入网页或声称超过人类。
