# 两摸听牌精确确认 Teacher v2：预注册协议

日期：2026-08-08
状态：100 墙 selection 未通过；不部署、不产训练标签

## 反证边界与新假设

`TwoDrawTenpaiReachTeacherAgent` v1 在 100 墙得到 `+1.335` 分／局，但 95% CI
`[-0.9195,+3.5895]` 跨零；首分歧二元随机干预在另一组 100 墙得到 HT `+1.945`，95% CI
`[-3.3906,+7.2806]`。二者都没有通过门槛，不能部署、不能产监督标签，也不能在原墙上调整 5% 阈值或 32 场景数。

v1 同时存在一个可独立修正的算法误差：它以 32 个分层分位近似两次本家摸牌，并在第一摸后按
`(最低向听、最高 hand_quality)` 贪心弃牌；这不等于“最大化第二摸前进入听牌的概率”。v2 不新增提议状态，
而把 v1 固定候选当作 acquisition/proposal：只有 v1 已经建议覆盖 Teacher 时，才对这一个动作和 Teacher 动作做
精确两步动态规划。因而 v2 只能确认或删除 v1 override，不能从旧 selection 事后搜索新动作。

## 固定两级候选

候选为 `ExactTwoDrawTenpaiReachTeacherAgent(score_margin=2.0,
minimum_probability_advantage=0.05)`：

1. v1 proposal 固定使用 32 个场景、2.0 Teacher 分差和 5 个百分点近似优势；
2. 没有 proposal 时严格返回 frozen `HeuristicTeacherAgent`；
3. 有 proposal 时，从本家手牌、公开河／副露和翻金构造 34 面额公开未见副本；
4. 对 proposal 与 Teacher 分别精确枚举两次无放回本家摸牌；第一摸后枚举所有公开规则允许的弃牌，并选择第二摸
   “已胡或进入听牌”概率最大的分支；立即胡先于荣牌跟打限制，顺序与引擎一致；
5. proposal 不得增加精确五面子向听，且其精确概率仍须至少高 5 个百分点，否则删除该 override；
6. response、胡、杠、游金、金牌锁、非 classic 及 v1 scope 外状态全部 frozen。

算法不读取真实 wall、seed、未来 RNG、对手暗手或他家暗杠牌面。公开未见副本是 actor-visible tile-efficiency 模型，
不是对真实牌墙位置的 oracle，也没有按对手行为构造后验。

## 只读结构与成本诊断

在既有不可变人工审阅 queue 的前 100 个 actor-visible 低分差状态上只比较动作，不读取源局赛果：v1 提议 12 次，
精确 DP 确认 8 次、否决 4 次，故精确层确实删除了三分之一 proposal；v1/v2 动作分歧 4 次。单进程 CPU 耗时
156.20 秒。它若通过强度门槛，也只优先作为离线标签器；在没有蒸馏和延迟门槛前不直接接入网页。

## 固定 100 墙 selection

- classic，`seed=202636100..202636199`，每个物理墙四座轮换，其他三座 frozen Teacher；
- 同墙四座 frozen Teacher 为 reference，按物理墙均分配对；
- 唯一通过条件：`paired_seed_score_delta_95pct_low > 0`；
- 按项目常规策略只有 100 墙，无事后 400 墙、阈值、场景数、动作源、DP horizon 或 seed 重试；
- 失败即不部署、不产训练标签；通过也只授权另写**离线标签与小模型蒸馏协议**，不自动替换网页 Teacher，且不构成
  胜过人类的证据。

固定入口：

```bash
.venv/bin/python scripts/select_exact_two_draw_tenpai_reach_teacher_v2.py \
  --output artifacts/exact-two-draw-tenpai-reach-v2/selection-report.json
```

## 正式结果

固定 selection 已完成。候选 400 局中 104 胜、1 流局、总分 `+170`，均分 **+0.425**；按 100 个物理墙
配对的标准误为 `0.9612`，95% CI **`[-1.4590,+2.3090]`**，未通过严格正下界。

3,207 次弃牌调用中，v1 proposal 184 次（5.74%）；exact DP 确认 102 次、否决 82 次，实际 override 3.18%，
proposal 条件确认率 55.43%。最小实际精确概率优势为 5.061 个百分点，向听恶化为 0。说明精确层确实删除近一半
近似提议，但没有把 v1 的方向性正点估计转化为可重复强度证据。

Teacher baseline 耗时 15.82 秒，候选耗时 **2,031.86 秒（33.86 分钟）**，约慢 128 倍；即使强度通过也只能作为
离线专家。本轮状态固定为 `selection_rejected`：不部署、不蒸馏、不产标签、不在本墙上改变 5% 阈值、32 proposal
场景、horizon 或 exact utility。TwoDraw 局部牌效族至此冻结，后续转向更大影响的响应／杠／游金决策，或真正的
人工纠错来源。报告 SHA-256 为
`3173fd8e076c6e8f94614c4eb1cba68c613176738bcbba6c1a762fc221065e27`。
