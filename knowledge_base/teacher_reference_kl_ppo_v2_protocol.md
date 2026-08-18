# Teacher-clone reference-KL 小 MLP PPO v2 协议

日期：2026-08-08
状态：classic pilot mechanics 通过；唯一 100 墙 selection 拒绝，配置冻结

## 旧训练为何没有发生有效学习

旧 `teacher-anchored-selfplay-v1b` 从随机 feature-v3 encoder 和零 policy head 出发，以固定 Teacher prior margin 5、
8×256 局、每轮 2 epoch、`5e-5` 学习率训练。新增 aggregate-only 复核在 3,000 个 actor-visible 决策上读取 fresh 与
iteration 1..8 checkpoint，不读牌局结果：iteration-8 最佳替代 residual gap 的 p95 仅 `0.001143`、最大
`0.001572`，所有 margin≥1 的穿透率为 0；margin=5 下非 Teacher 总概率从 fresh 的 `0.045897` 变为
`0.045892`。因此严格零实战分歧不是“学到保持 Teacher”，而是 actor 相对 prior 的量纲和样本预算不匹配。

同时修复 PPO rollout 报告错误：旧代码把所有非流局都计入 candidate wins，造成每轮 256/256 的假胜局；现在只在
`game.winner == candidate_seat` 时计胜。固定 classic 256 局吞吐 smoke 的真实结果为 66 胜，耗时 20.08 秒，即
12.75 candidate episodes/s；按当前实现，100 万局约 21.8 小时，RTX 2080 Ti 可以承担百万级小 MLP 路线。

## v2 与旧 PPO 的实质区别

- 不用随机 encoder；启动 checkpoint 固定为 feature-v3、hidden-128 `candidate_mlp`
  `artifacts/policy-value-contract-v4-candidate-mlp/policy-value.pt`，SHA-256
  `b087a0e496e266de4c9e116fb9a4b5c3950964abdefff24843c5f43ea2aa09df`。它在独立 test 的 Teacher 一致率为
  96.95%，100 墙相对 Teacher 点估计 +0.28、区间跨零；只是强启动点，不是已晋级模型。
- 训练开始时深拷贝完整小 MLP 作为冻结 reference；每个 on-policy 状态加入 forward KL，抑制跨迭代累计漂移。
- policy head 使用独立学习率倍率；共享 encoder/value 仍用基础学习率。每轮报告训练期 KL、最终 KL、总变差、与
  reference 的 argmax 分歧，以及真实胜局。
- 不使用 Transformer、特权 critic、旧 Q/outcome 标签、真人伪标签或手工牌效奖励；回报仍是引擎终局净分。

64 局 core mechanics smoke 固定 `reference_kl_weight=0.05`、policy-head 倍率 10、2 epoch。训练报告 KL 为
0.001046，真实胜局 15/64；在另一组 3,000 个 actor-visible 决策上，最终 actor 与 frozen 起点 argmax 分歧 0.8%、
forward KL 0.002803，Teacher 一致率从 96.73% 到 96.37%。这只证明 actor 能移动且没有立即崩溃，不选择强度参数。

## 固定 classic pilot

- 训练：4 iteration × 1,024 candidate-seat episodes；三名对手全部 frozen `HeuristicTeacherAgent`；rollout batch 64、
  PPO epoch 2、batch 512、基础学习率 `5e-5`、policy-head 倍率 10、reference KL 权重 0.05、clip 0.15、entropy
  0.002、reward scale 80；seed `202646000`。
- 只审计 final iteration-4，不挑中间 checkpoint。
- 先过 mechanics gate：final rollout states 上 reference KL mean ≤0.02、argmax 分歧率在 `[0.5%,10%]`，规则合法率
  100%，胜局统计使用修正语义。未过则不做强度筛选。
- mechanics 通过后，唯一 selection 为全新 classic seed `202647500..202647599` 的 100 个物理墙、候选四座轮换、
  其余三座 Teacher；门槛仍为 paired score delta 95% 下界严格大于 0。
- selection 失败则冻结本配置，不在同墙改 KL、倍率、epoch 或挑 checkpoint；通过也只允许把规模扩到 250,000 局，
  不能直接部署或声称超过人类。

## 正式执行结果

训练先后做了两次相同参数的完整重跑；第二次补齐可复现 provenance 后作为正式 artifact。报告现在记录输入 checkpoint
SHA-256、训练器与 `xiamen_mahjong` Python 源码树 SHA-256、全部采样／优化参数和每轮 rollout/update seed。两次运行
四个 iteration 的 `state_dict` 均逐张量完全一致，证明 provenance 补丁没有改变采样或权重更新。正式目录为
`artifacts/teacher-reference-kl-ppo-v2-classic-pilot-auditable/`；输入 checkpoint SHA-256 仍为
`b087a0e496e266de4c9e116fb9a4b5c3950964abdefff24843c5f43ea2aa09df`，训练源码树 SHA-256 为
`2858e7665f48f7099da5185af03b2cf0118c7f6b31661fdba4266fa6b3588fa6`。

四轮共 4,096 个 candidate-seat episodes、45,102 个候选决策；每轮动作计数均与决策数完全相等，三家对手计数均为
3,072 次 `heuristic_teacher`。final iteration-4 在其 rollout states 上：

- reference KL mean `0.0036966`，低于 `0.02` 上限；
- 与训练起点的 argmax 分歧率 `1.0415%`，位于 `[0.5%,10%]`；
- 真实 candidate wins 为 261/1,024，1 次流局；训练终局 reward mean `+0.01199`。

因此 mechanics gate 通过。随后由预先冻结的 fail-closed selector
`scripts/select_teacher_reference_kl_ppo_v2.py` 唯一读取 classic seed `202647500..202647599`。100 个物理墙各做四座
轮换；候选 99/400 胜、0 流局、总分 `-54`、均分 `-0.135`，Teacher 同墙基线为 100/400 胜、总分 0。按物理墙
配对差为 `-0.135 ± 0.9387`，95% CI **`[-1.9748,+1.7048]`**，未通过下界严格大于 0。

正式状态为 `selection_rejected_configuration_frozen_not_deployed`。这证明 4,096 局 v2 没有提供超过 Teacher 的可靠
证据；宽区间不能证明其一定更弱，但禁止据此复用选择墙调 KL、policy-head 倍率、epoch、训练规模或中间 checkpoint。
250,000 局扩容不解锁，网页仍使用 frozen Teacher。中断目录和早期无 provenance 完整目录只作确定性对照，不参与
任何模型选择。
