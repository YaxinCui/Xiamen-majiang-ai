# 真人纠错小 MLP + Teacher gate v1 协议

日期：2026-08-08
状态：训练、选门和网页推理基础设施就绪；真人数据为 0，未训练、未读取人类 test

## 目标与模型边界

本路线是规则模型与纯神经网络之间的中间态。网络不接管整局：frozen `HeuristicTeacherAgent` 默认拥有全部动作，
小 MLP 只能在普通、非游金、非金牌锁的摸后弃牌状态中，以另一张**弃牌**覆盖 Teacher。胡、杠、吃碰响应、游金、
金牌锁以及 gate 外状态严格回退 Teacher。网络观察仍只有本家手牌和公开信息。

这修正了旧 run3 gate 的两个问题：旧网络只学 Teacher，logit gap 不是优势；新网络会读取 opt-in 真人动作，而且阈值
必须在完整真人牌局留出集上证明比 Teacher 更接近真人。旧 run3/run4 权重不作初始化；首轮 candidate MLP 使用新
随机初始化，Teacher 语料只提供安全行为底座，真人训练局提供独立纠错方向。

## 固定首轮训练

- 架构：feature-v3 candidate MLP，hidden 128，不使用 Transformer/GRU；
- 初始化：fresh seed `202623100`，不读取任何旧 checkpoint；
- Teacher 语料：现有 trajectory-v4 的完整牌局 train/validation/test；
- 真人语料：只接受 `recording_purpose=training`、通过隐私/重复/参考标签审计并按完整牌局切分的数据；
- `human_weight=1.0`，Teacher 分歧额外乘数固定为 `2.0`，不网格搜索；
- best epoch 只按 validation 的 `local_human_opt_in` policy loss 选择，避免被大量 Teacher 决策淹没；
- 真人 test 不参与 epoch、学习率、权重或架构选择。

固定入口为 `scripts/train_human_teacher_residual_v1.py`。它只把真人 train/validation 传给通用训练器，设置
`--reserve-local-human-test-for-gate`；真人 test 只检查文件存在，不打开其内容、不进入训练报告。训练器的普通模式
仍要求三 split，但 reserve 模式会反向拒绝任何 human additional-test，防止命令拼错后静默消耗终检。

100 局/500 个可表示普通弃牌决策/其中 50 个弃牌分歧，只是“允许首次 pilot”的数据门槛，不保证 validation/test 已有统计功效。门槛不足以支撑
gate 下界时，结论只能是继续收集，而不是降低置信要求。

## validation 选门

`scripts/select_human_teacher_residual_gate.py` 只在真人 validation 比较预声明覆盖率 `{1%,2%,5%}`。每个覆盖率由
positive `(best alternative discard logit − Teacher discard logit)` 的严格分位阈值实现；并列边界整组丢弃，不随机
拆分。候选必须同时满足：

1. 至少 20 次真实 override，且至少覆盖 20 个完整牌局 group；
2. 实际覆盖不超过对应目标率；
3. override 动作匹配真人选择的 Wilson 95% 下界严格大于 50%；
4. 按完整牌局等权计算的 gated−Teacher 动作准确率增益 95% 下界严格大于 0。

若没有候选通过，真人 test 文件保持物理未读。若有多个通过，只选择完整牌局增益下界最高者；再以精度下界和更低
覆盖率做确定性并列裁决。

## test 与强度阶梯

validation 通过后才读取真人 test 一次。test 必须至少 20 override、20 个 group、覆盖率不超过 7.5%，且重复上述
两个正下界。通过仅得到 `ready_for_100_wall_teacher_screen`：

1. 用 checkpoint SHA-256、严格 margin 和 `human_correction_discard_gate_v1` wrapper 冻结候选；
2. 全新 100 副 classic 物理墙、四座轮换对三名 Teacher；paired score delta 95% 下界须大于 0；
3. 通过后才开 400 墙确认；
4. 再单独收集 `recording_purpose=evaluation` 的真人比赛，绝不回流训练或 gate 选择。

网页只有显式同时传入 `--ai-checkpoint` 与 `--ai-teacher-gate-margin` 才加载该 wrapper；默认仍是规则 Teacher。
identity 同时记录 checkpoint SHA、wrapper 版本和完整 margin，防止不同候选混入同一真人评测文件。
