# Teacher-anchored residual self-play v1：预注册协议

日期：2026-08-07。fresh PPO v1-a 从随机 actor 直接以稀疏终局净分训练，在独立 selection 相对 Teacher 得到
95% CI `[-23.165, -20.335]` 分/局，故不得通过增加 iteration 或调同一配置继续尝试。本协议定义一个不同候选族：
**规则 Teacher prior + 新初始化 neural residual**。

## 固定策略定义与信息边界

新 actor 的 candidate-MLP 由 `scripts/init_fresh_selfplay_policy.py --zero-policy-head` 创建；所有 policy-head 权重和
bias 均为零，不载入任何历史 checkpoint 或轨迹。对一个合法动作集合，`HeuristicTeacherAgent` 选择的动作获得 prior
logit `0`，其余动作获得 `-margin`；residual network logits 与 prior 相加后再选择或采样。固定 `margin=5`。
因此零 residual 的 deterministic 选择**严格等于** Teacher；网络只学习何时需要以超过该固定余量的公开信息纠正。

Teacher action、residual actor 和 PPO collector 都只读取本家手牌、规则允许的内部本家状态及桌面公开信息。不得读取
牌墙、任一家对手暗手、随机种子、oracle critic、历史 run3/run4 权重、Q/outcome 标签或 hidden-state features。PPO
step 必须持有生成行为时的 prior logits，更新时再次加入同一 prior；snapshot 对手也使用相同 prior，避免训练/推理分布
不一致。网页默认仍是纯 Teacher，所有 residual checkpoint 初始均为 `authorized_for_browser=false`。

## 分阶段门槛

1. **smoke**：新零 head actor 以 Teacher/snapshot 对手完成极小批 classic PPO。验证 prior 存在于每个 PPO step、零非法
   动作、无安全循环、checkpoint 可加载；不读取或比较任何强度 selection 墙。
2. **正式 run**：smoke 后必须另行预注册新随机 seed、完整训练预算和 selection/terminal 墙，且只比较 final checkpoint；
   不按训练 reward、loss 或中间 checkpoint 选模型。
3. **selection / terminal**：与 v1-a 相同，均使用按物理墙四座轮换的 paired score delta 95% 下界严格大于零；selection
   失败即 terminal 未读。通过 terminal 也只获得进入独立异质联赛的资格，绝不自动部署或宣称胜过人类。

该候选的目的只是检验“规则锚定的可见信息 residual 是否能安全探索”；它不是把 Teacher 伪装成神经模型，也不以与
Teacher 初始完全相同作为任何强度证据。
