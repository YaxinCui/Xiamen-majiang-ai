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

## 首个固定 run（v1-b）

v1-b 唯一初始化为 classic、`seed=202611950`、feature version 3、hidden size 128 的 zero-policy-head fresh actor，
固定 `margin=5`。训练固定为 8 iteration、每轮 256 个候选席、rollout batch 64、PPO epoch 2、batch 256、learning
rate `5e-5`、clip ratio `0.15`、value weight `0.25`、entropy weight `0.002`、reward scale `80`。每名非候选座位
独立以 `0.5` 概率使用 Teacher、`0.5` 概率使用本轮更新前 current-policy snapshot；无外部 checkpoint、无 oracle
critic，CUDA，训练 seed `202611951`。训练中只记录中间 checkpoint，不选择它们。

仅 final iteration-8 raw residual checkpoint 加 `margin=5` wrapper 参与 selection：classic `seed=202612900` 起 80 个
物理墙、四座轮换。若 paired score delta 的 95% 下界不严格大于零，terminal `seed=202613100` 起 320 墙必须保持未读；
若 selection 通过才读取 terminal，且采用同一严格正下界。无论结果如何都不接入网页、不开人类强度主张，也不在本墙组
上更改 margin、训练预算、PPO 参数或选择其他 iteration。
