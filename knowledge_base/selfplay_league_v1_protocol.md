# 从零初始化自博弈联赛 v1：预注册协议

日期：2026-08-07。Teacher 克隆、历史 neural checkpoint 延续、局部 belief-Q、单点 OPE residual、公开防守规则
v5 均没有通过独立强度门槛。因此此路线不继承 `run3`、`run4`、旧的 Teacher-clone 或任何被否决模型参数。

## 目标与固定训练身份

唯一训练起点由 `scripts/init_fresh_selfplay_policy.py` 以固定随机种子创建：candidate-MLP、feature version 3、128
hidden units、`policy` action head。checkpoint metadata 必须有 `ancestry=none` 和 `not_trained=true`；初始化报告记录
checkpoint SHA-256、seed 与 source revision。初始化 checkpoint 仅有资格进入训练，绝不进入网页。

训练只使用合法动作掩码、行动者本家手牌与公开状态，以及局终净分。不得使用对手暗手、牌墙、oracle critic、历史
run3/run4 权重、历史训练轨迹或 outcome/Q 标签。每一 PPO iteration 的对手在更新前冻结：每个非候选座位以固定比例
使用 `HeuristicTeacherAgent` 或当前策略快照；快照不共享参数或梯度。每轮的对手计数、source revision、seed、训练
参数和 checkpoint digest 都必须写入报告。

## 执行阶段与门槛

1. **管线 smoke**：fresh actor 用 classic 规则做一轮很小的 PPO；只检查零非法动作、zero safety-loop overflow、
   checkpoint 可加载、actor observation 无 hidden fields。它不是强度实验，不能据此选择参数或打开任何评测墙。
2. **预注册训练 run**：在 smoke 成功后另行固定初始 seed、iteration 数、每轮 episode 数、PPO 参数以及
   Teacher/current-snapshot 比例。训练过程中不按 rollout reward、胜率、loss 或中间 checkpoint 选 champion；只能评测
   最后一轮 checkpoint。
3. **selection**：最终 checkpoint 仅在全新 classic 物理墙上与三名 Teacher 四座轮换。要求按物理墙配对 score delta
   的 95% 下界严格大于 0，才可打开 terminal。该筛选只证明超越该冻结 Teacher，不等同于人类强度。
4. **terminal 与联赛**：唯一 selection winner 必须在另一批未读墙再次通过相同下界；此后还须对一个预先声明的异质
   合法对手阵容通过独立门槛。没有经过质量审计的人类对局前，任何结果不得称为“击败人类”。

每一阶段失败即停止该固定 run：不在其 selection/terminal 墙上改 learning rate、熵权重、对手比例、iteration、
checkpoint 或评测种子重试。原始 PPO actor checkpoint 与中间结果保留为本地 ignored artifact，只有通过完整门槛的
模型才可能提交为可部署参数。
