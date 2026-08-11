# Teacher + 小 MLP 置信门控 v1：预注册协议

日期：2026-08-08
状态：100 墙 selection 未通过；400 墙 terminal 未读取；不得接入网页或声称达到人类水平

## 为什么是这条路线

现有证据同时否决了两个极端：继续扩大同源 Teacher 模仿只能更准确复制 Teacher；fresh PPO 在稀疏终局奖励下又会
远离可用策略。旧 `run3` 是 feature-v3、hidden 128 的小 MLP，checkpoint 仅 218,453 字节；它在独立 80 墙对
Teacher 曾得到 +1.3375 分/局的正点估计，但区间跨 0，不能直接晋级。本实验把它降格为低覆盖 residual，只在模型
非常确信 Teacher 的普通弃牌不是最佳时介入，正是规则策略与纯神经策略之间的中间态。

## 固定组件

- checkpoint：`artifacts/policy-value-classic-v1-run3/policy-value.pt`
- SHA-256：`01616a38ff34af71f1774995d5d3f9e08c4408fd85757ad563d9dfa339b19f70`
- architecture：`candidate_mlp`；feature version 3；hidden 128
- gate：`best_non_teacher_policy_logit - teacher_policy_logit > 2.4152190685272217`
- 唯一允许覆盖的 Teacher 动作类型：普通 `discard`
- response、胡、杠、游金、金牌锁及没有严格越过 margin 的状态全部回退 `HeuristicTeacherAgent`

margin 只由 v4 part-01 validation 的行为覆盖率确定，不读取终局收益：14,851 个合格普通弃牌中覆盖 297 个，
即 1.9999%。在未参与阈值选择的 part-01 test 中，13,519 个合格弃牌覆盖 251 个，即 1.8566%，通过预设
0.5%–3.5% 稳定带。该步骤只证明门控覆盖稳定，不证明强度。

## 强度筛选

- classic 规则，CUDA policy logits。
- selection：`seed=202619000` 起连续 100 副全新物理牌墙；候选四座轮换，其他三座 frozen Teacher，共 400 局。
- reference：相同牌墙和换座协议，候选座也使用 frozen Teacher。
- 独立统计单位为物理牌墙；报告 paired score delta mean、stderr 与 95% CI。
- 唯一通过门槛：`paired_seed_score_delta_95pct_low > 0`。
- 未通过立即淘汰，不能在同墙修改 margin、覆盖动作类型或 checkpoint。
- 只有 selection 通过，才读取 `seed=202619200` 起连续 400 墙 terminal；terminal 同样要求 95% 下界严格为正。
- terminal 即使通过也只获得真人 A/B 评测资格，不自动接入网页，更不能据此宣称“打败人类”。

## 最终目标边界

人类水平必须由经过身份与水平分层、规则确认、盲测和足够局数的真人对局证明。Teacher 配对只是一道工程门槛，
不是最终目标的替代指标。

## 结果

selection 已严格按预注册配置执行。真实候选访问状态中共有 2,866 次合格普通弃牌，模型覆盖 41 次，覆盖率
**1.4306%**；说明离线 1.8566% 与在线访问状态没有灾难性漂移。候选在 400 个四座轮换对局中 98 胜，均分
**−0.7125**；相对同墙 frozen Teacher 的 paired score delta 95% CI 为 **[−2.3297, +0.9047]**。

下界未严格大于 0，且点估计为负，因此状态为 `selection_rejected_terminal_unread`。`seed=202619200` 起的
400 墙 terminal 没有读取；checkpoint、margin 和覆盖范围不得在同一 selection 墙上重调。候选不接入网页、
不作为训练 Teacher，也没有真人评测资格。

该结果直接否决“旧 Teacher 模仿 MLP 的最高置信分歧天然更好”。下一阶段的数据标签必须来自独立强度来源：
经授权并分层审计的真人选择，或在同一个公开信息集上用足够多共享 belief worlds 得到的局部 paired advantage；
不能继续从同源 Teacher 一致率或 policy logit 自举。
