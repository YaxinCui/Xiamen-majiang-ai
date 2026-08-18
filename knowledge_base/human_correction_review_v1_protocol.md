# 低分歧专家纠错审阅 + 小 MLP v1 协议

日期：2026-08-08
状态：600 题本地题库与全链路基础设施就绪；confirmed 标签为 0，未训练模型

## 为什么增加独立审阅模式

完整真人对局是最终强度评测不可替代的数据，但采集效率低：一局只有几十个普通弃牌决策，真正与 Teacher 不同的
判断更少。资源受限路线不需要先训练一个大序列模型，而是应集中询问 Teacher 最不确定、规则分数接近的状态，再让
熟悉规则的人独立判断。

这里的“专家”表示人工审阅角色，不是自动的能力认证。标签只说明一个独立行为判断；即使与 Teacher 不同，也可能是
人工判断错误。因此后续仍需要 validation/test、Teacher 配对实战和独立真人 evaluation，不能把审阅题的模仿
准确率当作强度。

## 题库构造与复现审计

来源限定为 classic、`collector=teacher_self_play` 的安全轨迹。只选择：

1. 普通 discard phase；
2. 非游金、非金牌锁；
3. 至少两种合法弃牌；
4. frozen `HeuristicTeacherAgent` 第一名与第二名的规则分数差不超过 2.0；
5. 每个物理牌局 group 最多两题。

`scripts/build_human_correction_review_queue.py` 从 actor-visible 决策快照重新计算 Teacher 的
`hand_quality + 18 × waits − 7 × discard_gold`，并验证排序与轨迹中原 Teacher 选择一致。当前固定输入扫描 789 个
Teacher 牌局、34,456 个决策，得到 17,844 个低分差候选；抽取 600 题、覆盖 443 个 opaque group，分数复现
不一致为 **0**。

题库比 500-confirmed 训练门槛多 100 题：审阅者可以诚实选择 `uncertain` 或跳过，而不会因一道不确定题使整个 pilot
永远无法达到门槛。前 500 题在扩容时逐项验证完全不变。

题库不导出 trajectory ID、原 group ID、源路径、终局结果、随机 seed、物理牌墙或对手暗手。item/group ID 都是带
版本 salt 的不可逆摘要。源轨迹的完整公开事件流位于外层文件，独立题目只保留决策快照中的 bounded 最近公开动作，
因此明确写 `public_history_complete=false`，不能误称拥有完整历史。

## 独立选择与追加写入

`scripts/serve_human_correction_review.py` 默认只监听本机。GET `/api/review` 返回手牌、公开玩家信息、合法弃牌和聚合
进度，但不返回：

- `reference_teacher_index`；
- `teacher_score_margin`；
- `review_group_id`；
- 任何隐藏牌或源数据身份。

人类先选弃牌，再选择 `confirmed` 或 `uncertain`。服务端完成追加写入后，才返回本题 Teacher 对照。label 保存
immutable queue digest，重启恢复时逐项校验 state、合法动作、Teacher 参考和 group；已有行不匹配、重复 item 或不属于
当前 queue 时拒绝启动。跳过只在当前浏览会话生效，不伪造标签。

## 盲主动选题优先级 v1

顺序采完 600 题会浪费大量时间在多个自动规则都同意的容易状态上。固定 queue 生成后，另用
`TwoDrawTenpaiReachTeacherAgent(score_margin=2.0, minimum_probability_advantage=0.05)` 从同一个 actor-visible
快照计算一次离线 priority。它只使用本家手牌、牌河、公开副露、金牌指示与公开跟打约束；不读取真实 wall、对手暗手、
赛果或源轨迹身份。每条 priority 都绑定 immutable queue digest、原始 item index 和完整 0–599 rank，缺行、重复 rank、
摘要不符或 Teacher 选择无法从公开快照复现时拒绝启动。

当前 queue SHA-256 为 `9477e7a5099a46616ebbfd217fad1b08500031e21ca1f3f37ceebb4b9ad43ce6`；生成 priority
后复核仍完全相同。priority SHA-256 为 `a1a652144db48c161de6d7a5624cf7af42afde2b1b1862acea176c8267012bfd`。
600 题中 SlowExpert 与 frozen Teacher 分歧 **51** 题，按分歧优先、两摸概率优势降序、原始 index 稳定排序。

这 51 题不是自动正确标签。该 TwoDraw 候选在预注册 100 墙中虽有 +1.335 点正点估计，但 95% CI
`[−0.9195,+3.5895]` 跨 0，已经判为 selection rejected；这里只把它当作 acquisition function，让人工更快看到
可能有信息量的状态。GET `/api/review` 在作答前不发送 priority、SlowExpert、Teacher index、Teacher margin 或 group；
提交后才返回两种自动选择供对照。append-only human label 仍只保存人工选择、置信度和 queue 绑定，绝不写入 priority 或
SlowExpert 选择。

主动排序使“先完成的部分标签”不是 queue 的随机子样本，不能用前 51 题的分歧率估计总体 Teacher 错误率，也不能据此
调整既定训练门槛。训练仍要求 500 confirmed、50 个真实人类/Teacher 分歧和 100 个 group；group split、保留 test、
100/400 墙配对实战与独立真人 evaluation 均保持不变。

## 审计与切分门槛

进入首轮 pilot 前固定要求：

- 500 个 confirmed 标签；
- 至少 50 个 confirmed 人类/Teacher 弃牌分歧；
- 至少 100 个 confirmed 物理牌局 group；
- 无无效字段、重复 item/label、scope 外动作或隐私字段。

`uncertain` 只计采集记录，不进入训练。`scripts/split_human_correction_reviews.py` 按 opaque 原始牌局 group 做确定性
80/10/10 切分；同一局的两道题绝不跨 split。输出仅允许位于 Git 忽略的 `local_human_data/`，并拒绝覆盖已有目录。

## 固定 CPU 小模型

`scripts/train_human_review_residual_v1.py` 固定：

- feature-v3 `candidate_mlp`，hidden 128；
- fresh seed `202623200`，12 epochs，不加载旧 checkpoint；
- 默认 CPU；Teacher v4 轨迹维持全动作安全底座；
- review 标签只提供 policy 行为目标，`value_target=None`，绝不伪造终局分数；
- review weight 1.0，人与 Teacher 分歧固定额外乘 2.0；
- best epoch 只看 `local_human_review_opt_in` validation policy loss；
- 通用训练器只有 review train/validation 参数，物理上没有 review test 参数。

训练前只读取 train/validation，并要求至少 350 个 confirmed 标签、30 个分歧、80 个 group；test 只检查文件存在，字节
保持未读。此门槛是总量 500/50/100 在 80/10/10 切分后的预期下限，不得在看到结果后修改。

## validation/test Teacher gate

由于约 50 条 validation/test 无法支持 1%–5% 覆盖下至少 20 次 override，review v1 单独预注册覆盖率
`{10%,20%,30%}`；这不是沿用完整对局 gate 后临时放宽。validation 每个候选仍必须：

1. 至少 8 次 override、20 个 group；
2. 实际覆盖不超过目标率；
3. override 匹配人工选择的 Wilson 95% 下界大于 50%；
4. group 等权 gated−Teacher 准确率增益 95% 下界大于 0。

validation 全失败时，`scripts/select_human_review_residual_gate_v1.py` 不读取 test。通过后只冻结一个 margin，再对 test
要求至少 8 次 override、20 个 group、覆盖不超过 35%，并重复两个正下界。test 通过只解锁全新 100 墙 Teacher
筛选，不等于超过 Teacher 或人类。

## 强度阶梯与失败解释

1. 100 副新物理墙、四座轮换，对三名 frozen Teacher；paired delta 95% 下界必须大于 0；
2. 通过后才做 400 墙确认；
3. 冻结 checkpoint SHA、wrapper 和 strict margin；
4. 最后才收集从未进入训练/选模的 `recording_purpose=evaluation` 真人局。

若 review validation/test 失败，可能原因包括人工分歧噪声、标签数量不足、feature-v3 无法表达判断依据、Teacher 近分并不
代表错误，或 MLP 在 Teacher 大语料中仍被淹没。正确动作是按预注册诊断决定继续采集、改善审阅质量或回到规则特征；
不能读取 test 后调权重、偷偷加入序列网络、降低置信下界或直接部署。
