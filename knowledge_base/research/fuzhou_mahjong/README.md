# 抚州麻将区

专项来源：

- [仓库主页](https://github.com/YaxinCui/fuzhou-mahjong-ai)
- [训练技术报告](https://raw.githubusercontent.com/YaxinCui/fuzhou-mahjong-ai/main/docs/technical_training_report_20260520.md)
- [Artifact policy](https://raw.githubusercontent.com/YaxinCui/fuzhou-mahjong-ai/main/artifacts/README.md)

访问日期：2026-08-08。来源等级：A（项目自身公开仓库、报告与源码）。本次源码审计固定到 commit
`53f73d2495c4a49790ab2af7c064906cd5d9fd2b`。

## 已核实事实

公开报告称，当前生产 agent 是 hybrid：神经策略 checkpoint、公开信息 lookahead、Teacher 融合、记忆风险和弃牌进展共同参与推理；
报告给出的 Teacher 权重为 20、lookahead 待牌价值权重为 2.0、记忆风险为 0.25、弃牌进展为 1.0。它不是纯神经网络直接替换规则 Teacher。

报告描述的有效路线是：规则引擎 → heuristic/lookahead 监督数据 → route/score/shanten/wait 辅助损失 → DAgger → Rust rollout/数据加速 → hybrid →
duplicate-seat paired evaluation。报告还明确写出从零 PPO/共享 DQN、纯终局 Monte Carlo Q 等路线不稳定或未作为主线。

公开晋级证据使用 16,000 duplicate sets / 64,000 hands 的 teacher-lineup 和 opponent-pool 评测，并同时看平均分、胜率和放铳率。这说明它把强度证据放在大规模换座配对，而不是训练 loss 或单局胜率；该规模是抚州项目的生产晋级证据，不代表厦门每次迭代都必须使用同样规模。

## 2026-08-08 源码级复核

本次不只读取技术报告，还核对了 `nancheng_mahjong/agents.py`、`nancheng_mahjong/models.py`、生产 agent JSON 和长期
handoff 记录：

- 生产 `BlendedModelAgent` 先把网络 logits 和 reference 分数分别标准化，再计算
  `model + 20 × reference + progress − risk`；reference 实际是 `GreedyStructureAgent(use_wait_lookahead=True,
  wait_value_weight=2.0)`，不是 heuristic Teacher。
- `progress_static_risk` 只在 reference 最高分存在多个动作时强制消歧；生产 JSON 明确写明当前提升只改了 reference
  wait-value，神经 checkpoint 没有改变。
- 长期诊断记录给出的两个 200-set 样本中，hybrid 相对 lookahead 的分歧几乎全是 reference tie-break：
  `1744` vs base-model blend `11`，以及 `1737` vs `7`。因此生产强度主要来自规则 lookahead 与公开 tie-break，
  不能表述成“纯神经网络已超过规则模型”。
- 报告所称 route/score/shanten/wait auxiliary loss 和 DAgger 确实存在，但它们更多改善纯模型近似能力；最终生产仍保留
  高权重 reference。

对厦门最重要的修正是：先验证 reference/tie-break 本身，再训练网络压缩或消歧。依此实现的 exact-score
`shanten + public ukeire` Pareto 候选在全新 100 墙改变 24.53% 弃牌，却得到均分 −0.08、区间跨零且慢 60.46×。
所以只迁移系统结构，不迁移 `progress` 指标；厦门仍缺少可验证的长期 tie-break 标签。

## 可迁移到厦门的内容

- hybrid 推理结构：网络负责泛化，Teacher/公开 lookahead 负责规则锚定。
- route、score、shanten、wait 多任务标签，而非只做动作分类。
- DAgger 覆盖候选真实访问状态。
- Rust/批量环境只用于加速，先做 Python parity 再用于训练数据。
- duplicate-seat 与异质对手池双重评测。

## 不能直接迁移的内容

- 抚州/南城规则、十三烂、买马和结算不能替换厦门规则。
- 抚州 checkpoint、权重、牌谱和训练标签不是本项目输入；当前未把它们复制进仓库。
- 报告中的大规模分数不能证明厦门策略有效；必须重新用厦门规则、厦门的隐藏信息边界和本项目 Teacher 基线评测。

## 对厦门项目的实验翻译

先得到一个经厦门数据验证的 reference/tie-break 标签源，再实现 `teacher_anchored_hybrid`：网络 logits 只在 reference
真正并列或长期价值门控通过时获得小残差；所有特殊规则由引擎决定。100 墙做快速筛选。当前公开牌效 hard proxy 已
失败，下一标签源应是盲态人工纠错或新的高支持随机干预，而不是复制抚州权重。
