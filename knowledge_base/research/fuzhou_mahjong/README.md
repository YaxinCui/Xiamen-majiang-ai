# 抚州麻将区

专项来源：

- [仓库主页](https://github.com/YaxinCui/fuzhou-mahjong-ai)
- [训练技术报告](https://raw.githubusercontent.com/YaxinCui/fuzhou-mahjong-ai/main/docs/technical_training_report_20260520.md)
- [Artifact policy](https://raw.githubusercontent.com/YaxinCui/fuzhou-mahjong-ai/main/artifacts/README.md)

访问日期：2026-08-07。来源等级：A（项目自身公开仓库和报告）。

## 已核实事实

公开报告称，当前生产 agent 是 hybrid：神经策略 checkpoint、公开信息 lookahead、Teacher 融合、记忆风险和弃牌进展共同参与推理；
报告给出的 Teacher 权重为 20、lookahead 待牌价值权重为 2.0、记忆风险为 0.25、弃牌进展为 1.0。它不是纯神经网络直接替换规则 Teacher。

报告描述的有效路线是：规则引擎 → heuristic/lookahead 监督数据 → route/score/shanten/wait 辅助损失 → DAgger → Rust rollout/数据加速 → hybrid →
duplicate-seat paired evaluation。报告还明确写出从零 PPO/共享 DQN、纯终局 Monte Carlo Q 等路线不稳定或未作为主线。

公开晋级证据使用 16,000 duplicate sets / 64,000 hands 的 teacher-lineup 和 opponent-pool 评测，并同时看平均分、胜率和放铳率。这说明它把强度证据放在大规模换座配对，而不是训练 loss 或单局胜率；该规模是抚州项目的生产晋级证据，不代表厦门每次迭代都必须使用同样规模。

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

先实现一个 `teacher_anchored_hybrid`：网络 logits 只在 Teacher 与公开 afterstate value 接近时获得小残差；所有特殊规则由引擎决定。
100 墙做快速筛选，只有通过下界门槛才扩大到 400 墙，并同时加入随机/规则异质对手。重点不是复制抚州权重，而是复制“规则锚定、数据分层、配对晋级”的实验纪律。
