# 真人—Teacher 纠错数据 v1 协议

日期：2026-08-08
状态：采集、审计与轻量训练接口就绪；本地尚无合格真人训练牌局，不启动训练

## 为什么这是新的监督来源

Teacher 自博弈和 DAgger 的目标都是“复制 Teacher”，理论上不能提供超过 Teacher 的动作方向。已有公开代理、旧
MLP logit 和两次 32-world belief 标签试验也都未通过独立门槛。真人选择则是与 Teacher 独立产生的行为标签；其中
真人与 Teacher 不同的决策，才是资源受限条件下最值得学习的局部纠错候选。

这不等于假定真人每次都正确。v1 只建立可审计的数据契约，并要求整副牌局留出验证；不会把单次分歧直接部署。

## 每个决策保存什么

- `chosen_index == executed_index`：真人实际选择；仍是行为模仿目标；
- `reference_teacher_index`：同一状态、同一合法动作集合中 `heuristic_teacher_v1` 的参考选择；
- actor-visible 状态、合法动作与公开历史；
- 完成本局的四家本局分差。

不保存牌墙顺序、任何随机 seed、对手暗手、账号、网络标识或时间戳。参考索引只用于分歧诊断/加权，不作为模型输入。
训练与 evaluation 文件用途在采集时固定，后者永远不得进入训练或选模。
记录模式下服务端强制拒绝 `?debug=1` 的对手暗手揭示，并写入
`opponent_hand_reveal=server_forced_disabled`；缺少该标记的旧记录一律不通过审计。仅在 JSON 中删掉暗手不够，
因为看过暗手的人类选择本身已经携带特权信息。

## 数据门槛

运行 `scripts/audit_human_teacher_corrections.py`，必须同时满足：

1. 至少 100 个结构合格、无重复的完整 training 牌局；
2. 所有真人决策的 `reference_teacher_index` 覆盖率为 100%，其中至少 500 个属于 gate 可表示的普通
   “弃牌覆盖弃牌”决策；
3. 上述普通弃牌决策中至少 50 个真人与 Teacher 的牌面选择不同；响应、胡、杠、游金和金牌锁分歧不凑数量；
4. 单一 classic 规则版本、单一对手策略身份和单一参考 Teacher 身份；
5. 人工确认记录者知情、规则熟悉度和数据质量。

门槛未过时不训练，也不通过降低门槛、把 evaluation 局混入或按单决策随机切分来补数量。

## 第一轮轻量模型

通过后按完整牌局切成 train/validation/test。模型限制为 fresh candidate MLP（不使用 Transformer/GRU，也不读取
旧 run3/run4 权重）；Teacher 语料提供行为底座，再用真人 `chosen_index` 提供纠错。分歧额外权重在读取数据前固定为
`2.0`，不做 `{1,2,4}` 网格，避免多个模型反复读取真人 test。具体 gate 与强度阶梯见
`human_teacher_residual_gate_v1_protocol.md`。
训练编码同样启用 `--human-discard-corrections-only`，因此审计、优化目标和网页 wrapper 使用同一个状态子集。

离线通过也不能晋升。候选先做全新 100 墙四座 Teacher 配对筛选，95% paired 下界须大于 0；之后才允许冻结
checkpoint，另收集 `recording_purpose=evaluation` 的独立真人对局。真人评测数据不得回流训练、early stopping、
权重选择或规则修改。
