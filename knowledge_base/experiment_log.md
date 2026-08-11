# 训练实验日志

本日志只记录已运行、可定位的实验；未通过实战门槛的检查点不作为网页默认 AI。
评测均为「每个初始牌墙让候选轮换四座、其余三座为冻结 Teacher」；标准误以每个
牌墙四座均分为独立样本计算。

## 2026-08-08：Teacher 响应因果错误地图 v1（validation 拒绝）

为避免继续用局部牌效代理修改吃碰，新增 aggregate-only 因果错误地图。它只读已有 response-v2 单点随机干预，在 train
按 `chi × 三阶段`、`pong × 三阶段 × 荣／序数牌` 建立互斥类别，目标统一为 `pass−Teacher claim`。2,332 次随机
response／939 个墙组中，`pong→pass` 汇总为 **−8.913**、95% CI **[−16.091,−1.735]**；荣牌碰为
**−13.614**、**[−25.309,−1.920]**。因此削弱碰牌有直接反证。`chi→pass` 汇总为 +2.450，但区间
`[−5.017,+9.917]`，只用于冻结一个待筛选假设。

在读取 validation 前固定唯一候选：只把随机选中的 Teacher-chi 改为 pass，随后恢复 Teacher；门槛要求 100 次
override、75 组、conditional 双侧 ESS≥30，且 conditional 与全部响应位置 IPS 的 95% 下界均正。v2 validation 有
338 次 override／194 个 conditional 墙组，ESS 64.69/262.49；conditional 点估计 **+7.406**，95% CI
**[−6.905,+21.718]**，全部位置为 **+5.511**、**[−4.519,+15.540]**。覆盖通过但双下界失败，状态
`validation_rejected_no_targeted_collection`。不收集新二元干预、不细分 validation、不读取 terminal、不部署或训练。

## 2026-08-07：可观察牌桌与吃碰后续牌效 Teacher v1

网页新增服务端单步 AI 推进：可选即时、5 秒、10 秒、15 秒。慢速时每次仅结算一项 AI 决策，
使弃牌、吃碰杠与响应能停留在桌面上观察；切回即时才恢复原有自动循环。另新增同浏览器的
“四人手动验规则”模式：四席均为手动座位，响应按 claim 优先级逐席确认，默认只向当前操作者
显示其私有手牌；显式调试开关才展示所有手牌。该模式不写入人类训练或评测记录。

同时审计发现 frozen Teacher 的 `chi`/`pong` 比较发生在“已消耗两张牌、却尚未执行规则要求的
立即弃牌”的中间手牌。`MeldContinuationTeacherAgent` 因而只在吃碰响应处枚举后续合法弃牌，其他
弃牌、杠、游金、胡和明杠保持 frozen。它只使用本家手牌与公开信息，并以
`hand_quality + 18 × 等待牌种数 − 打金惩罚` 对后续弃牌评分。协议、固定阈值与未读终检约束见
[吃碰后续牌效 Teacher v1](meld_continuation_teacher_v1_protocol.md)。

| `minimum_claim_gain` | 新 classic selection（seed 202613200 起 160 墙、四座换位）相对 Teacher | 结论 |
| ---: | ---: | --- |
| 0 | −0.35 ± 0.36，95% CI **[−1.06, +0.35]** | 未通过正下界。 |
| 16 | +0.65 ± 1.21，95% CI **[−1.73, +3.03]** | 方向不确定，未通过正下界。 |
| 32 | −6.12 ± 1.35，95% CI **[−8.78, −3.47]** | 显著变差。 |

三项均未通过，独立 400 墙终检（seed 202613500 起）按协议**没有读取**。候选不接入网页默认
Teacher，也不作为训练标签来源；当前默认仍为 `HeuristicTeacherAgent`。该反证表明，仅修正吃碰后
的局部时序不足以构成可验证的整体强度提升，后续应先由四人手动验桌核对地方规则与实际响应偏好，
再提出新的、互不重复的专家策略假设。

## 2026-08-07：外部抚州／南城项目来源审计（无训练）

只读核查了用户指定的 `YaxinCui/fuzhou-mahjong-ai` commit
`53f73d2495c4a49790ab2af7c064906cd5d9fd2b`。其 README 明确说明项目是**江西
抚州／南城**而非福建福州，且采用 136 张无花、13／14 张起手、不可吃等不同规则；厦门
classic 则为 144 张含花、16／17 张和可吃／游金等规则。因此不导入其代码、权重、牌谱、
动作标签或价值目标。审计中也未发现明确的 `LICENSE`／`COPYING`，所以不能以「参考」为由
复制其实现或 artefact。

保留的只是方法层启示：轨迹标注与模拟器 parity、先模仿再自博弈、以及带产物身份的独立
配对晋级。详见 [外部项目迁移审计](external_transfer_audit_2026-08.md)。该审计没有产生新数据、
模型或强度结论；当前仍须先获得经审计的同规则真人数据，或独立验证更强的公开信息 Teacher。

同日也只读核验了一个声明含厦门玩法、许可证为 Mulan PSL v2 的 Gitee 游戏服务仓库。它没有发布
神经训练器或专家牌谱，内置只是固定启发式；其事件 recorder 可以含行动者完整手牌，因此更不是可直接
导入的 actor-visible 数据集。没有复制代码或记录。详见 [训练数据来源审计](data_source_audit_2026-08.md)。

回归复核还发现，已淘汰的 `core_public_history_constraint_repair_v0` 在当前固定 seed 重放中为
79/100 接受（原测试的 `>80%` 边界不再成立），但原始 rejection sampler 仍低于 10%，故测试改为验证
`>75%` 的“消除塌缩”性质，同时保留 ESS、私有信息隔离和 audit-only 约束。它的 ESS 仍远低于训练门槛，
不改变该路线「不可用于 collector/Q/网页」的结论。

为补齐未来人类数据的可复现入口，新增 `split_human_trajectories.py`：它先执行本地 opt-in 数据的完整
结构审计，再按整局／opaque `split_group_id` 确定性切分 train、validation、test，拒绝重复、混合规则／
对手、私有字段、空 split 或默认覆盖。输出只允许落在 Git 忽略的 `local_human_data/`。这只是将来行为
模仿前的数据准备；当前该目录为空，尚未训练任何模型。

进一步把记录的用途变为不可混用的 source metadata：启动网页记录时必须指定 `training` 或 `evaluation`。
训练器与切分器只接受前者，后者一律失败关闭；新增 `audit_human_match_strength.py` 只读汇总一个冻结 AI 三人组
相对真人座位的分差和 95% 下界。该工具不记录身份、不生成标签、不上传数据；正下界也只可供人工核验参与者同意、
招募质量、冻结 AI 身份和训练隔离，不能自动写成“击败人类”。这仍是证据基础设施，没有模型更新或真人结果。

为避免把同一位玩家的连续多局误当成许多独立人类样本，记录器随后为每次本地启动自动加入随机、无身份信息的
`recording_session_id`。真人评测审计按 session 均分计算 AI 方分差及其下界，并要求至少 10 个 session；它不导出
session ID。本项目不能据此自动证明这些 session 就是不同玩家，故报告仍要求人工核验预注册的参与者／区块映射。
重复牌局指纹刻意排除 session ID，防止复制同一局到新会话来绕过重复审计。没有真人数据、模型参数或强度结论改变。

2026-08-08 复核又修正了报告解释口径：原 `ai_side_score_delta_*` 实际是三张相同 AI 座位的零和合计，并非单个 AI
座位分差。审计器保留旧字段兼容，同时新增 `ai_team_*` 和除以三的 `ai_per_seat_*`；正式人工复核只解释每席指标。
这不改变正下界的符号，却防止把强度幅度夸大三倍。独立
[真人强度基准协议](human_strength_benchmark_v1_protocol.md) 同时把“超过本地预注册群体”与“超过一般人类”分开；当前
没有候选通过 Teacher 400 墙，因此没有启动真人终评。

同日也审计了一个看似可用于「公开事件表征预训练」的候选：MahjongLM 的处理后 Tenhou 日麻数据集及其
100M 权重。数据卡明确标为 `source-data-terms-apply` 并要求接受访问条件，模型卡也要求下游核验原始
数据条款；令牌流还含日麻选项／决议和某些视图的完整牌墙。故没有下载、初始化、蒸馏或提取任何内容。
在取得相容的原始来源授权及 fail-closed 脱敏转换器前，跨变体预训练不构成训练信号。详见
[训练数据来源审计](data_source_audit_2026-08.md)。

## 2026-08-04：监督 v1、DAgger、前瞻 v2

| 检查点 | 训练 | 独立评测种子 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| `rule-policy-classic-mlp`（v1，76 维） | 40 Teacher 局 + 136 游金课程，6 epoch | 20,900,000 起的 100 个牌墙 | 平均净得分 **−8.44 ± 1.59**，胡率 14.25%。 | 明显弱于 Teacher，不上线。 |
| `dagger-classic-run1`（v1） | v1 起点；40 Teacher 局 + 136 课程；3 轮各 20 策略局、每轮 4 epoch | 同上 | 均分 −8.35，胡率 17.25%。 | 点估计改善小于噪声；不宣称更强。 |
| `rule-policy-classic-mlp-v2`（80 维） | 40 Teacher 局 + 136 课程，6 epoch | 同上 | 平均净得分 **−7.54 ± 1.70**，胡率 14.5%。保留集 Teacher 动作一致率 78.5%。 | 比 v1 点估计更接近 Teacher，但仍显著为负；保留为研究候选，不上线。 |

训练种子与独立评测种子不重叠。v1 的早期评测曾与训练种子重叠，已废弃，不能用于
任何比较。下一次候选进入网页前，必须在至少 400 个全新牌墙上重新评测，并要求其
对 Teacher 的净得分置信区间不劣。

## 2026-08-05：DAgger v2、公开余牌 Teacher 与首轮在线 RL

| 检查点 / 对手 | 训练或选择方式 | 独立评测 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| `dagger-classic-v2-run1`（80 维） | 在 v2 上收集策略访问状态并以冻结 Teacher 标注；保留集动作一致率从 74.78% 升至 83.89%。 | 21,000,000 起 100 个牌墙 | 平均净得分 **−3.57 ± 1.65**，胡率 20.5%。 | 是当前最接近 Teacher 的研究候选，但置信区间仍未证明已胜过 Teacher。 |
| `AvailabilityTeacherAgent` | 对听牌按公开可见的剩余张数加权；系数在 20,800,000 起 50 墙上选择。 | 21,000,000 起 100 个牌墙 | 平均净得分 **−0.58 ± 0.79**，胡率 23.5%。 | 选择集上的正向点估计没有在留出集复现；Teacher 保持不变。 |
| 80 维 wide-and-deep 直连通路 | 与 v2 相同 Teacher 训练集。 | Teacher 保留集 | 动作一致率 76.87%，低于 v2 的 78.5%。 | 训练前淘汰，未做昂贵对局评测。 |
| `reinforce-classic-run1` | 从 `dagger-classic-v2-run1` 出发；3 × 40 局，学习率 0.0003，结算净得分 REINFORCE，对手为 3 个冻结 Teacher。 | 21,100,000 起 100 个牌墙；与 DAgger v2 同墙配对 | 自身均分 **−4.21 ± 1.71**；相对 DAgger v2 为 **+0.59 ± 1.13**，95% CI **[−1.64, 2.81]**。 | 方向不确定，不晋升、不上传；该实现只作为在线训练与配对验收基础。 |

REINFORCE 的训练批次平均净得分从 −15.98 升至 −1.90，但这不是独立强度证据。

| 检查点 | 训练 | 独立评测 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| `actor-critic-classic-run1` | 从 `dagger-classic-v2-run1` 出发；600 局（6 × 100，四座均衡），学习率 0.00015；线性状态价值基线每批拟合 4 次。 | 21,200,000 起 100 个牌墙；与 DAgger v2 同墙配对 | 自身均分 **−4.23 ± 1.59**；相对 DAgger v2 **−0.40 ± 1.12**，95% CI **[−2.60, 1.80]**。 | 价值 MSE 每批下降，但未转化为强度；淘汰，不上传。 |

纯 Python 的线性 critic 已证明流程正确，却不足以带来可验证收益。下一项实验不是继续堆叠
该训练，而是准备带共享表征的 policy-value 网络、固定的按牌墙切分数据清单及冻结对手池；
完成后仍以全新牌墙通过配对区间门槛。

## 2026-08-05：安全轨迹 v3 与 deployment-matched policy-value DAgger

轨迹 v3 将训练观测与复现元数据分离：训练 JSONL 默认不含 deal seed、行为 seed、对手
暗牌或牌墙顺序；可选 replay 索引独立保存且不交给训练器。切分按完整物理牌墙进行，
候选四座轮换共享一个不透明切分组，避免同一墙进入不同分区。

| 数据 / 检查点 | 构造或训练 | 结果 | 结论 |
| --- | --- | --- | --- |
| `trajectory-balanced-classic-v3-run1` | 160 Teacher 局、120 随机合法探索局、80 个 Teacher 选择 `pass` 的物理响应课程、136 个规则验证游金课程。 | 496 条完整轨迹、16,402 决策；`discard` 12,470，`pass` 85，`advance_tour` 136。 | 比纯 Teacher 数据覆盖更好，但稀有杠类仍偏少，不能单独作为强度依据。 |
| `policy-value-classic-v1-run3` | 145 维公开候选特征，128 隐层，共享 policy/value 头；按验证集选择 epoch 33。 | 留出集动作准确率 92.79%；在 21,600,000 起 200 个新牌墙上相对旧 DAgger v2 **+1.87 ± 1.36**，95% CI **[−0.79, +4.53]**。 | 方向为正但区间跨 0，不晋升。 |
| `candidate-teacher-dagger-classic-v1-run1` | 用 run3 在 100 副牌墙的四座轮换中对抗 3 个冻结 Teacher；只保留候选实际决策的 Teacher 标签。 | 400 局、4,366 个候选访问决策；训练/验证/测试各按整副牌墙分组。 | 修正了旧 DAgger 全席位模型自博弈与实际部署分布不一致的问题。 |
| `policy-value-classic-v1-run4-dagger` | 从 run3 微调 20 epoch；加入上述在线训练/验证轨迹；验证集选 epoch 20。 | 混合留出集准确率 94.80%，在线验证 96.33%；在 21,800,000 起 200 个新牌墙上相对旧 DAgger v2 **+3.80 ± 1.23**，95% CI **[+1.40, +6.20]**。 | 已显著优于旧 DAgger v2；但与 run3 在同一组牌墙上差 **+0.08 ± 0.90**，95% CI **[−1.68, +1.83]**，故不把 run4 宣称为比 run3 更强，也不替换网页默认 Teacher。 |

policy-value 网络当前每步以单一候选集合推理，GPU 小批量调度会使长对局评测较慢；下一轮
工程优化应为批量 rollout / 事件序列编码。模型升级仍须先在全新牌墙上胜过当前冻结候选，
并保留相对 Teacher 的独立配对报告。

## 2026-08-05：公开事件序列与 neural PPO 的反证结果

| 检查点 / 实验 | 设置 | 验收结果 | 结论 |
| --- | --- | --- | --- |
| `policy-value-public-sequence-v1-run1` | 两层 64 隐层 Transformer，最近 24 个公开事件；暗杠牌面脱敏。 | 混合留出集准确率 93.82%、测试集 90.10%，均低于 candidate MLP。 | 随机初始化的序列模型在当前数据量下不值得实战评测，淘汰。 |
| `policy-value-public-sequence-residual-v1-run1` | 从 run4 精确复制候选 MLP 路径，事件 Transformer 仅学习零初始化残差；继承参数学习率为残差分支的 10%。 | 离线测试准确率 95.13%，但在 21,900,000 起 200 个新牌墙上相对 run4 **−1.15 ± 0.57**，95% CI **[−2.28, −0.03]**。 | 离线一致率提高却显著伤害实战；淘汰。公开事件顺序仍应保留在数据中，但下次须以更大规模或更强目标训练。 |
| `torch-ppo-classic-v1-run1` | 从 run4 起点；3 × 256 局候选一席 vs 三 Teacher，PPO 裁剪策略目标 + 同网络终局净分 value。 | 8,276 个采样决策；在 22,000,000 起 200 个新牌墙上相对 run4 **+0.09 ± 0.58**，95% CI **[−1.04, +1.22]**。 | PPO 管线与合法动作约束正确，但首轮没有可验证收益；不晋升。 |

因此下一轮不应继续在 16k 级别标签集上堆叠更大网络，优先级为：扩大候选对冻结对手的
在线数据、保存多 checkpoint 对手池、批量化 rollout，并以独立牌墙选择 PPO 迭代而非
依据训练回报或 Teacher 动作准确率选择。

## 2026-08-05：批量 rollout、冻结对手池与 PPO 规模验证

`TorchPolicyValueAgent` 新增变长合法动作集合的批量前向；PPO 同时推进独立牌局，并按
冻结 checkpoint 分组批量执行对手决策。规则执行、动作合法性检查和结算仍逐局进行。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `torch-ppo-classic-v1-run2-pool` | run4 起点；3 × 1,024 局，batch 32；75% Teacher、25% 冻结 run3 对手；33,719 个候选采样决策。 | 22,100,000 起 200 个新牌墙，相对 run4 **−0.08 ± 0.29**，95% CI **[−0.64, +0.49]**。 | 规模与对手池都没有产生可验证增益；不晋升。批量化与对手池保留为下一次更大规模联赛的基础设施。 |

## 2026-08-06：单次反事实动作价值的反证与数据质量门槛

新增采集器在候选实际访问的公开状态中，将每个合法动作强制应用到私有内存分支并结算；
JSONL 只保存本家/公开观察、合法动作和对齐的 `action_values`，不保存牌墙、暗牌或种子。
首轮使用 run4 对三 Teacher、每动作一次确定性 continuation：64 副墙四座轮换共 252 个
决策、1,797 个动作分支，动作价值训练/验证/测试按整副墙分组（192/32/28）。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-v1-run1` | run4 起点；Teacher / DAgger 监督加权 1，单次反事实软偏好权重 8，温度 16；以 32 条反事实验证集选择 epoch。 | 第一组 22,200,000 起 200 墙：相对 run4 **+1.41 ± 1.21**，95% CI **[−0.95, +3.78]**；预先隔离的第二组 22,300,000 起 400 墙：**−1.98 ± 0.87**，95% CI **[−3.68, −0.28]**。 | 第二组显著为负，淘汰、不晋升。单次、同一隐藏世界 continuation 的标签噪声/条件偏差不能靠提高权重解决。 |

由此增加两个强制质量控制：动作价值跨度为零或很小的样本按跨度降权；采集器在
`rollouts_per_action > 1` 时导出每个动作的标准误，训练可按标准误降权。下一轮必须混入
冻结对手池、重复 rollout，并仍以全新牌墙配对评测决定去留。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-v1-run2-mixture` | run4 起点；40 副墙四座轮换，75% Teacher / 25% 冻结 run3；每合法动作 4 个共享对手配置的 rollout，共 160 决策、4,456 分支，动作均值标准误 2.70；跨度和标准误降权，反事实权重 6。 | 22,400,000 起 200 墙：**+1.77 ± 1.12**，95% CI **[−0.43, +3.96]**；预先隔离的 22,500,000 起 400 墙：**−1.20 ± 0.72**，95% CI **[−2.61, +0.22]**。 | 离线反事实测试损失从 run4 的 25.30 降至 6.34，但两组实战均未证明显著提升，第二组点估计为负；不晋升。重复 rollout/对手池降低了噪声，却没有解决“特定隐藏世界”分支目标与部署信息集不一致的问题。 |

随后实现了 `--belief-resample`：对每个 rollout 固定本家手牌、公开河/副露、花数量、金牌和
公开动作，重新分配未知墙、对手暗手、非本家花、对手暗杠牌面；若存在无法由当前先验正确
条件化的他家公开天听状态，安全跳过。真实 Torch smoke 在 8 副墙四座轮换中产生 32 决策、
436 分支、64 个 belief worlds，未导出私有状态。它只验证数据构造和安全边界，尚未训练或
评测，不能作为强度证据。下一步是足量 belief 数据、独立分组切分和相同的实战验收。

| 检查点 / 实验 | 设置 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-belief-v1-run1` | run4 起点；40 副墙四座轮换，公开信息 belief 重采样、75% Teacher / 25% 冻结 run3、每动作 3 个 world，共 160 决策、3,039 分支、480 worlds；温度 24，按价值跨度与标准误（均值 18.93）降权。 | 22,600,000 起 200 墙：**+1.71 ± 0.96**，95% CI **[−0.17, +3.59]**；预先隔离的 22,700,000 起 400 墙：**+0.35 ± 0.65**，95% CI **[−0.92, +1.62]**。 | 不显著，不晋升。相比具体隐藏世界目标没有复现显著负收益，说明信息集重采样是必要的数据修正；但公开先验没有吸收对手历史行为，且数据量仅 160 个状态，尚不足以形成可验证优势。 |

## 2026-08-06：逐动作 Q 头的可辨识性诊断

candidate MLP 新增零初始化的逐合法动作 Q 头，旧 v1/v2 checkpoint 加载时保持 policy 头选牌
不变。训练同时保留 soft preference；Q 目标为 `terminal_score / 80`，报告每动作 Huber/MAE 与
Q-argmax。checkpoint 选择也支持用 Q 的验证 Huber 而非软策略损失，防止训练目标和选择指标错位。

| 检查点 / 实验 | 设置 | 隔离结果 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-q-only-diagnostic-v2` | 从 run4 加载；只使用上表 belief-v1 的 88/52/20 个训练/验证/测试状态，soft preference + 逐动作 Q 回归，按验证 Q Huber 选第 15 epoch。 | 验证 Q Huber 0.1064、Q-argmax 46.15%；测试 Q Huber 0.1249、平均绝对误差 27.29 分、Q-argmax 25.00%。 | 这是头部与指标管线诊断，不是候选策略实验。20 个隔离测试状态太少且 Q 排序接近随机，严禁使用该 Q 头试玩或晋升；下一轮必须扩大独立 belief 状态并混入原 policy 监督。 |

## 2026-08-06：反事实分支批量调度基础设施

collector 新增 `rollout_batch_size`。批量模式只把多个独立的“合法动作 × rollout world”分支的
`scores_batch` 推理合并；每个分支仍保有自己的规则实例、随机数、强制动作和逐步合法性检查。
串行模式（batch=1）保持原实现，作为回归基线。

| 验证 | 设置 | 结果 | 结论 |
| --- | --- | --- | --- |
| 串行—批量等价性单测 | 相同 core 牌墙、相同确定性策略、每动作 2 个 continuation；batch=1 对比 batch=8。 | 所有安全导出的观察、合法动作、chosen index、动作价值、标准误和公开事件完全相同。 | 调度不改变规则推进或目标语义。 |
| 真实 CLI 冒烟 | run4-dagger checkpoint、core、10 墙四座、batch=8、每动作 1 个 continuation；临时数据使用 30/30/40 切分仅为验证三分区输出。 | 40 个决策、321 个分支；manifest 明确 `private_simulator_state_exported=false`。 | 参数、Torch 批量推理和安全导出已跑通；这是基础设施验证，不是模型训练或强度证据。 |
| 实际合并指标 | run4-dagger checkpoint、core、1 墙四座、每动作 4 个 continuation、batch=32；不落盘。 | 160 个分支，116 次 batch 推理、合并 728 个决策，最大单次 32；导出观察未出现 `wall` 或 `opponent_hands`。 | collector 已能在同一决策的多个 rollout world 间实际合并推理；下一步可安全提高 world 数。 |

下一步不直接训练新 checkpoint：先用该调度扩大每状态的 belief worlds，并在缩小规则的 toy profile
上实现、校准顺序粒子过滤，再构造保守分布价值目标。

## 2026-08-06：顺序粒子 belief 的数学与安全原型

新增通用、非序列化的 `SequentialParticleBelief`：它只保存运行时不透明粒子和标准化权重，
每个公开事件由调用方提供粒子 transition 与行为似然；导出层只能读取 ESS、熵、log evidence、
重采样/拒绝计数等聚合诊断。

| 验证 | 设置 | 结果 | 结论 |
| --- | --- | --- | --- |
| 可枚举隐藏类型玩具博弈 | 两个等先验对手风格；观察 `claim` 后 `pass`；与手算贝叶斯后验比较。 | aggressive 后验为精确值 `4/13`；无重采样时权重与枚举一致。 | 数学更新正确。 |
| 粒子退化与失败安全 | 强偏好似然触发 ESS 阈值重采样；另设所有粒子零似然。 | 重采样后保留重采样前 ESS；零总似然抛错且不改写原 belief。 | 不会把崩溃的后验静默伪装成均匀分布。 |

该原型尚未接入厦门 collector，未产生训练数据、模型或强度结论。下一步先构建可回放的本家私有
摸牌轨迹和完整公开事件 replay，再在缩小规则环境中与精确后验校准。

随后 collector 已在**内存**中把本家初始手牌/花、每次私有摸牌（含补花与补杠）和本家动作
绑定到每个候选决策快照；动作记录覆盖公开日志会隐藏牌面的暗杠。定向测试确认庄家的开局摸牌
可定位到首个公开 `draw`，而安全导出的 JSONL/state/source metadata 均没有 trace、墙或他家暗手。
这仍是 replay 前置数据层，不是新的训练数据或模型结果。

### core 公开历史 replay oracle（仍未接入 collector）

在内存中保留开局快照后，新增 core replay：候选自己的出牌/响应取自 private trace；对手按冻结
策略的平滑行为似然处理（若策略提供 action scores 则使用温度 softmax，否则才回退确定性）；每次规则
推进都逐项核对 `draw/discard/chi/pong/ming_kan/an_kan/add_kan/hu/result` 公开事件。固定 `seed=953`
的 13 个候选快照全部重放成功，且庄家后续私有摸牌在回放时被严格核对。新增 audit 仅报告提议/接受
粒子、接受率、平均 log-likelihood 和拒绝类别，不会返回任何暗牌或牌墙。

同时测试“随机未知开局 + 拒绝不一致历史”的朴素 proposal：同一局 5 个公开事件前缀只接受
6/200 个，13 个事件前缀为 0/200。该高拒绝率说明它不是可用的历史 posterior proposal，故没有
修改 `--belief-resample`、没有生成数据、没有训练模型。后续必须采用事件约束的序列 proposal，并先
报告接受率和 ESS，才可进入 collector 实验。

### 最新弃牌局部 SIR：受门槛保护的中间数据构造

先实现最小的可重建事件：response 状态的末尾公开历史必须为同一对手的
`normal draw → discard`。从本家/公开先验 world 逆转该弃牌并按冻结策略行为概率 SIR；神经策略使用
可用 action scores 的温度 softmax，Teacher 使用平滑确定性 fallback。该局部条件化默认关闭，绝不称为
完整历史 posterior。

| 校准（seed 930–939，32 个合格 response 状态，每状态 32 粒子） | 结构一致率 | 平均 ESS 比例 | 最低 ESS 比例 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原始行为似然，power=1 | 1.000 | 0.156 | 0.035 | 权重退化，不可直接采集。 |
| 保守 tempered likelihood，power=0.25 | 1.000 | 0.669 | 0.560 | 可作为 gated 小范围实验的起点。 |

collector 现提供显式 opt-in：32 粒子、power=0.25、最低 ESS 比例 0.5。状态不在受限事件范围、候选
合法动作不一致或 ESS 低于门槛时会跳过；source metadata/summary 只写入配置和聚合 ESS，没有墙、暗手、
粒子或随机种子。端到端 core 导出回归测试已通过；尚未用该数据训练或做强度结论。

经典档增加了显式安全边界：游金、天听和早局 response 不尝试逆转；gold lock 与上一张未被吃碰的弃牌
只从公开 action 序列恢复。seed 930–939 的 34 个合格 classic response 状态中，power=0.25 的平均/最低
ESS 比例为 0.425/0.094；power=0.10 为 0.583/0.094。因此 classic 的初始 gated 实验使用 32 粒子、
power=0.10、最低 ESS 0.20；仍未开始训练或实战评测。

## 2026-08-07：局部行为条件化的独立反证

| 检查点 | 构造或训练 | 独立评测 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-local-belief-v1-run1` | run4 起点；40 墙四座轮换，局部 latest draw→discard SIR（32 粒子、power 0.10、ESS ≥ 0.20）、75% Teacher / 25% 冻结 run3、每合法动作 3 个 belief world。采得 124 个 response 决策、822 分支、372 条条件化 world，平均 ESS 比例 0.960；温度 24，价值跨度和标准误降权。 | 22,800,000 起的 200 墙筛选：**+1.04 ± 0.76**，95% CI **[−0.46, +2.53]**；独立 22,810,000 起的 400 墙：**−0.14 ± 0.59**，95% CI **[−1.29, +1.01]**。 | 没有通过 400 墙正向下界门槛；不晋升。局部后验比公开先验更精确的假设尚未获得实战支持。 |

这批数据的平均每动作标准误为 18.55 分，平均价值跨度为 28.47 分，124 个状态也不足以稳定选择
checkpoint。训练器新增逐动作下置信界软标签 `Q - z × stderr`（`--action-value-confidence-z`）：它和既有
整体标准误降权互补，只在明确开启时对高方差动作保守回缩。下一轮先用固定数据进行 z 的小范围筛选，
再决定是否值得扩大数据；任何离线损失改善都不能替代新的同牌墙实战验收。

| 检查点 | 固定数据消融 | 新牌墙筛选 | 结论 |
| --- | --- | --- | --- |
| `policy-value-action-value-classic-local-belief-lcb-v1-run1` | 上表相同 124 状态、run4 起点，只有 soft preference 改为 `Q - 1.0 × stderr`；验证集按同一反事实来源选择 epoch 20。 | 22,820,000 起 200 墙：**+1.04 ± 0.75**，95% CI **[−0.44, +2.51]**。 | 与均值目标筛选几乎相同、仍不显著；不做 400 墙终检。小数据下标签风险厌恶不是主瓶颈。 |

## 2026-08-07：训练期特权 critic 基础设施（尚无强度结论）

参考 Suphx 的 oracle-guiding / global-reward-prediction 方向，PPO 新增默认关闭的
`--privileged-critic`。它在采样时读取完整模拟状态，**仅作为终局净分 advantage baseline**；actor 仍为
本家/公开特征加规则合法动作掩码。critic 特征只保存在 `PpoStep` 的进程内存：不进入 `TeacherDecision`、
轨迹、JSONL、网页、报告或保存的 actor checkpoint，进程退出即丢弃。单测覆盖 batch rollout、PPO 更新、
合法动作以及 actor observation 不含 `wall`/`opponent_hands`。尚未进行 PPO 训练或实战评测，不能据此
声称增益；下一个对照须固定 actor 起点、对手池和 rollout 预算，比较 advantage 方差后再上新牌墙。

16 局 core CLI 冒烟（run4 actor、8 路 rollout batch、32 隐层 critic、1 PPO epoch）产生 143 个候选
决策，actor checkpoint 中没有 critic 参数，报告没有 `wall`、`opponent_hands` 或 `privileged_features`。
该过程仅验证可执行性和隔离边界，不产生可比较强度数字。

## 2026-08-07：配对动作价值误差（采集器升级）

反事实分支已在同一 belief world、同一冻结对手配置中对所有合法动作执行，故新增对每个动作相对于
均值最优动作的配对 gap 标准误，而不仅是各动作边际 Q 标准误。真实 classic 校准（1 墙四座、4 个
response、每动作 6 world）边际标准误为 **18.58** 分，配对 gap 标准误为 **12.13** 分；这支持在策略
软标签中按配对置信界收缩不可靠的排序。该字段没有保存任何私有 world。当前正在采集按物理牌墙分组的
正式数据；尚未训练或做强度判断。

正式数据由两个互不重叠的墙组组成：188 个 response 决策，训练/验证/测试为 106/53/29；每动作 6 个
共享 belief world，平均结构一致率 65.0%、ESS 比例 74.5%、边际标准误 14.33 分、配对 gap 标准误
9.96 分。`policy-value-action-value-classic-local-belief-v3-paired-gap-run1` 从 run4 微调，使用 pairwise
confidence z=1、反事实权重 1，在全新 22,980,000 起的 200 墙筛选为 **+0.84 ± 0.70**，95% CI
**[−0.53, +2.21]**。未通过筛选，不做 400 墙终检、不晋升。结论是配对估计显著降低标签方差，但这批
状态量和离线 policy 更新仍不足以形成可验证胜率；下一轮转向更大规模、方差更低的 on-policy 训练。

## 2026-08-07：训练期特权 critic 的受控 PPO 反证

`torch-ppo-classic-privileged-critic-v1-run1` 固定 run4 起点、经典档、3 × 1,024 局、32 路 rollout batch、
75% Teacher / 25% 冻结 run3 对手；只开启 128 隐层、损失权重 0.25 的训练期特权 critic。共产生 33,688 个
候选决策。三个迭代的 advantage 标准差依次为 0.591、0.578、0.555，critic Huber loss 为 0.164、0.161、
0.146；这是同一 run 内的优化诊断，旧 PPO 报告没有该方差字段，**不能**据此声称相对无 critic 的因果降方差。
三份 actor checkpoint 均不含 critic 参数或 `wall`、`opponent_hands`、`privileged_features` 字段。

在同一组全新 23,000,000 起的 100 个物理牌墙筛选中，相对 run4 的配对净分为：iteration 1 **+0.545 ±
0.583**，95% CI **[−0.597, +1.687]**；iteration 2 **−1.285 ± 0.834**，95% CI **[−2.921, +0.351]**；
iteration 3 **−0.223 ± 0.740**，95% CI **[−1.673, +1.228]**。没有一个 iteration 通过正向下界门槛，故不做
400 墙终检、不晋升。结论是 privacy boundary 和训练管线成立，但当前 critic 配置及 3k 局 PPO 没有给出可验证
强度增益；下一步应先补齐固定策略、同一 rollout 的 critic-on/off 方差 A/B，再决定是否继续投入该路线。

该 A/B 现已完成：用 run4 actor、相同的 run3/Teacher 对手池，先以 512 局独立墙校准 128 隐层 critic，再在
另 512 局（5,704 个候选决策）上以完全相同的 actor 轨迹比较基线。轨迹逐条一致；公开 value 的残差标准差/
均方为 **0.513 / 0.263**，critic 为 **0.591 / 0.350**，残差均方变化为 **−33.0%**（负数表示劣化）。这直接
否定“当前 critic 已降低 PPO 方差”的假设：停止该配置的进一步 self-play，不生成或晋升任何模型。新增
`scripts/measure_ppo_baseline_variance.py` 固化该检验，且仅写入聚合残差；critic、墙与暗手永不落盘。

## 2026-08-07：公开 Q 头校准门槛

为验证是否值得把动作价值接入 PPO，先冻结 policy 偏好（`--action-value-weight 0`）并只训练可部署的
`action_value_head`。过程中发现并修复训练器耦合：此前 Q 回归错误地复用了 policy source weight，因而
`--action-value-weight 0` 也把 Q 标签全置零；现新增独立的 `--action-value-regression-sample-weight`，并有
单测保证 Q-only 校准确实更新 Q 头而不依赖 policy 偏好权重。

修复后的 `policy-value-public-q-calibration-v1-run2` 在同一 188 个、按物理牌墙隔离的 local-belief paired
数据上选择 epoch 31。相对零初始化 Q 头，隔离测试的 Q Huber/MAE 从 **0.122 / 30.85 分**降至
**0.053 / 22.76 分**；但最关键的动作 argmax 排序从 **55.2%** 降至 **48.3%**（仅 29 个状态）。这说明小数据
足以拟合平均数值，却不足以可靠决定吃/碰/过；不接入 PPO、VRPO 或 Q 选牌，不做实战评测，也不晋升。后续
动作价值路线必须先大幅提高按墙隔离的有效状态数与排序精度，而不能只降低 MAE。

为扩大排序检验，新增独立 `local-belief-v4-scale-a`：120 副物理牌墙、356 个 response 决策、每动作 6 个
共享 belief world。结构一致率 57.6%（预设下限 55%）、ESS 96.1%、配对误差 10.56 分；质量没有改善，故只作为
独立加量数据，且不与任何已有测试墙重叠。加入后 Q-only 校准的验证/测试状态增至 111/103。隔离测试上，公开 Q
的 MAE 从零初始化的 **29.79** 分降至 **19.85** 分，argmax 从 **41.7%** 到 **50.5%**；但逐状态配对准确率差
仅 **+8.7** 个百分点，95% CI **[−7.5, +25.0]**，仍跨零。不接入 Q 选牌、PPO 或 VRPO；继续增加独立墙组，直到
排序下界通过或该路线被反证。

第二独立墙组 `local-belief-v4-scale-b` 追加 369 个 response 状态（一致率 62.9%、ESS 96.0%、配对误差
10.80 分）。联合五组墙后，公开 Q 在 146 个隔离测试状态上相对零初始化从 **39.0%** 到 **52.7%**，MAE 从
**29.96** 到 **18.92** 分；配对排序差 **+13.7pp**，95% CI **[+0.6, +26.8]**，首次满足离线 Q 门槛。因此只为
已受监督的 response 阶段实现 `response_action_value`：摸牌后仍严格使用 policy，避免将未训练 Q 用于弃牌。

但全新 23,300,000 起的 200 墙四座轮换筛选相对 run4 为 **−1.88 ± 1.28**，95% CI **[−4.39, +0.63]**。不做
400 墙终检、不晋升。结论不是 Q 头无法拟合 local-SIR 标签，而是这类局部 posterior 反事实排序仍未能外推为
完整对局强度；停止继续扩大这条 local-SIR/Q 数据路线，不能将离线排序当作网页 AI 改进证据。

## 2026-08-07：公开后状态 outcome 数据契约（诊断阶段）

对福州麻将工程的只读调研带来一个可迁移但须重新验证的方向：其文档把成功候选描述为
`policy prior + lookahead / rule + expected score + win probability - risk` 的**保守混合**；同时明确把单隐藏
世界的动作反事实 Q 与原始 PPO 视为不可靠路线。福州与厦门的规则、计番和可见信息并不相同，因而没有复制
权重、规则代码或强度结论；这里只采用其数据纪律：先把实际执行动作的长程结果记录清楚，再讨论混合选牌。

为此 `TeacherDecision` 新增可选 `executed_index`。`chosen_index` 始终是 Teacher 的监督建议，
`executed_index` 则是生成该轨迹的行为策略真正执行的合法动作；旧语料保持 `null` 并可正常读取。Teacher、随机
探索和 DAgger collector 均已写入这一字段；JSONL 仍不含牌墙、他家暗手或随机种子。这样终局得分才只监督
它所属的那个动作，绝不把同一条隐藏 continuation 伪装成所有合法动作的 Q 标签。

首轮数据为 run4 对三名冻结 Teacher 的 120 个物理牌墙、四座轮换：480 局、5,282 个候选决策，按墙组隔离为
3,432/919/931 个 train/validation/test 决策。约 95% 行为动作与 Teacher 建议相同（训练/验证/测试的不同动作数
为 169/43/51），因此这不是足以支持全候选 argmax 的探索数据，而是先校验长程标签是否可学习的 on-policy
数据。`policy-value-afterstate-outcome-classic-v1-run1` 从冻结 run4 编码器训练独立的 score、己方获胜和对手获胜
头；它没有改动 policy logits，报告状态固定为 `diagnostic_only_not_authorized_for_action_selection`。

| 独立集 | 终局净分 MAE | 零预测 MAE | 己方 / 对手胜率 Brier | 实际己方 / 预测己方胜率 | 实际对手 / 预测对手胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation（919） | 27.25 分 | 32.09 分 | 0.154 / 0.180 | 24.7% / 25.8% | 70.4% / 74.2% |
| test（931） | 34.78 分 | 39.66 分 | 0.172 / 0.172 | 28.4% / 27.8% | 71.6% / 72.2% |

这只证明公开特征可预测**已执行动作**在固定后续策略下的软终局结果；它没有观察未执行动作的结果，不能授权
afterstate head、Q head 或网页 agent 对全体候选直接取 argmax。下一阶段必须先以已知 propensity 的温和随机
行为混合增加动作覆盖，并在按墙隔离的动作集上做 propensity/支持度审计；只有 score 与两类概率继续通过校准、
且保守 policy-prior 混合在新牌墙的 200 墙筛选为正，才允许进行 400 墙终检。当时冻结的 run4 仅作为
历史实验起点；网页默认始终是 Teacher。

### 单动作随机干预：使终局标签匹配目标策略后缀

持续 epsilon 行为会让某个动作之后的整局策略也偏离 run4，因此不能为 run4 的动作价值提供干净标签。collector
现改为每一局候选座位只随机选一个决策序号：在该点按 `0.4 × uniform + 0.6 × run4` 选合法动作，此前与此后均
严格执行 run4；每条记录写入实际 `executed_probability`，而随机种子仍从训练 JSONL 剔除。由此，干预点的
终局结果可解释为“已执行动作 + run4 后续”，而不是局部 SIR 的隐藏世界分支。

`afterstate-intervention-classic-v1-run1` 使用 240 个物理牌墙四座轮换（960 局、10,745 个候选决策）、
按墙组隔离为 7,338/1,779/1,628 个一般决策；随机干预点为 **419/95/90**，最小已知 propensity 为约 **0.025**。
训练只取 `executed_probability < 1` 的干预点，并冻结 run4 的 encoder 和 policy head。

| 检查点 | 训练／验证／测试干预点 | 独立测试终局净分 MAE | 零预测 MAE | 己方/对手 Brier | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `policy-value-afterstate-intervention-classic-v1-run1` | 419 / 95 / 90 | **32.38** 分 | 40.47 分 | 0.195 / 0.195 | 标签构造和公开预测均通过最小校准检查，但样本小、尚无未执行候选排序或实战证据；保持 diagnostic-only。 |

该结果与之前的 local-SIR Q 不同：它没有把一个隐藏 continuation 共享给所有动作，并且干预后立即回到目标
策略。不过这仍是稀疏 logged-bandit 数据，不能凭 head 对未执行动作的外推直接 argmax。下一步是扩大按墙隔离的
单点干预覆盖、按 action kind／概率分桶审计支持度，并仅在 policy-prior 锚定的小幅混合通过新的 200 墙筛选后，
才有资格进行 400 墙终检。

### response 定向干预：补齐吃／碰／过覆盖

首轮全阶段干预的 90 个测试点有 64 个弃牌，response 覆盖不足。因此 collector 增加
`--intervention-phase response`：只在本家 response 决策中计数并选择唯一干预点，弃牌和后续 action 都保持
冻结 run4。专项数据用另一组 240 个物理牌墙（960 局），获得 train/validation/test **437/56/116** 个干预；
全部在 response。测试动作组成是吃 54、碰 34、过 23、胡 2、明杠 3，最小 propensity 为 0.133。

`policy-value-afterstate-response-classic-v1-run1` 仍冻结 run4 policy、只训练 outcome head，且仅在
response 干预数据上选择 epoch。测试的终局净分 MAE 为 **30.84** 分，优于零预测 **34.28** 分；己方／对手胜率
Brier 为 **0.182 / 0.183**（测试常数基线为约 0.188）。这说明公开 afterstate 特征在吃／碰／过的目标策略后缀
中有可学习信号。

但 56 个 validation、116 个 test 干预仍不足以校准融合系数、量化候选间排序错误或支持任何 response argmax。
它不替代此前 response-only Q 的失败结论，也不进入 200 墙实战；下一步先训练多 seed outcome ensemble、报告每个
候选的均值/离散度和 policy-prior 偏移率，再预注册一个小幅保守混合及其独立 200 墙筛选。

### response outcome ensemble：校准通过、实战否决

为避免 outcome 训练改变已验证的 run4 policy，网络升级为 v5：新增独立的 `afterstate_encoder`，它从 run4
candidate encoder 初始化，但默认训练时只有它和 score／己方胜率／对手胜率三头可更新；policy encoder、policy
logits、Q 和 value 均冻结。v1 的 240 墙与另一组 v2 的 240 墙 response 单点随机干预按原有物理墙分区追加，得到
**844 / 148 / 220** 个 train / validation / test 干预点。五个独立随机种子训练为 ensemble；每个 checkpoint 都
保留 `diagnostic_only_not_authorized_for_action_selection` 状态。

联合独立测试集上，ensemble 对已执行动作的 score MAE 为 **28.92** 分（零预测 **32.94**），己方／对手 Brier 为
**0.165 / 0.172**；平均 score ensemble 标准差为 **2.77** 分。它说明第二批数据提高了 *logged-action* outcome
校准，却不表示未执行候选的排序正确。审计还显示：在 validation 的 run4 policy top-2 候选内，score LCB 会改动
**36.5%** 的 response 决策，足以构成可检验、但仍受 policy-prior 限制的干预。

预注册候选只在 response 阶段使用该五模型的 `mean(score) - 1.0 × std(score)` 重排 run4 policy top-2；所有
摸牌／弃牌仍逐位复用冻结 run4 policy。它在从训练、验证、测试均隔离的 **23,400,000** 起 200 个物理牌墙、四座
轮换中相对 run4 的配对净分为 **−3.821 ± 1.319**，95% CI **[−6.406, −1.236]**。候选胡率为 21.75%，run4 为
25.25%。负向下界明确，故**不进行 400 墙复核、不晋升、不接入网页**。

结论：执行动作的终局校准不是 response 反事实排序的充分条件；有限策略先验集合和 ensemble LCB 也不能弥补该
数据识别缺口。停止继续扩展这一 selector 或调融合系数。下一轮改为构造规模化、同一环境样本中逐合法 response
动作分支到结算的直接 Monte-Carlo Q 数据；私有状态只留在 collector 内存，导出／输入仍严格限于本家手牌和公开
信息。它将先以小型按墙隔离 pilot 验证覆盖、方差和离线排序，再决定是否值得训练或实战筛选。

## 2026-08-07：direct-response Q pilot 与冻结路径门槛

counterfactual collector 新增显式 `--decision-phase response`；与旧的混合抽样不同，它只导出 response 决策，
没有 response 的牌局直接跳过。每个选中的公开／本家私有信息集在 collector 内存中克隆为所有合法动作的分支，并
在冻结 run4 后缀与 Teacher 对手下结算。分支所用的暗手和牌墙既不进入 action feature，也不进入 JSONL 或 manifest；
跨大量牌墙时，它们是信息集条件终局分布的 Monte-Carlo 样本，而非部署时可见输入。

60 个全新 classic 物理牌墙的 pilot（每座最多两个 response，单分支）得到 **416** 个决策／**920** 条分支，
动作为 pass 255、chi 84、pong 73、hu 3、ming_kan 1，平均动作回报跨度 **34.40** 分。按墙隔离为
267 / 64 / 85 train / validation / test，所有 416 个导出决策均为 response，证明定向采集及边界检查可用；但单个
hidden world 的标签方差仍需要靠扩大墙组而非把同一 world 重复当作独立 rollout。

为保证 Q-only 诊断不影响当前 best policy，网络升级为 v6：Q head 使用独立的 `action_value_encoder`，从 run4
candidate encoder 初始化；`--freeze-policy-path-for-q-only` 进一步冻结所有非 Q 参数，避免即使损失系数为零时
AdamW 的 weight decay 仍轻微移动共享 policy 路径。回归测试和实际 pilot 均确认 run4 与训练后 checkpoint 的
policy logits 最大绝对差为 **0.0**；v1–v5 checkpoint 仍可加载并以 candidate encoder 初始化缺失的独立编码器。

pilot 的冻结 Q 在 85 个独立测试决策上 MAE 为 **32.32** 分（零 Q 36.30 分），但 MAE 不是选择证据。修正
“多个动作同为最优”后的 tie-aware 指标中，Q 最优动作率 60.0%，run4 policy 为 62.35%，29.4% 目标有最优并列。
`audit_response_q_policy.py` 给出的配对 Q−policy 最优率差为 **−2.35pp**，95% CI **[−15.87pp, +11.16pp]**；
policy−Q 后悔改善为 **+1.74** 分，95% CI **[−7.09, +10.57]**。两项都不通过，故 pilot 不进入真实 200 墙评测、
不晋升。

后续 direct-response Q 只有同时满足以下离线门槛才允许真实筛选：policy identity 与 run4 完全一致；按未见物理
牌墙的 Q−policy 最优率差和 policy−Q 后悔改善的 95% 下界都大于零；随后仍须在全新 200 墙、四座轮换中通过配对
净分正下界，才有资格做 400 墙复核。下一步只扩大互不重叠的 direct-response 墙组，不改 selector 或网页默认 AI。

### direct-response Q scale-a：三种排序目标的共同反证

独立 scale-a 使用 240 个全新 classic 物理牌墙，收集 **1,666** 个 response 决策与 **3,699** 条逐合法动作分支；
按墙隔离为 1,265 / 202 / 199 train / validation / test。未见测试包含 pass 131、chi 37、pong 27、hu 2、
ming_kan 2，平均动作回报跨度 26.53 分。所有 rollout 均为同一信息集的动作分支，且未导出牌墙或他家暗手。

先以四个独立 seed 的绝对 Q ensemble 在 run4 policy top-2 内重排：测试平均后悔改善 **+2.23** 分，但 95% CI
**[−1.78, +6.23]**；最优动作率差 **−1.51pp**，95% CI **[−8.13pp, +5.11pp]**。随后只用 validation 墙组尝试
0／1／2／4／8／12／16 分 Q 优势阈值；没有任何阈值使最优率和后悔改善的下界同时为正，故不对 test 做阈值挑选。

为检验绝对终局分数的共同噪声是否是瓶颈，训练器又加入按信息集中心化的 Q advantage regression，以及独立 Q
listwise soft-ranking loss；两者均使用 v6 独立 Q encoder/head，并启用 `--freeze-policy-path-for-q-only`。中心化模型
在 top-2 中只覆盖 15.1% 决策，测试最优率持平、后悔改善 **+0.21** 分，95% CI **[−1.53, +1.96]**；validation
阈值同样没有正下界。listwise 单模型虽有正均值（最优率 +1.01pp、后悔 +2.67 分），但四 seed ensemble 回落为最优率
**−1.51pp**，95% CI **[−8.83pp, +5.81pp]**，后悔改善 **+0.77** 分，95% CI **[−3.34, +4.88]**。

结论：在当前 run4 访问分布、完整隐藏 world 的单样本分支与约 1.7k response 信息集下，direct-response Q 没有可靠
超过冻结 policy 的排序证据。停止继续扩大该数据或调 Q loss；不做 200/400 墙、不晋升。下一阶段回到更根本的信息
集问题：先实现可验证、全公开历史条件化的 sequential belief proposal（必须先报告可接受率、ESS、与 toy exact
posterior 的校准），再考虑用其多 world 目标重新构造长程行动价值。网页默认维持 Teacher；run4 保留为历史
实验 checkpoint，不能据此宣称强度。

### 全公开历史约束修复 proposal：接受率恢复，但权重 ESS 淘汰

为区分“随机暗手下公开行动不合法”的 rejection 问题和行为似然本身的问题，新增 core 专用、audit-only 的
`core_public_history_constraint_repair_v0`。当他家的已观察弃牌/吃/碰/明杠缺少所需暗牌时，它只在未知墙和
非本家暗手间做物理牌守恒交换；不触碰本家手牌、私有摸牌 reservation、公开副露或导出数据，也没有接入 collector。

固定 seed 953 的 9-event 单元夹具，100 个 proposal 中朴素重放接受 **0** 个；修复后接受 **94** 个，平均
3.38 次交换。但以冻结 Teacher 行为概率加权的 ESS 仅 **3.55/94（3.8%）**，长前缀探针约 1%。此外交换 proposal
的密度尚未显式计算，不能把当前权重当作严格 posterior 修正。故该方案只证明错误来源，**不通过** belief 数据、
Q 训练或实战评测门槛；网页默认 Teacher 不变。下一步应先在可枚举小牌墙上实现带 proposal-density 的逐事件
条件化，再决定是否重启多 world 行动价值采集。

### 重要性修正的微型物理牌墙校准

为给下一种 proposal 建立不可绕过的数学门槛，`SequentialParticleBelief` 增加运行时 `initial_weights`，
用于输入 `p/q` 的显式重要性修正。新单测枚举含重复牌的微型牌墙：未知 `0,0,1,1`、对手两张暗手、观察弃 1；
真实后验中 `P(01|弃1)=16/19`。把只生成合法暗手的刻意偏置 proposal（`q(01)=3/4,q(11)=1/4`）连同每粒子
`p/q` 输入后，过滤器精确恢复 `16/19`。这仅是 proposal-density API 的校准，不是厦门麻将 belief 或训练样本；
真实全历史方案必须先满足同类枚举测试，才有资格替代当前 audit-only repair。

对同一 core seed 953、9 个公开 action、200 个固定 proposal 粒子的复核显示，单靠放宽确定性
Teacher fallback 的均匀平滑也不能挽救全历史权重：`uniform_mixture=0.02/0.10/0.25/0.50` 时，接受数均为
192，行为权重 ESS 比例仅为 **1.09% / 1.33% / 1.94% / 4.16%**。故不能用“更大 uniform noise”伪装出
健康 posterior；下一代方案需要可校准的行为能量模型与显式 proposal density，两者缺一不可。

为排除“整段 rejection 没有逐步重采样”的解释，又实现了 `core_public_history_sequential_smc_v0`：setup prior
保持 `p/q=1`、不交换暗牌，公开日志按玩家动作与其自动 draw/result 分组；每组按增量行为似然更新、低 ESS
时 systematic resample。seed 953 的 9-event fixture 中，**200** 粒子在完成 4 个公开 event 后耗尽；**1024**
粒子走完，但最小预重采样 ESS 为 **1/1024**，重采样后最终均匀权重不能掩盖该退化。该严格基线同样不通过
collector/Q 数据门槛；其作用是将下一步明确限定为“可计算密度且经小牌墙 posterior 校准的条件 proposal”。

### 精确 setup-claim density 的最窄正向验证（不进入训练）

新增 `core_initial_response_claim_exact_density_v0`：仅限 core 开局中“本家弃牌后另一家立即吃／碰／明杠”。
该家尚未摸牌，因此可对其初始暗手以多元超几何分布直接抽样，条件为包含公开副露消耗的牌。组合动态规划精确
给出条件概率，完整 setup 的修正为 `p/q=P(初始暗手包含所需牌)`；不是通过手工交换暗牌伪造一个 world。

固定 `seed=2` 的首吃前缀、256 粒子审计：普通 setup prior 重放为 **40/256** 接受；该 exact-density proposal
为 **256/256** 接受，`p/q=0.13146149`，加冻结 Teacher 平滑行为似然后的 ESS 是 **190.04/256（74.2%）**。
数学 micro-deck 单测也验证 `{0,0,1,1}` 抽 2 张且要求含 `1` 的条件概率为 `5/6`，条件后 `11` 的概率为 `1/5`。
这只是一个极窄 setup 事件的**构造正确性**验证：后续摸牌、弃牌、暗杠与多个事件尚未有 density，故不改变 run4、
不接入 collector/Q/网页，也不能解释为完整 history posterior 或模型强度提升。

为扩展 draw→discard，普通摸牌的 `public_actions` 已新增 `draw.tiles`：它按顺序记录该次摸牌公开补到的花牌，
不记录随后拿到的可打牌。重放先校验对手最终花数能由这些事件解释；旧记录或补杠／明杠后的未定位 replacement draw
仍拒绝为 `opponent_flower_history_unsupported`。小牌墙先验证 wall-only 条件 proposal：固定公开花序列、依剩余物理
张数抽取隐藏底牌，`p/q` 等于该观察的解析先验概率；其底牌条件分布也有独立回归测试。随后已将未知对手 setup
hand／flower-slot、花序列与公开弃牌的结构可行性联立；微型枚举验证联合概率。固定 core、dealer 0、seed 271、首弃
后下家补 `(40,)` 并弃牌的 256 粒子审计达到 **256/256** 结构接受，reference `p/q=0.0012275911710061692`，但
冻结 Teacher 行为权重 ESS 仅 **17.76/256（6.9%）**，不通过数据门槛。

随后开局翻金 transition 已补齐：将引擎骰位环形扫描／跳花规则抽为共享 `gold_indicator_index`，固定墙的
`P(indicator face)` 条件 sampler 与小牌墙完整枚举一致；“翻金 + 庄家已知首摸”还验证了联合结构概率。两个 core
重建夹具（seed 271、seed 32，其中后者首摸补花）均保持本家手/花、公开金指示牌/骰子、牌墙长度与 144 张物理牌
多重集。随后 opening 与首位对手 draw→discard（含公开弃牌可行性）合并为层次 proposal；标号微型牌墙的枚举先验
为 **1/450**，proposal 的平均 per-particle `p/q` 与之相符。真实 core seed 271、256 粒子审计已达 **256/256**
结构接受，但 `p/q` 随隐藏分配为 `2.5967952709322814e-07` 到 `1.0387181083729125e-06`，Teacher 行为加权 ESS 仅
**9.55/256（3.7%）**，比旧 reference 更低。因此继续拒绝 collector/Q/网页；普通重放仍以
`opponent_flower_transition_unsupported` 安全拒绝。下一步先校准行为条件 proposal，而不是继续延长该前缀。

## 2026-08-07：人类对局评测与数据采集入口（默认关闭）

“胜过人类”不能由 Teacher 配对分数替代。网页现在支持显式 `--human-log local_human_data/*.jsonl`：只在
玩家完成一局且至少行动一次后，追加玩家可见状态、合法动作、实际人类选择、公开 action 历史和本局分差。
牌墙顺序、AI 暗手、随机种子、账号、网络标识和时间戳都不写入；本地目录被 Git 忽略。记录来源固定为
`local_human_opt_in`，默认不进入 Teacher/DAgger 训练，须单独质量审查与按完整牌局留出评测。显式 checkpoint
对手只以文件 SHA-256 记录，不保存本地路径；`audit_human_trajectories.py` 会在人工评审前拒绝混合规则/对手、
重复手牌、重放种子、私有字段和非本局分差记录。

审计报告另对**结构合格**的完整局计算人类 seat 的本局平均分差、样本标准误、正态近似 95% 区间、胡率和
流局率。候选对手的 `agent_profiles` 也明确记为 `explicit_policy_value_checkpoint`，不会伪装成 Teacher；对手
的可比身份仍以 SHA-256 为准。这只是未来真人 A/B 的描述性汇总，既不检验记录者水平，也不构成“模型胜过
人类”的结论。

为进行受控试玩，`serve_web_game.py --ai-checkpoint <policy-value.pt>` 会将三名 AI 显式替换为该 checkpoint，
网页标识为 `EXPLICIT EXPERIMENTAL CHECKPOINT`；没有参数时仍是 Teacher。显式 checkpoint 已通过“加载、响应、
人类一手、AI 自动推进”的规则引擎冒烟检查。该路径仅提供未来人类 A/B 与数据采集能力，不构成任何 checkpoint
战胜人类的结论。

## 2026-08-07：信息集 rollout 规则基线（初筛未通过）

新增 `InformationSetRolloutAgent` 作为与训练数据无关的在线诊断：对当前行动者的每个合法动作，从本家可见
事实重采样暗手/牌墙，在同一 world 内强制动作、其后回退 Teacher 结算；live game 的真实墙与对手手牌不读且
不修改。单测确认动作合法、live hidden state 不变，以及只替换 live 未见牌不会改变固定 RNG 下的选择。

全回合 mode 每副物理墙四座约需 20 秒，且首个单墙净分为 **−14.75**，不具可用性。收缩到默认的 response-only
mode 后，1 belief world 在预注册 10 墙、经典换座筛选相对 Teacher 为 **−1.775 ± 6.692**（95% CI
**[−14.892, +11.342]**）；2 worlds 在另一 5 墙筛选为 **−9.30 ± 5.704**（**[−20.480, +1.880]**）。均无正向
证据，故不接入网页、不开更大筛选，也不产生训练数据。该组件仅保留为日后有经校准 belief/value 时的安全
信息集搜索基线。

## 2026-08-07：独立回归评测——淘汰 run4，冻结 Teacher 默认

为检验“DAgger 混合验证集更高一致率”是否真的提高实战强度，使用此前未参与训练或 checkpoint 选择的
经典档 **80** 个物理牌墙（seed `202608081` 起），每墙四座轮换；推理固定为 CUDA、policy head。评测 CLI
现在同时记录 checkpoint SHA-256，避免同名参数文件被替换后让历史结果失去身份。

| 候选 | 对三名 Teacher 的均分（每局） | 胡率 | 结论 |
| --- | ---: | ---: | --- |
| run3 | **+1.3375 ± 1.3770**，95% CI **[−1.3615, +4.0365]** | 26.25% | 方向为正，但区间跨 0；只可作显式实验候选。 |
| run4-dagger | **−2.3813 ± 1.1274**，95% CI **[−4.5911, −0.1714]** | 22.19% | 相对 Teacher 已显著为负，不可试玩默认或再作为训练起点。 |

同一批牌墙的配对 run4 − run3 为 **−3.7188 ± 1.5691**，95% CI **[−6.7942, −0.6433]**，因此 run4 的退化不是
牌墙难度差异。可复核评测产物为 `artifacts/evaluation-run3-vs-teacher-independent-202608081.json` 和
`artifacts/evaluation-run4-vs-teacher-independent-202608081.json`；两个参数 SHA-256 分别为
`01616a38ff34af71f1774995d5d3f9e08c4408fd85757ad563d9dfa339b19f70` 与
`e8a23464717762421ff6616c95d4b290188864d8d0363a0dac19f80500e39909`。

同日对 `InformationSetRolloutAgent` 做了另一组经典 10 墙 response-only 诊断（1 个 belief world、seed
`202608071`）：**−4.975 ± 4.862**，95% CI **[−14.504, +4.554]**。这与先前的初筛一样没有正向证据；不扩大
搜索、不让它产生训练标签。下一轮优化应重新构造可验证的强度来源（经审计的人类数据、或通过联赛门槛的
自博弈），而不是继续在 Teacher/DAgger 轨迹上微调。

### paired-LCB 信息集搜索：控制选择偏差后仍未通过筛选

原始 rollout 对每个动作取少量 determinization 的最高均值，存在典型的“在噪声中选最大值”偏差。为使这条
规则搜索基线本身更保守，`InformationSetRolloutAgent` 新增实验性 `selection_mode="paired_lcb"`：每个候选
和 Teacher 回退动作必须在**同一批** world 中结算，只在候选的 paired mean delta − `z × stderr`（本轮 `z=1`）
严格为正时才覆盖 Teacher；不足两个 world 时必定回退。该模式不改变既有默认 `mean`，也没有接入网页。

经典档 response-only、2 个 belief world 的预先隔离筛选使用 seed `202608121`–`202608160` 共 **40** 个物理
牌墙、四座轮换，分为八个 5 墙可复核分片执行。候选对三名 Teacher 的均分为 **−3.5688 ± 2.3135** 分/局，
95% CI **[−8.1032, +0.9657]**。下界不为正，且点估计为负，故淘汰：不扩大至终检、不生成训练标签、不替换
Teacher。保留的只是经过单元测试的保守选择机制，日后只有在校准 belief/value 能降低 world 方差时才可重审。

### Teacher 单参数探针：没有实际决策覆盖，不进入代码

为排除把静态启发式系数误当作优化，三类临时 probe 都直接以四座轮换对抗默认 Teacher：弃金惩罚
`0/14/21`、吃碰允许的 hand-quality 损失 `0/1/2`，各在同一经典 10 墙中均得到严格 **0** 总分差；听牌张数
权重 `9/27/36` 也在 10 墙中严格为 0，权重 36 在另一 40 墙仍严格为 0。检查初始 300 个随机局面，只有把
权重夸大到 0 或 1000 才出现 1 个不同弃牌。结论是这些邻域参数并不覆盖 Teacher 的实际排序边界，临时参数化
已撤回；后续只研究会产生足够行为覆盖、并能独立评测的状态/价值改进。

## 2026-08-07：固定对手阵容联赛基线（只作筛选，不晋升）

单独“对三名 Teacher”的结果不能检验候选是否只适应了 Teacher。评测器现接受恰好三名命名对手；对每个物理
牌墙，候选轮换四座，三名对手被赋到其余绝对座位，因此每名对手会覆盖候选周围的三个相对位置。配对比较额外
拒绝不同的对手标签，CLI 会记录所有 checkpoint 的 SHA-256。

首个经典档筛选阵容为 `[run3, run4-dagger, Teacher]`，seed `202608271` 起 **20** 个物理牌墙、四座轮换、
CUDA policy head。绝对均分依次是：Teacher **+0.6875 ± 2.2509**、run3 **−0.4000 ± 1.8075**、run4
**−1.7250 ± 2.2914** 分/局。相对同墙 Teacher，run3 为 **−1.0875 ± 2.8592**（95% CI
**[−6.6914, +4.5164]**），run4 为 **−2.4125 ± 3.1959**（**[−8.6765, +3.8515]**）。

这只是 20 墙筛选，所有区间均跨 0，既不建立模型强度排序，也不授权更多 Teacher/DAgger 微调。可复核产物为
`artifacts/evaluation-league-{teacher,run3,run4}-run3-run4-screen-202608271.json`。今后 PPO 或其他候选须先
在 Teacher 基准和固定异质阵容两者中通过筛选，才允许进入 200/400 墙终检；这仍不等同于胜过人类。

## 2026-08-07：当前策略快照 PPO（小规模消融：拒绝）

参考四人不完全信息牌类的近期自博弈工作，PPO 增加“本轮更新前 actor 的冻结快照”对手。快照在每轮采样前
复制，不共享参数或梯度，更新后才刷新；所有对手抽样、源代码 revision 和各类型计数写入 report。该消融从
run3（SHA-256 `01616a38…b19f70`）出发，连续三个独立 stage，每个经典档 128 局、32 路 batch、默认 PPO
超参数，Teacher/current-snapshot 概率各 0.5。三段共 **384** 局、**4,273** 个候选决策；各段的实际对手数为
Teacher/current snapshot `178/206`、`190/194`、`196/188`。训练时的回报不作为强度指标。

冻结 stage3 参数（SHA-256 `c39d3a63f9ab7b6c4f3220161162261dd82df68e5697ee5711a0c1fddcc47d23`）在全新
Teacher 筛选的两段 40 墙（合计 **80** 墙）为 **+0.7406 ± 1.7495** 分/局，95% CI **[−2.6885, +4.1698]**；
同墙 run3 为 **−0.5031 ± 1.5810**。候选 − run3 的配对差为 **+1.2438 ± 1.4029**，95% CI
**[−1.5060, +3.9935]**，不通过第一道正向下界。

固定 `[run3, run4-dagger, Teacher]` 阵容的另一组全新 20 墙筛选中，候选绝对均分 **+1.1375 ± 2.2310**；
相对 Teacher 为 **+0.8375 ± 2.9100**（95% CI **[−4.8661, +6.5411]**），相对 run3 为
**−1.1875 ± 1.7310**（**[−4.5802, +2.2052]**）。双门槛均不通过，故不做 200/400 墙终检、不晋升、不作为
下一轮训练起点或网页 AI。产物保留在 `artifacts/torch-ppo-classic-current-selfplay-v1-stage{1,2,3}/` 及
`artifacts/evaluation-{torch-ppo-classic-current-selfplay-v1,run3,league-*-current-selfplay}-*.json`，用于复核而非部署。

## 2026-08-07：公开信息一步前瞻规则候选（初筛拒绝）

为测试“静态弃牌后牌形”是否是 Teacher 的主要局限，新增 `OnePlyLookaheadTeacherAgent`。它只看本家手牌、
四家河/副露和翻金：对每个弃牌枚举下一次可能的可打牌，并按公开可见后的剩余张数加权；若下一摸成和，则用与
引擎同尺度的自摸结算值，否则取下一步合法弃牌的廉价牌形值。单测通过“交换对手暗手与牌墙中的未见牌，动作
不变”验证信息边界；它不读取墙或任何对手暗手，也没有接入网页或数据采集。

经典档 seed `202608401` 起的 **20** 个独立物理牌墙（80 个四座换位对局）已足够否定该启发式：一步前瞻候选
对三名 Teacher 的均分为 **−21.4000 ± 2.5405**，同墙相对 Teacher 为 **−21.4000 ± 2.5405**，95% CI
**[−26.3794, −16.4206]**，胡率仅 2.5%。这是强负结果，不以调权重挽救：不扩大终检、不生成训练标签、不作为
任何模型的 Teacher、不替换网页默认。完整逐局分数和配对统计见
`artifacts/evaluation-one-ply-lookahead-teacher-pilot-202608401.json`；CLI 仅保留
`--one-ply-lookahead-teacher-candidate` 以便复核。

## 2026-08-07：人类数据训练闸门（基础设施，不是强度实验）

本地工作区当前没有 `local_human_data/` 对局，因而没有任何真人强度或人类监督结论。审查训练入口发现：尽管网页
记录带有 `training_default=excluded_until_separate_quality_review`，手工把该 JSONL 传入旧训练 CLI 时仍会被当作
普通 hard-label 样本读取。现已修正为 fail-closed：检测到 `collector=local_human_opt_in` 默认报错；操作者必须显式
传 `--allow-local-human-data`，并让所有人类输入共同通过现有隐私/重复/规则档位/对手身份/最小局数审计。混合人类和
其他来源的同一文件也会拒绝，checkpoint report 仅留下无路径的聚合审计和人工授权标志。

这只建立将来经过人工复核的人类**行为**安全地用于 policy imitation 的路径；人类终局分数尚未用于 value regression，
也没有任何模型因该基础设施而变强或获得“击败人类”主张。下一步仍需要独立、足量且多来源的真人留出对局，或在
不依赖伪人类标签的自博弈联赛中取得可复现优势。

## 2026-08-07：run3 的 400 墙 Teacher 终检（未通过）

run3 曾在独立 80 墙上得到正向但跨零的点估计，不能据此作为“当前最强”继续累积训练。为一次性解决这个不确定性，
冻结参数 SHA-256 `01616a38ff34af71f1774995d5d3f9e08c4408fd85757ad563d9dfa339b19f70` 在完全未参与训练、选择或
此前评测的经典 seed `202608431` 起 **400** 个物理牌墙上对三名 Teacher 四座轮换（共 1,600 局、CUDA policy
head）终检。结果为 **−0.8550 ± 0.6448** 分/局，95% CI **[−2.1189, +0.4089]**，胡率 **23.8125%**，另有 5 局流局。

下界不为正、点估计也转负，因此 run3 不升级、不替换网页 Teacher、也不再被描述为下一轮优化的最强冻结起点。该
结果不宣称 run3 已显著弱于 Teacher（区间仍跨 0），但足以否定其具有已验证优势。完整可复核产物为
`artifacts/evaluation-run3-vs-teacher-terminal-202608431.json`。

## 2026-08-07：Teacher 单点 response 干预收集器（基础设施）

既然 run3 与 run4 都不再是可验证的最强基线，扩展 run4 后状态 selector 或从其继续 PPO 都不符合证据。为获得
直接对网页默认 Teacher 的因果数据，`collect_candidate_teacher_dagger_trajectories.py` 现允许 `--teacher-base`。
它以 Teacher 作为干预前、干预后和三名对手的冻结策略；只在一个预选 response 位置按已知
`ε × uniform + (1−ε) × Teacher` 行为采样，并导出真实执行动作和其 propensity。manifest 显式写入
`behavior_base=heuristic_teacher`、无 checkpoint／无推理设备，避免将规则 Teacher 伪装成神经参数来源。

core 小规模 smoke（3 个物理墙、12 个换座手）确认该路径可完成结算、生成安全 split 轨迹，并保持规则 Teacher
身份。尚未收集经典训练墙、训练 outcome/Q 头或测试 selector，因此这不是模型或强度实验。后续先做独立 OPE/支持度
审计，再决定是否值得生成足量经典数据。

## 2026-08-07：Teacher response 单点 OPE 审计器（基础设施）

新增 `xiamen_mahjong.off_policy` 与 `audit_teacher_response_intervention_ope.py`。每个有效观察必须有完整的
Teacher epsilon-mixture propensity、受保护的 `split_group_id` 和一个真实终局分数；IPS 与 doubly-robust delta 均按
物理牌墙组先平均、再计算标准误，避免把同一牌墙的四座轮换误当独立样本。DR 的 direct value 只允许来自不接触
test 墙的 outcome ensemble；候选 action 用跨成员终局分数 LCB 与 Teacher 标签比较，从而优先保持 Teacher。

输出状态固定为 `audit_only_one_response_teacher_override`。即使 IPS、DR 两项 95% 下界为正且两侧 ESS 达标，也只可
进入「一局至多一次 response override，随后 Teacher 后缀」的 200 墙筛选，不能作为完整策略、网页部署或打败人类的
证据。当前只有脚本与单测，尚无经典 Teacher 干预数据、outcome checkpoint 或 OPE 数值结果。

## 2026-08-07：Teacher response 干预数据与首轮 OPE（否决）

收集 `teacher-response-intervention-classic-v1`：800 个独立经典物理墙、四座轮换共 3,200 局，按墙组为
553/134/113 个 train/validation/test。安全 JSONL 不含随机种子、墙或隐藏手牌；三分区 `split_group_id` 交集均为
零。随机 response 记录为 **1,341/332/282**，测试动作覆盖吃 125、碰 88、过 53、胡 8、明杠 8，propensity 范围
0.1–0.8。原始 JSONL 保留在本机训练盘（train 文件 106 MB，超过 GitHub 单文件上限）；manifest 与 SHA-256 为
`c5dcc1d20697b8213dc5d961d74f9a069e3e1d786485083c81213965b7cc2034`（train）、
`97d4b72c0875323f08e6ad784bbe58398c562d53f8136cbaddfc36e3c88dadb8`（validation）、
`6dd701fc4e04340a491adbe446173f0a90c20606bd5bcec7ce6d90ef1efcdb77`（test）。

为避免任何历史 checkpoint 重新成为训练起点，以相同的**从未训练** policy anchor（seed 202608471）训练五个
afterstate outcome 成员，分别只更新其独立的 afterstate encoder/score/win heads。独立 test 的 score MAE 为
30.70–32.22 分，零预测为 32.85 分；这是 logged-action 校准改善，仍非选牌许可。

在 108 个有有效干预的 test 墙组（282 条随机 response）上，预先固定的 `LCB(z=1) 最大 score，否则 Teacher`
一次 override 候选产生 IPS **−1.6428 ± 4.4087**（95% CI **[−10.2839, +6.9983]**）与 DR
**−0.7897 ± 4.6188**（**[−9.8425, +8.2632]**）。target/base ESS 为 **66.1/218.7**，支持度足够但两个下界均不为正。
因此明确拒绝，不进行 200 墙、网页试用或任何强度主张；这个 test 墙组已冻结，不能用于事后调 LCB 阈值或选择成员。

## 2026-08-07：四层盲态选择协议与第二轮 LCB 网格（否决）

为避免把一份 test 既用于阈值搜索又用于强度结论，新增 `split_heldout_trajectory_groups.py`：强制按
`split_group_id` 将已隔离的 held-out 池分成 `selection` 与 `terminal`，terminal manifest 不导出动作或终局统计。
新增 `select_teacher_response_override.py`：它只在 selection 比较预注册网格；没有任何候选通过时，程序不读取
terminal 文件并写入 `selection_rejected_terminal_unread`。训练器同时加入 `--skip-test`，使 outcome 模型的 epoch
选择只接触 train/validation，report 显式保存 `terminal_test_read=false`。

第二批 `teacher-response-intervention-classic-v2` 使用 1,600 个全新经典物理墙；训练/验证含 2,332/583 条随机
response，selector 含 522 条、200 个墙组，且 Teacher 基线／propensity 契约均为零违规。五个全新随机 anchor
outcome 成员在 validation 的 score MAE 为 31.52–32.40 分，零预测为 34.97 分。此改善仍只针对 logged action。

在**打开 selector 结果前**固定 `{0, 12, 24, 36}` 分 LCB 优势网格。0 分候选 override 35.63%，IPS/DR 为
−2.5208/−2.8229，95% 下界 **−7.7110/−7.5706**；12 分候选 override 3.45%，下界 **−1.5236/−1.1817**；24 和
36 分不再偏离 Teacher，估计恰为 0，仍不满足严格正下界。所有候选均失败；terminal 209 个墙组（文件 SHA-256
`12ec74325811ac96457fbcc4ac03ec7f6b055d1ff3563ea57f7bfd11614a144f`）未读取、不提交结果、不允许改阈值后重跑。
结论是当前静态 afterstate score-LCB 候选族没有正向因果证据，而不是 Teacher 变弱或模型已经具备人类强度。

## 2026-08-07：Teacher 行为似然代理的 SMC smoke（否决）

为检验全历史 SMC 的低 ESS 是否只是确定性 Teacher likelihood 太尖锐，独立训练 core Teacher 行为 MLP（100 训练局、
40 独立验证局、16 隐层、4 epoch）。验证集 top-1 为 **76.84%**、交叉熵 **0.7254**，因此它足以作为“软概率是否缓解
退化”的 audit-only 候选，但不是游戏策略。

固定 core seed 202608503 的同一私有回放 snapshot、64 个 actor-visible setup 粒子、相同温度 1 与 uniform mixture
0.02 下，确定性 Teacher sequential SMC 在 8 个请求公开事件中只条件到 3 个，最终 ESS **3.01**；行为代理同样只到
3 个，最终 ESS **2.86**。两条路径都因后续事件在 proposal world 中结构不一致而全粒子失败。故软化行为 likelihood
没有改善、也不能修复提议分布；不提交到 SMC collector/Q/网页。下一阶段必须先改进带已知密度、可枚举 toy posterior
校准的结构 proposal，而不是继续训练代理或调温度。

## 2026-08-07：opening claim 精确 proposal 与行为相容性诊断

加强了 opening-aware structural proposal 的证据门槛：微型标记物理牌墙不只验证 `P(事件)`，还穷举目标 posterior，
并确认 proposal 的逐粒子 `p/q` 加权后能恢复暗手和翻金后墙位的边缘分布。该检查覆盖首个 claim 所需初始暗手，
以及 claim 后立即弃牌所需的联合初始手牌多重集；所有测试只在内存运行，不导出暗牌或墙。

新的 audit 将 structural ESS 与包含行为似然的 total ESS 分开。core `seed=271`、256 粒子、RNG `202608504` 的
opening→首次对手摸打为 256/256 接受，structural ESS **248.76/256**，total ESS **15.28/256**；旧总 ESS 的低值不应
再误读为该 `p/q` 的退化。core `seed=2`、256 粒子、RNG `202608505` 的 opening→claim 为 255/256 接受，structural
ESS **246.33/255**、total ESS **187.88/255**，但覆盖范围仅两条动作。

最初将同一 core `seed=2` 延长到 claimant 下一弃牌（RNG `202608506`）所见的 32/256 replay 接受已撤销为无效诊断：
该弃牌无人可响应时，引擎会在同一 transition 自动产生下一 normal draw，旧代码在 draw 前停止导致 224 个
`public_event_mismatch`。recognizer 现只接受弃牌后仍有 actor response option 的原子边界；不具备该条件者会在
未建模下一 draw 的 density 前安全跳过。

修正后，独立 core `seed=67`、256 粒子、RNG `202608507` 的 claim→discard 前缀为 **256/256** 接受，structural ESS
**250.74/256**、total ESS **33.30/256**。因此这条更长 prefix 的低 total ESS 确为行为 likelihood 退化，而非事件边界
错误。该 audit 仍没有进入 SMC、collector、动作价值、训练或网页；对确定性 Teacher 动作做未归一化 rejection 仍不是
合法的下一步。

为探索同时保留归一化的行为导向 proposal，新增 tile-factor 条件多元超几何 sampler：每种面额的正权重用 count-DP
精确归一化，返回条件 event 概率和 conditional `p/q`；toy 后验测试确认重要性加权可还原原始超几何分布。它只是一个
数学 primitive，不是行为模型。

在打开结果前固定 factor `{1, 1.5, 2, 3}`，只提高 claim 后公开弃牌面额的额外副本权重；墙组为 core `67,139,201,275`，
每个 factor/墙 256 粒子。平均 structural ESS fraction 为 **0.9787、0.9655、0.9441、0.8801**；平均 total ESS
fraction 为 **0.1334、0.1399、0.1252、0.0979**。所有候选都有远低于 20% 的单墙 total ESS，故此 tile-only family
整体失败；1.5 的小幅同组差异不用于选择或二次调参。没有生成训练数据、模型参数或网页行为。

## 2026-08-07：独立线性行为能量 proposal（终检否决）

以独立 core Teacher 数据（seed `202608520`，300 训练局 10,476 决策；独立 100 局 3,558 决策）训练
`RulePolicyModel`。其验证 top-1 为 **67.09%**、loss **1.0179**。对固定公开弃牌，模型的 `target × hand_tile`
线性系数被转换为 clipped `exp(scale × coefficient)` tile factors；DP 精确返回条件 hand proposal 的归一化和 `p/q`。
公共特征不会进入 factor，原始训练 JSONL 保持 Git 忽略；仅模型参数、报告和 SHA-256
`80587fcbc108888b884a39b59b4462040c6d54e618e88aa69093042adc0edfd3` 可复核。

selection 墙 `390,492,499,585` 与训练及旧 audit 墙不重叠。预先固定 scale `{0,0.25,0.5,1}`、clip=2、每墙 256：
平均 total ESS fraction 分别为 **0.2358、0.3187、0.3319、0.2395**，据此唯一选择 0.5。全新 terminal core `seed=707`
以 1,024 粒子复核，接受 1,023，total ESS **226.17/1,023 (22.1%)**，但 structural ESS 仅
**158.26/1,023 (15.5%)**。joint structural+total ESS ≥20% 门槛失败，故整条线性能量 proposal 路线拒绝；不导出
belief world、不生成训练数据、不训练/部署任何选牌模型。

## 2026-08-07：Progressive hiding 字段隔离（基础设施）

训练脚本新增固定维度的 `oracle → hide_wall → visible` 遮蔽向量，作用域仅为训练进程；visible stage 把三家暗手及
墙组成的片段置零。对同一 actor-visible core 局面重采样不同未知世界：oracle 向量不同、visible 向量严格一致，且
unknown stage 直接报错。actor `TeacherDecision`、轨迹、checkpoint、网页均未新增该向量；legacy privileged critic
仍保持 oracle-only。没有进行 progressive-hiding 模型训练、PPO、数据导出或强度评测。

## 2026-08-07：Progressive hiding critic 校准 smoke（拒绝，未训练策略）

字段隔离本身不足以说明课程有用，因而先执行一个可证伪的最小校准实验，而不是恢复 privileged-critic PPO。
core 固定 seed `202608560` 下，一个**全新随机**、未更新的合法动作 actor 对三名 Teacher 产生 64 个训练局
（554 个本家决策）与不同牌墙的 32 个验证局（269 个本家决策）。隐藏向量仅驻留进程内；没有写出暗牌向量，
没有 actor 或 critic checkpoint。两个 32 隐层 critic 从逐字节相同的初值出发、使用相同训练决策及相同按 epoch
打乱序列：direct baseline 全程以 visible 输入训练 9 epoch；课程候选按 `oracle → hide_wall → visible` 各 3 epoch。

唯一的预注册门槛是独立验证上的**最终 visible** Huber 与 MAE 均严格低于同预算 direct baseline。结果相反：
direct 为 Huber **0.1707421**、MAE **0.4069980**；课程为 **0.1715056**、**0.4072743**，差值（课程−直接）为
**+0.0007634**、**+0.0002763**。两项都变差，状态
`rejected_before_policy_training`。完整无私有特征报告为
`artifacts/progressive-hiding-core-smoke-v1.json`，生成脚本为
`scripts/audit_progressive_hiding_critic.py`（source `b025af5`）。

这否决的是该预注册的 core/9-epoch 课程作为进入策略训练的依据，并不把小样本差异解释为 progressive hiding 的一般
定理。若日后重访，必须先建立真正缩小规则的独立基准和新的预注册 schedule／墙组；不得在本报告的训练或验证墙上
调节阶段比例、学习率后声称通过，也不得以此报告为依据训练 PPO、导出模型或作任何强度主张。

## 2026-08-07：Teacher 单点弃牌干预 v1（terminal 否决）

此前 response-only 干预覆盖过窄，故新建严格 phase-isolated 的 `discard`（摸后决策阶段，包含可用的胡／杠）
单点因果路径。预注册先于数据生成提交：经典档连续 1,600 个物理墙 seed `202608700` 起、四座轮换共 6,400 局，
三名对手和前后缀均为 `HeuristicTeacherAgent`；每局仅在前 8 个该阶段决策随机选一处，以
`0.4 × uniform + 0.6 × Teacher` 执行并记录 exact propensity。按物理墙分出 train/validation/held-out，后者再按墙组
拆成 selection/terminal，均无组重叠。原始 JSONL（296 MB）只保留本地；GitHub 只存 manifest、无标签覆盖统计和最终审计。

随机干预数为 train **3,003**（968 墙）、validation **727**（228 墙）、selection **579**（191 墙）、terminal
**581**（190 墙），均超过预注册最少 2,500/500/500/500。五个相同 fresh policy anchor、不同训练 seed 的 128 隐层
afterstate outcome 成员只训练独立 outcome encoder/heads，均传 `--skip-test`；validation score MAE 为
**29.03–29.36** 分，优于零预测 **33.99** 分，因而允许打开 selection。所有五组参数与报告已提交，均明确
`terminal_test_read=false`。

selection 仅比较预注册 LCB 优势阈值 `{0,8,16,24}`（5 成员均值减 1 倍成员标准差），要求 grouped IPS 和 DR 的
95% 下界均正、两侧 ESS ≥50。0 与 8 分候选下界为负或 target ESS 不足；24 分同样失败。16 分阈值仅以
**1.73%** 的 override rate 通过 selection（IPS/DR 下界 **+0.049/+0.099**，ESS **372.6/378.6**），因此按既定规则
唯一进入 terminal。

独立 terminal 的同一阈值立即失败：IPS **−0.0148 ± 0.1665** 分/局，95% CI **[−0.3411,+0.3115]**；DR
**+0.1392 ± 0.1788**，**[−0.2113,+0.4897]**；target/base ESS **372.7/376.7**。支持度充足而双下界为负，
故拒绝该静态 score-LCB 单点弃牌候选，不做 200 墙实战筛选、PPO、网页接入或强度主张。terminal 墙组已消耗，
不得在此数据上更换阈值、成员、LCB z 值或模型后重跑。完整 protocol、覆盖与选择报告分别为
`knowledge_base/teacher_discard_intervention_v1_protocol.md`、
`artifacts/teacher-discard-intervention-classic-v1/coverage.json` 与
`artifacts/teacher-discard-intervention-classic-v1-heldout/selection-result.json`。

## 2026-08-07：原始 DR 相对优势诊断（拒绝直接训练）

为给下一候选族建立正确统计基元，新增每个合法动作相对 Teacher 的 DR 伪优势：直接模型差
`q(a)−q(Teacher)`，再对 logged action 与 Teacher action各加一次已知 propensity 的 residual。五个 outcome 模型均只
见过 train 墙，因此在 v1 validation 的 727 个随机弃牌干预／228 个墙组上可以安全做 aggregate-only 审计；没有读取
selection/terminal、没有训练 advantage head 或选择动作。

结果否决了朴素的未缩减伪标签：7,135 个非 Teacher 动作的直接模型 gap 标准差仅 **11.33** 分（p99 **14.40**），
而 DR 伪优势标准差为 **206.66** 分，范围 **[−1,737.25, +3,251.66]**；logged action 伪优势标准差更为 **628.66**。
相应非 Teacher propensity 的中位数为 **0.0333**，范围 **0.0235–0.2**，重要性残差尾部放大与此一致。故不以原始 DR
向量训练 action head、不做 AWR/PPO 或网页选择。完整无私有特征报告为
`artifacts/teacher-discard-intervention-classic-v1/dr-advantage-validation-audit.json`。

未来若再研究相对优势，必须使用按物理墙 cross-fitting 的 direct model，并在**全新**四层墙组上预注册有限的
shrinkage／SWITCH 类方差控制与偏差审计；不能在已消耗 v1 selection/terminal 上调 clipping 或 threshold。

## 2026-08-07：Teacher 相对优势 shrinkage v2 覆盖门槛（拒绝，未训练）

为降低非 Teacher action 的低 propensity，v2 预注册把单点摸后干预的 epsilon 从 0.4 提至 0.8，并使用全新 classic
seed `202608900` 起的 2,000 个物理墙、四座轮换。其余契约不变：每局只在前 8 个 `discard` phase 决策中干预一次，
随后恢复 Teacher；train/validation/held-out 按整墙隔离，held-out 进一步分 selection/terminal。原始轨迹 372 MB
仅在本地，GitHub 只保存 manifest 和无标签覆盖统计。

得到的随机干预覆盖为 train **3,673**、validation **931**、selection **693**、terminal **787**。预注册门槛为
3,500/800/700/700，因此 selection 少 **7** 条即失败。即使其余三项通过，也不允许事后降低门槛、改变 split salt、
合并 v1 数据或开始 K-fold/direct/shrinkage learner；terminal 只作无结局的覆盖计数，不导出动作/得分汇总。该墙组
整体拒绝，完整统计见 `artifacts/teacher-advantage-shrinkage-classic-v2/coverage.json`。下一份协议必须使用全新墙组。

## 2026-08-07：Teacher 相对优势 shrinkage v3（训练完成，selection 待审计）

v3 严格不复用 v1/v2 墙组：classic 从 seed `202609200` 起的 **2,400** 个新物理墙、四座轮换；行为为
`0.8 × Uniform(legal) + 0.2 × Teacher` 的单次摸后 `discard` 干预、随后恢复 Teacher。只读取无标签的覆盖统计后，
train/validation/selection/terminal 随机干预数分别为 **4,402 / 1,152 / 921 / 911**，超过预注册
**3,500 / 800 / 700 / 700** 门槛；selection 与 terminal 的完整墙组无重叠。原始轨迹、fold 和 checkpoint 均只保留本地。

train 按完整物理墙分成 K=3（482/487/464 墙组，重叠 0）。每一个 direct outcome 模型只学习另外两折的随机弃牌行动，
固定 32 epoch 且读取 validation/selection/terminal 均为 false。validation 上三个 direct 模型的等权 logged-action
score MAE 为 **28.96** 分；原始 DR 非 Teacher 伪优势仍有标准差 **162.09**、范围
**[−1,060.83,+2,115.07]**，因此再次明确拒绝未缩减标签。预注册 winsor C=20/40/80 后对应标准差为
**14.70 / 23.90 / 41.86** 分；这三个标签均显式标记为有偏，只能训练候选，不能 OPE。

三个独立的 centered relative-advantage MLP 均只接收 OOF actor-visible 标签，Teacher 动作按构造输出严格为零，
无终局结果进入标签文件或训练器。validation 的伪标签拟合 MAE 为 **8.02 / 14.43 / 24.67** 分，top-action 一致率为
**50.17% / 45.66% / 40.71%**；这些仅为固定模型的诊断，未用于改变 C、阈值、架构或 epoch。下一步只能在 selection
一次性比较 `(C,T)` 的九个预注册组合，要求未缩减 grouped IPS 与 DR 95% 下界都为正、两侧 ESS ≥75；若全失败，terminal
必须保持未读。

selection 已执行且**全部拒绝**，状态 `selection_rejected_terminal_unread`。C=20 的 T=0/8/16 的 IPS/DR 下界分别为
−12.416/−9.977、−1.761/−1.105、−1.792/−0.974 分/局；C=40 为 −15.344/−12.012、−4.069/−2.690、
−0.535/−0.428；C=80 为 −10.495/−6.921、−4.676/−3.875、**−0.101/+0.061**。后者虽有正 DR 下界且两侧
ESS 242.2/245.2，仍因 IPS 下界不为正而失败；其余组合也均失败（C=20,T=0 的 target ESS 仅 74.69）。因此没有
唯一 winner、没有读取 terminal、没有 200 墙实战筛选，也没有模型参数上传为可用 AI。v3 selection 墙组已消耗，禁止
在其上改变 C、T、网络、训练轮数、direct 成员或 OPE 聚合后重跑；terminal 仍保留，但只可用于未来已通过其自身新协议的
候选，不能作为本候选的重试集。完整审计为本地
`artifacts/teacher-advantage-shrinkage-classic-v3-heldout/selection-result.json`。

## 2026-08-07：Teacher 随机相对优势 v4（训练完成，selection 待审计）

为与 v3 的确定性 argmax+threshold 族严格区分，v4 固定为小概率 stochastic residual：Teacher 仍占主要概率，
其余概率按 clipped relative-advantage softmax 分配。新 classic seed `202610000` 起的 **3,000** 个物理墙、四座轮换，
行为仍为一次 `0.8 × uniform + 0.2 × Teacher` 摸后弃牌干预再恢复 Teacher。无标签 coverage 为 train/validation/
selection/terminal = **5,509 / 1,469 / 1,032 / 1,126**，超过预注册 5,000/1,200/1,000/1,000；selection 和 terminal
完整墙组为 347/365 且重叠 0。

训练墙按完整物理墙分为 K=3（617/590/600 墙组、重叠 0）。三个固定 epoch direct outcome 模型均只用另两折的
随机弃牌行动、没有读取 validation/held-out；validation 的三成员 logged-action score MAE 为 **28.32** 分。未缩减
DR 非 Teacher 伪优势标准差仍为 **162.28** 分、范围 [−1,378.88,+3,793.60]，故未直接学习；C=20/40/80 的 OOF
截断标签分别训练独立 centered relative MLP。其 validation pseudo-label MAE 为 **7.80 / 13.67 / 23.08**，Teacher
预测零点均严格为 0；这些诊断没有改变任何 C、beta、temperature、架构或 epoch。

下一步只能在未读 selection 上比较固定 12 个 `(C,beta,tau)`，其中 beta∈{.05,.10}、tau∈{8,16}；随机分布的 OPE
使用未缩减 stochastic IPS/DR，且两侧 ESS 必须≥100、双 95% 下界为正。若无 winner，terminal 必须保持未读；若有唯一
winner，脚本才可读取 terminal 一次。

selection 已执行且全部拒绝，状态 `selection_rejected_terminal_unread`。支持度已不再是限制：12 个组合的 target ESS
为 **294.0–326.4**、Teacher baseline ESS **268.0**，全部远高于 100；但 4.27%–8.79% 的期望非 Teacher 概率仍没有
正向证据。最接近的是 C=80、beta=.05、tau=8，grouped IPS/DR 95% 下界仍为 **−0.372/−0.395** 分/局；其余组合的
下界更低（beta=.10 普遍约翻倍负向）。因此 v4 stochastic residual policy 整体拒绝：不读取 terminal、不进行 200 墙
实战筛选、不接入网页，也不把诊断模型参数上传为可用 AI。v4 selection 墙组已消耗，禁止在其上增加 beta/tau/C、换
网络、direct member 或 OPE estimator 后重跑；terminal 保持封存。完整本地审计为
`artifacts/teacher-stochastic-relative-classic-v4-heldout/selection-result.json`。

## 2026-08-07：公开信息防守 Teacher v5（selection 拒绝，terminal 未读）

为避免继续在已失败的 direct-score residual 与 OPE 家族中微调，v5 采用可直接四座对局验证的确定性规则候选：在
`AvailabilityTeacherAgent` 的自家牌形／公开余牌听口得分上，扣除 `risk_weight × public_danger`。危险度只使用本家
手牌、河牌、副露、翻金和公开回合数；不得读取牌墙、对手暗手、随机种子或未来事件。固定权重网格为
`{0, 1, 2, 4, 8}`，并在 classic `seed=202611000` 起 160 个物理墙上四座轮换；每个候选均为 640 局，对手均为
冻结的 `HeuristicTeacherAgent`。promotion gate 预先固定为按物理墙配对的 score delta 95% 下界严格大于零。

五项均未通过，终局 `seed=202611500` 起 400 墙保持**未读**。`risk_weight=0`（仅公开余牌听口）相对 Teacher 的
均值／95% 下界为 **−0.525 / −1.725** 分/局；`1` 为 **−3.422 / −5.949**，`2` 为 **−3.423 / −6.029**，`4` 为
**−9.322 / −11.836**，`8` 为 **−17.033 / −19.022**。除 0 外每个权重的上界也为负，且随权重增大持续恶化，表明该
公开危险代理与本规则及 frozen Teacher 的牌效率权衡不相容，不能解释为“防守增强”。因此整个固定族被拒绝：不重跑
selection、不读取 terminal、不接入网页，也不将其作为训练教师或模型参数上传。审计汇总仅保留在本地
`artifacts/risk-aware-teacher-v5/selection-result.json`；协议和实现为 commit `3ec99db`。

## 2026-08-07：从零初始化自博弈联赛 v1-a（selection 拒绝，terminal 未读）

此前所有可部署候选与旧 neural checkpoint 均已失去训练起点资格，故 v1-a 不继承任何历史参数或轨迹：fresh candidate-MLP
由 `seed=202611800` 初始化（metadata `ancestry=none`），只读取行动者本家手牌和公开信息。固定训练为 classic 下
8 iteration × 256 个候选席、rollout batch 64、PPO epoch 2、learning rate `5e-5`；每个非候选座位独立以 0.5 概率
使用冻结 Teacher 或本轮更新前的 current-policy snapshot。没有外部 checkpoint、oracle critic、Q/outcome 标签或中间
checkpoint 挑选。训练共生成 2,048 个候选席对局；只审计 final iteration-8 checkpoint。

预注册 selection 为 classic `seed=202612500` 起 80 个物理墙、四座轮换（320 局）；唯一门槛是相对三名 Teacher 的
按墙 paired score delta 95% 下界严格为正。结果为 **−21.750 ± 0.722** 分/局，95% CI
**[−23.165, −20.335]**，2/320 胜、5/320 流局。下界和点估计均大幅为负，状态
`selection_rejected_terminal_unread`；独立 terminal `seed=202612700` 起 320 墙从未读取。

这否决的是“随机 actor + 终局净分 PPO + Teacher/current 快照 50:50”这一固定 bootstrap 配置。它表明在当前稀疏终局
回报中，纯随机自博弈无法获得与规则 Teacher 竞争的初始策略；不再扩增 iteration、调整 PPO 超参、改 snapshot 比例或
复用该 selection/terminal 墙。checkpoint 和原始报告只留在本地 ignored artifact，不上传为模型参数，也不接入网页。
下一步必须先建立一个可独立审计的强启动信号（经质量审计的人类对局，或新的、带独立强度证据的规则/搜索教师），而不是
重复纯随机 PPO。

## 2026-08-07：Teacher-anchored residual self-play v1-b（selection 拒绝，terminal 未读）

为避免 fresh PPO v1-a 的随机 actor 灾难性退化，新候选不继承任何历史权重：candidate-MLP 由
`seed=202611950` 初始化且 policy head 全零。每个合法集合中，冻结 `HeuristicTeacherAgent` 推荐动作固定加 prior
logit 0，其余动作为 −5；因此残差为零时 deterministic policy 严格等于 Teacher。prior 只由本家／公开信息生成，并
同时记录在 PPO 行为 step、重放到 update 和 current-snapshot 对手，防止训练/推理不一致。训练固定为 classic
8×256 candidate-seat episodes、PPO epoch 2、Teacher/snapshot=0.5/0.5、无历史 checkpoint、无 oracle critic。

最终 iteration-8 residual 只在 wrapper margin=5 下评测 classic `seed=202612900` 起 80 个物理墙、四座轮换。结果是
**严格零差异**：candidate 与 Teacher 均为 80/320 胜、0 流局、总分 0，paired mean/stderr/95% 下界均为 **0**。这说明
在固定训练预算和安全余量下，任何 learned residual 都没有跨过 deterministic Teacher 的排序边界；它既没有退化，也
没有可证明提升。严格门槛要求下界 >0，故状态 `selection_rejected_terminal_unread`，`seed=202613100` 起 320 墙
terminal 未读。

该固定 margin/预算/最终 checkpoint 整体拒绝：不在 selection 墙上降低 margin、增加 iteration、减少 entropy、挑选
中间 checkpoint 或改变 teacher/snapshot 比例；不上传该 residual 参数，不接入网页。与 v1-a 合并的证据表明，当前
仅靠终局 PPO 一端要么远离 Teacher（随机 actor），要么因安全锚定而没有可见政策改动。下一阶段不能继续微调这两条
PPO 族；需要经审计的人类数据，或一个独立验证过的更强搜索／规则教师来提供启动与探索信号。

## 2026-08-07：精确一摸听牌诊断与限域平分裁决候选 v1（selection 运行中）

全局公开信息一步前瞻 Teacher 已在独立评测中显著为负，不能把“再看一摸”当作默认改进。本轮先加入纯手牌结构
oracle：对一个正常弃牌后的手牌，枚举 34 种可能摸入和每一种摸入后的合法弃牌，精确返回能够进入听牌的路线；该
函数默认不读牌墙、对手暗手或未来随机数，并可由调用者传入公开的跟打限制。40 墙、classic、`seed=202614100`
起的只读聚合诊断记录了 1,245 次 Teacher 弃牌，其中 1,060 次为原分差不超过 2、且没有直接听牌的比较对象；
Teacher 的一摸听牌摸入牌种数均值为 2.074，只有 47 次（**4.43%**）存在有更多路线的近似替代选择。路线差值
直方图为 0:1013、1:15、2:11、3:5、4:9、5:3、6:2、7:2。诊断状态明确为
`diagnostic_only_not_authorized_for_action_selection`，不写逐局手牌、牌墙、对手暗手或训练标签。

因此没有重复已淘汰的全局一步前瞻，而是预注册唯一 `ExactOneDrawTenpaiTieBreakTeacherAgent`：frozen Teacher
的直接听牌、主评分、胡/杠/游金/吃碰逻辑均不动；只有两个未听牌弃牌的 frozen 分差不超过 2，且在公开可见张数、
假设摸入扣张、经典跟打约束下，替代选择的“可摸张数 × 后续活听张数”严格多至少 1 时，才改变弃牌首选。单测覆盖
精确枚举的跟打约束、参数 fail-closed、强制高阈值回退到 frozen Teacher、受控严格门槛重排，以及交换墙/对手暗手
后不变。候选尚未接入网页或训练。

独立 selection 已在 classic `seed=202614500` 起 160 个物理墙、每墙四座轮换完成。候选 640 局均分为
**+2.1203 ± 1.1525** 分/局、175/640 胜；相对同墙三名 frozen Teacher 的 paired delta 95% CI 为
**[−0.1386, +4.3793]**。尽管点估计为正，预注册门槛要求下界严格大于 0，故状态为
`selection_rejected_terminal_unread`。`seed=202614800` 起 400 墙 terminal 没有读取，候选不接入网页、不生成
训练标签、不替换默认 Teacher；selection 墙不得以不同阈值或评分函数重跑。

## 2026-08-07：用户授权的 400 墙复核（仍不能确认优于 Teacher）

在上述 selection 被拒绝后，用户明确要求增加样本确认。于是使用此前封存的 classic `seed=202614800` 起连续 400
副新物理墙，扩展至 `202615199`，每墙四座轮换，共 1,600 局；候选固定为同一
`ExactOneDrawTenpaiTieBreakTeacherAgent`，对手为三名网页版默认 `HeuristicTeacherAgent`。结果为候选相对 Teacher
平均 **+0.5119 ± 0.6464** 分/局，95% CI **[−0.7550, +1.7787]**，候选 413/1,600 胜，5 局流局。

该复核的点估计仍为正，但区间明显跨 0；因此 160 墙与 400 墙都没有给出“新候选更强”的证据。它不进入网页、不
生成训练标签，也不改变原 selection 的拒绝状态。常规后续筛选改用 100 副物理牌墙，只有出现稳定正向信号才扩大
到 400 墙确认。

## 2026-08-07：安全轨迹 v4 首轮 GPU 模仿基线（筛选拒绝）

为使后续序列模型能利用真实的历史出牌而不读未来，本轮把训练轨迹升级为 v4：完整牌局只保存一次全局公开事件流，
每个决策保存该时刻的公开前缀游标和本家相对座位的 24 事件短窗口。训练导出不含牌墙种子、行为种子、对手暗手或
牌墙顺序；`audit_training_trajectories.py` 会在训练前验证这一边界及短窗口与前缀的一致性。游金单决策课程明确标记
为 `recent_window_only`，金牌锁课程则由规则引擎构造并验证“可碰、不可抢胡”的完整公开前缀，不能伪装成自然对局。

首轮冻结 Teacher 语料由 2,500 副独立 classic 物理牌墙、250 个响应课程、272 个游金课程和 136 个金牌锁课程组成。
安全审计通过：共 3,158 条轨迹、109,579 个决策，其中完整历史 109,307 个、窗口课程 272 个；响应决策 25,333 个、
游金决策 273 个、金牌锁决策 136 个，均不携带隐藏信息。数据按整副物理墙划分 train/validation/test，不以决策行
切分。

在 RTX 2080 Ti 上训练的 `candidate_mlp`（128 hidden、20 epoch、`synthetic_weight=16`）以验证集 epoch 19
为 checkpoint。独立测试集动作一致率为 **96.95%**、policy loss **0.1375**，仅说明对冻结 Teacher 的离线模仿质量；
它不是强度结论。随后在从 `202612000` 起、与训练种子隔离的 100 副物理牌墙上四座轮换（400 局）对阵三名
`HeuristicTeacherAgent`。候选均分为 **+0.280 ± 0.847** 分/局；按物理墙配对的 95% CI 为
**[−1.380, +1.940]**，跨越零。相同牌墙的 Teacher 基线严格为 0。

因此该 checkpoint 状态为 `selection_rejected_terminal_unread`：不进行 400 墙复测、不接入网页、不作为新 Teacher、
不上传模型参数。下一阶段只扩大经 v4 审计的物理墙数据，并在能读取完整公开历史的序列模型上重新训练；若新的
100 墙筛选未给出严格正向下界，仍立即淘汰。

## 2026-08-07：v4 扩展数据集与完整公开历史模型（训练启动）

以四个互不重叠的种子段各采集 5,000 副 classic Teacher 物理牌墙，并在每段加入 500 个物理响应课程、2,500 个
游金课程和 2,500 个金牌锁课程。合并安全审计通过：20,000 副真实牌墙；42,000 条总轨迹；**900,589** 个决策；
其中 **890,589** 个完整公开历史、10,000 个显式 `recent_window_only` 课程，历史游标违规、隐藏字段违规均为 **0**。
响应决策为 213,469 个，游金 10,008 个，金牌锁 10,000 个。最大自然公开前缀为 143 个事件。

为使该数据量可训练，训练器改为有界随机缓冲的逐条 JSONL 流式输入：每个 epoch 覆盖每个 train 决策一次，却不把整个
5,000 墙分片展开到内存。新的 sequence 配置以 `--full-public-history --history-window 160` 按每条决策的 v4 游标读取
公开前缀；合成窗口课程保持短历史边界。模型 checkpoint 也保存 `event_length`，网页/评测推理从实时游戏的公开事件流重建
同一座位相对历史。

此时只授权开始 GPU 离线模仿训练和固定留出集比较；尚未产生新 checkpoint 指标，也没有进行实战筛选、400 墙终检、网页
接入或“超过 Teacher”的声明。

## 2026-08-08：full-history Transformer 完成并按资源政策退休

`policy-value-v4-full-history-transformer-run1` 已完成：`public_sequence_transformer`、hidden 128、4 heads、
160 事件、8 epochs，使用 724,243 / 91,808 / 84,538 个 train/validation/test 决策；验证集选择 epoch 7。
validation 动作一致率为 **98.68%**、policy loss **0.06838**；独立 test 动作一致率为 **98.72%**、policy loss
**0.07030**。游金、金牌锁课程 test 一致率均为 100%，`physical_response_pass_search` 为 100%。checkpoint
约 1.62 MB，训练在 RTX 2080 Ti 上实际运行接近 3 小时。

这些指标只证明模型能高精度复制冻结 Teacher。训练目标没有 Teacher 之外的动作优势标签，且当前机器、数据和项目目标
不支持继续投入大序列模型。根据 2026-08-08 资源约束决策，该 checkpoint 固定标记为
`diagnostic_only_retired_by_resource_policy`：**不做 100/400 墙实战、不接入网页、不作为新 Teacher、不用于
Transformer/GRU 续训，也不上传为可用模型参数。** 原 checkpoint/report 只保留本地供复核。

下一阶段停止采集同源 Teacher 百万级模仿数据，转向现有语料中的低 margin/分歧状态：先建立只用公开信息的
`SlowExpertTeacher` 单类局部 oracle，再比较 linear、GBDT 和不超过约 50 万参数的小 MLP residual。第一版只允许
1%–5% 高置信 override，其余状态严格回退 `HeuristicTeacherAgent`；完整协议见
[`resource_constrained_training_plan_2026-08-08.md`](research/technical_routes/resource_constrained_training_plan_2026-08-08.md)。

## 2026-08-08：公开听牌价值 SlowExpert v1（selection 拒绝）

首个资源受限局部 oracle 从 1,000 副已有 v4 natural Teacher 轨迹出发，只做聚合审计，不导出手牌、历史、牌墙、
种子或逐决策候选。33,043 次普通非游金弃牌中，946 次存在 Teacher 分差不超过 2 的多个直接听牌选择；229 次
替代项具有更高的“公开剩余张数 × 可见立即自摸结算”，覆盖普通弃牌 **0.693%**、近分听牌选择 **24.2%**。
224/229 次差异主要是更多公开剩余张数。审计 1,000 副耗时 2 分 29.77 秒、峰值 RSS 约 283 MiB，符合轻量预算。

据此只实现唯一 `PublicTenpaiValueTeacherAgent`：胡、杠、响应、游金、金牌锁和非听牌弃牌均严格回退 frozen
Teacher；只有 frozen 选择已直接听牌、另一听牌弃牌分差不超过 2，且上述公开加权值严格增加至少 1 时才覆盖。
实现不读对手暗手、真实牌墙、未来随机数或他家暗杠牌面。预注册协议与参数在强度筛选前写入
[`public_tenpai_value_teacher_v1_protocol.md`](public_tenpai_value_teacher_v1_protocol.md)。

全新 classic `seed=202616000` 起 100 副物理牌墙、每墙四座轮换的 selection 中，候选 97/400 胜，均分
**−0.7275**；相对同墙 frozen Teacher 的 paired delta 95% CI 为 **[−1.8200, +0.3650]**。未通过严格正下界
门槛，故 `seed=202616200` 起 400 墙 terminal 保持未读，候选不接入网页、不成为训练 Teacher。这个反证表明
“可见余牌 × 立即胡牌分”能发现结构差异，但不能单独近似整局 Q；下一类 SlowExpert 必须直接针对规则错误或使用
经过校准的局部价值证据，不能继续手调同一代理。

## 2026-08-08：Teacher + 小 MLP 约 2% 置信门控（selection 拒绝）

为检验规则与纯神经网络之间的中间态，复用历史上唯一在独立 Teacher 对局中出现正点估计的 compact run3，而不
重新训练网络。固定 checkpoint 为 feature-v3、hidden 128 的 `candidate_mlp`，文件 218,453 字节，SHA-256
`01616a38ff34af71f1774995d5d3f9e08c4408fd85757ad563d9dfa339b19f70`。候选只允许模型覆盖普通弃牌；response、
胡、杠、游金和金牌锁均严格回退 frozen Teacher。

门控 margin 不用实战收益选择。v4 part-01 validation 的 14,851 个合格普通弃牌中，以严格 logit advantage
`> 2.4152190685272217` 覆盖 297 个（1.9999%）；未参与阈值选择的 test 为 251/13,519（1.8566%），通过预设
0.5%–3.5% 覆盖稳定带。聚合审计不导出手牌、历史、动作 trace、牌墙、对手暗手或种子，耗时 2 分 20.19 秒，
峰值 RSS 约 1.04 GiB。该步骤只校准行为覆盖，不作为强度证据。

预注册后在全新 classic `seed=202619000` 起 100 副物理牌墙、每墙四座轮换执行 selection。真实访问状态覆盖
41/2,866（1.4306%）；候选 98/400 胜，均分 **−0.7125**，paired delta 95% CI
**[−2.3297, +0.9047]**。未通过严格正下界门槛，故 `seed=202619200` 起 400 墙 terminal 未读取，候选不接入
网页、不进入真人评测、不产生新的训练标签。

这一反证说明旧 MLP 的“高置信”只表示其自身 logit 尺度，不是动作优势校准。Teacher 模仿准确率、logit gap 和
低覆盖 gate 三者组合仍没有独立强度来源。下一轮停止从同源 Teacher logits 构造 residual 标签；只接受经审计真人
数据，或至少 32 个共享 belief worlds 的局部 paired advantage 作为新标签候选，并先做标签稳定性诊断再训练。

## 2026-08-08：run3 override 的 32-world paired 标签门槛（未通过）

为区分“模型动作本身无优势”和“100 墙评测方差过大”，在强度筛选失败后另行预注册一个只读标签质量试验。
classic `seed=202620000` 起 50 副新物理墙、四座观察者，共扫描 1,388 次普通弃牌并命中 35 个固定 run3 gate
状态。每状态从行动者公开信息集重采样 32 个 private worlds；同一 world 分别强制模型替代牌和 Teacher 牌，之后
均由 frozen Teacher 完成。产物只保存聚合 paired delta，不保存状态、手牌、历史、动作 trace、world、牌墙、暗手
或 RNG。

1,120/1,120 个 world 均通过合法集合一致性检查。35 个状态中只有 **1** 个替代项 95% 下界严格为正，**4** 个
上界严格为负，剩余 **30** 个跨 0；跨状态均值 **−1.9143 ± 2.5612**，95% CI
**[−6.9343, +3.1058]**。预注册门槛要求至少 5 个显著正例、5 个显著反例，因此
`paired_label_pilot_not_ready`：不导出训练行、不训练 linear/GBDT/MLP，不以 noisy argmax 伪造优势标签。

工程上这证明 32-world shared-belief paired 管线可用且本轮 world 接受率为 100%；科学上它否决的是 run3 高
logit-gap 作为候选来源。下一数据试验若继续使用该管线，必须换成独立、已有正向依据的候选规则，而不是调 run3
margin；优先级是精确一摸候选，其次是经授权真人纠错数据。

## 2026-08-08：精确一摸 public-fix v2 的 32-world 标签门槛（未通过）

复核 `ExactOneDrawTenpaiTieBreakTeacherAgent` 时发现旧 v1 把他家暗杠的具体牌面计入“公开”余牌；trajectory-v4
明确只公开暗杠事件、不公开其 face。实现已修为忽略他家暗杠牌面，并加入替换该隐藏 face 后决策不变的回归测试。
因此旧 160/400 墙正点估计只保留为候选来源的方向性先验，不能证明修正版 v2 强度。

在全新 classic `seed=202622000` 起 50 副物理墙、四座观察者上，扫描 705 个候选普通弃牌并取得 40 个
v2/Teacher 分歧状态。每状态 32 个共享 actor-visible belief worlds，1,280/1,280 个 world 可用。仅 **2** 个状态
的 v2 paired delta 95% 下界为正、**2** 个上界为负，**36** 个不确定；跨状态均值 **−0.9000 ± 1.7625**，
95% CI **[−4.3546,+2.5546]**。未达到正负各 5 个的标签可辨识门槛，故不导出逐状态样本、不训练任何模型，
也不追加 world 数来精化一个尚未按完整历史校准的近似 belief。

## 2026-08-08：真人—Teacher 分歧记录契约 v1（基础设施就绪，尚无真人数据）

由于两个独立候选来源的 32-world 标签试验均未通过，下一项监督来源改为经明确授权的真人训练局。网页记录器现在
在每个真人决策旁保存 `reference_teacher_index`：它是同一合法动作集合中 frozen `heuristic_teacher_v1` 的动作
索引；`chosen_index`/`executed_index` 仍是人类实际选择。该字段不进入观察特征、不保存 Teacher 的隐藏状态，且
记录仍不含牌墙、种子、他家暗手、账号、网络标识或时间戳。
审计同时发现仅在导出 JSON 中删掉暗手仍不够：网页原有调试开关可能让人类在行动时看见三家暗手。记录模式现由
服务端强制拒绝暗手揭示，并要求 metadata 为 `opponent_hand_reveal=server_forced_disabled`；缺少该标记的旧记录
直接判无效，防止 privileged human label 污染。

新聚合审计要求至少 100 个完整 training 牌局、500 个可表示的普通弃牌参考决策、其中 50 个真人/Teacher 弃牌
分歧、全部真人动作 100% 参考覆盖、
单一规则/对手/参考 Teacher 身份。当前 `local_human_data/` 没有可用记录，所以状态只是
`collector_ready_no_human_data`，没有启动模型训练。训练器增加显式 `--human-teacher-disagreement-weight`，默认
1.0；它只允许在完整牌局 train/validation/test 切分后比较轻量候选，不能把 evaluation 对局或这项基础设施本身
解释为人类强度证据。

### 真人纠错 residual gate v1：选门和网页一致性基础设施

新增 `select_human_teacher_residual_gate.py`。它只比较另一张普通弃牌，validation 阈值固定为 1%/2%/5% 三个覆盖
目标；每项必须至少 20 override/20 个完整牌局 group、override 真人精度 Wilson 95% 下界 >50%，且按牌局等权的
gated−Teacher 动作准确率 95% 下界 >0。validation 没有 winner 时，程序不会打开 test 文件；专项测试验证了这一
物理未读边界。通过后 test 仍重复相同下界，且覆盖不得超过 7.5%，只能解锁全新 100 墙 Teacher 筛选。

`ConfidenceGatedTeacherAgent` 现同时限制 Teacher 动作和替代动作都必须为 `discard`，避免未来模型把高 logit 的杠、
胡或响应误当成普通弃牌 override。网页增加显式 `--ai-teacher-gate-margin`，identity 写入 checkpoint SHA、wrapper
版本和精确 margin；不传时默认仍为规则 Teacher。当前没有真人完整局，也没有 fresh checkpoint，故状态为
`infrastructure_ready_human_test_unread`，不能作任何强度结论。

训练入口也已补上物理隔离：`train_policy_value.py --reserve-local-human-test-for-gate` 要求人类 train/validation
存在、同时要求训练命令中完全没有人类 test；报告标记 test 为 `physically_unread_reserved_for_teacher_gate`。
固定 wrapper `train_human_teacher_residual_v1.py` 使用 fresh seed 202623100、candidate MLP 128、12 epoch、真人分歧
权重 2.0，并按 `local_human_opt_in` validation 选 epoch；命令不含 `--init-checkpoint`。单测核对固定命令不包含
人类 test 路径或旧 checkpoint。

随后把数据门槛与部署域进一步对齐：响应、胡、杠、游金或金牌锁分歧不再计入 500/50 门槛；审计只统计普通
“discard→discard”可表示决策。固定训练命令加入 `--human-discard-corrections-only`，gate scanner 复用同一个
eligibility 函数，避免“审计通过但真正可部署纠错为零”的假阳性。
网页状态同步增加本次 recording session 的完成局数、可表示普通弃牌数与弃牌分歧数，便于采集者看到有效进度；
它不替代跨会话去重审计，也不把未完成牌局写入文件。

## 2026-08-08：低分歧专家纠错审阅 v1（题库就绪，等待人工标签）

完整对局纠错采集效率较低，因此新增独立 actor-visible 审阅路径。固定 Teacher v4 train 输入扫描 789 局、34,456 个
决策；离线复算 frozen Teacher 规则评分的选择不一致为 **0**。Teacher top-2 规则分数差不超过 2.0 的候选共有
17,844 条，按每个物理牌局 group 最多两题，确定性抽取 600 题、443 个 opaque group。额外 100 题为
500-confirmed 门槛提供 `uncertain`/跳过缓冲；扩容前后前 500 题逐项一致。题库不含赛果、wall、seed、
对手暗手、原 trajectory/group ID 或源路径；服务端在提交前不发送 Teacher 索引、分差或 group。

本地审阅服务已在 `127.0.0.1:51861` 启动，当前 confirmed 标签 **0/500**，故没有训练 checkpoint。追加 label 带
immutable queue digest，可断点恢复；`uncertain` 不进入训练。总量门槛固定为 500 confirmed、50 个 Teacher 分歧和
100 个 group，并按原始牌局 group 做 80/10/10 切分。

训练与 gate 基础设施同时就绪但未运行：fresh feature-v3 candidate MLP、hidden 128、CPU、12 epochs，只把 confirmed
人类选择作为 policy target，明确没有终局 value target；review test 没有训练器 CLI 参数。validation gate 固定覆盖率
10%/20%/30%，失败时 test 字节保持未读；test 通过仍只解锁 100 墙 Teacher 筛选，不形成超过 Teacher 或人类的结论。
## 2026-08-08：公开知识向听 Teacher v1

- 路线：只在 frozen Teacher 近分、双方均未听牌的普通弃牌中，用五面子加一对的精确标准向听数做严格门控；公开有效进张只破同向听平局。
- 前置：40 组 1/2 金牌随机手牌与通配枚举完全一致；20 墙覆盖率诊断为 24/608（3.95%）；真实回放确认从二向听严格降到一向听。
- 固定 selection：classic，`seed=202625000..202625099`，100 个物理墙、400 局四座轮换。
- 结果：candidate 103 胜、均分 +0.2675；覆盖 129/3,187（4.05%）；paired delta 95% CI `[−1.4671, +2.0021]`。
- 结论：点估计略正但未通过 `95% CI low > 0`，状态 `selection_rejected`；不部署、不产训练标签、不在同墙调参重跑。
- 推断：精确向听数应保留为状态特征，但“严格降一向听”不能单独充当动作价值；下一步应估计公开可达性、金牌/游金价值和有限视野结算，而不是扩大模型。

## 2026-08-08：参数化弃牌规则 Teacher v1

- 模型：只暴露 frozen Teacher 普通弃牌评分中的刻子、对子、相邻、隔张、等待和弃金惩罚 6 个线性权重；其余规则与响应逻辑不变。
- 训练：CPU CEM 固定 4 代 × 12 候选；两个 20 墙训练段，以较差分段均分为 fitness。
- 训练最优：两个分段 +5.175 / +1.975，表面上都优于 frozen。
- 独立 validation：`seed=202627000` 起 100 墙，93/400 胜，总分 −612，均分 **−1.53 ± 1.1566**，胜率 23.25%。
- 结论：`validation_rejected_selection_unread`；正式 `202628000..202628099` 未读取。40 墙稀疏终局回报对 6 个非连续排序权重明显过拟合，后续改用同状态配对反事实标签降低方差。

## 2026-08-08：两摸听牌可达性 SlowExpert v1

- 路线：只在 frozen 近分非听牌弃牌中，用 32 个共同分层公开摸牌场景估计两次本家摸牌内进入听牌的概率；候选不得增加向听，且概率至少提高 5 个百分点。
- 前置：10 墙覆盖 19/293（6.48%），没有向听恶化；固定真实状态的分层估计与精确枚举排序一致。
- 正式 selection：classic，`seed=202630000..202630099`，100 个物理墙、400 局四座轮换。
- 结果：106 胜，总分 +534，均分 **+1.335**，胜率 26.5%；覆盖 182/3,221（5.65%）；paired 95% CI **[−0.9195, +3.5895]**。
- 结论：方向为正但置信下界未过 0，`selection_rejected`。该特征比单步向听更有希望，但 552.82 秒／100 墙过慢且没有新标签证据；不部署、不在同墙调门槛。

### 盲主动审阅优先级（只优化采集效率，不是新强度实验）

在不修改 600 题 immutable queue 的前提下，对每题从 actor-visible 快照离线运行上述已拒绝 SlowExpert。它与 frozen
Teacher 在 **51/600** 题上选择不同；priority 将这些题排在前面，再按两摸概率优势和原始 index 稳定排序。queue
SHA-256 在生成前后均为 `9477e7a5099a46616ebbfd217fad1b08500031e21ca1f3f37ceebb4b9ad43ce6`，priority 自身
SHA-256 为 `a1a652144db48c161de6d7a5624cf7af42afde2b1b1862acea176c8267012bfd`，600 条绑定审计无异常。

网页 GET 在人工提交前不返回 priority、SlowExpert 或 Teacher 字段；提交后只显示对照，追加 label 不含两种自动答案。
服务已用该 priority 在 `127.0.0.1:51861` 重启，人工标签仍为 **0**。因为选题是主动而非随机，早期标签分歧率不得
外推总体；训练和强度门槛仍是 500 confirmed、50 个真人/Teacher 分歧、100 groups、group-held-out gate 和全新
100/400 墙实战。

## 2026-08-08：精确向听吃碰响应 Teacher v1（selection 拒绝）

对 1,236 局安全 natural Teacher 轨迹的聚合审计显示，10,061 次 response 中 frozen 选择 chi 5,389、pong
3,970、pass 仅 110；响应阶段足够自然且 Teacher 极度激进。新候选真实执行吃碰后强制弃牌，并按精确五面子向听、
直接听口面数、frozen hand shape 的唯一词典序比较 pass/chi/pong；只有严格改善 frozen profile 才覆盖。胡、明杠、
游金、金牌锁、天听和普通摸打完全冻结，不读 wall 或对手暗手。

10 墙覆盖诊断为 1/94 次候选座 response，耗时 1.74 秒；随后执行预注册 classic `seed=202632000` 起 100 墙、
四座轮换。正式覆盖 23/966（2.38%），方向为 `chi→chi` 1、`chi→pass` 5、`pong→chi` 9、`pong→pass` 8，特殊
状态违规为 0。候选 98/400 胜、总分 −207、均分 **−0.5175**；paired 95% CI **[−1.6406,+0.6056]**，未过
严格正下界，状态 `selection_rejected`。产物
`artifacts/deficiency-meld-teacher-v1/selection.json` SHA-256 为
`69588fae85e26eb2b3db05e9cf3d10be58600df753f15851f56cdd967bf317b5`。

该结果与旧响应后形分阈值族一致地说明：局部牌效不能单独识别吃碰过的长期价值。候选不部署、不产标签、不在同墙添加
有效牌或 tie-break 重跑。下一数据入口是单独的 actor-visible 人工响应审阅；现有 600 题弃牌 queue 不可修改，响应题库
须使用新版本和独立文件。

### 独立 response 人工审阅 queue v1（基础设施就绪，0 标签）

固定 Teacher v4 train 扫描 789 局／34,456 决策，筛得 7,312 个仅含 pass/chi/pong 的普通 actor-visible 响应；
Teacher 从导出快照复现不一致为 0。已拒绝精确向听候选与 frozen 分歧 221 个，按每物理局最多两题后保留 219 个并
优先排列；总 queue 为 400 题、319 groups，Teacher 动作分布 pong 220、chi 176、pass 4。

queue SHA-256 `653e1063d3d46603abf6dd2f43850a9844723f3f19c34681921395848301062d`；报告 SHA-256
`397ec6e7495212d6c5459537009b009c00af15cb91cbbe4f36eb2f5e4fd4ddaa`。独立服务已在
`127.0.0.1:51862` 启动，预提交 API 不含 Teacher／SlowExpert／group／priority；append-only label 仍为 0。
门槛固定为 300 confirmed、50 个真人／Teacher 分歧、100 groups，之后才按 group 80/10/10 切分并考虑 CPU 小型
response residual。主动排列的部分样本不得估计总体错误率。

响应训练与选门基础设施现已完成但仍未运行：standalone split loader 只接受 confirmed 普通 pass/chi/pong，并拒绝
重复、越界、私有字段及伪造 Q/value target。固定 wrapper 使用 fresh seed 202633000、feature-v3 candidate MLP、
hidden 128、CPU、12 epoch；响应标签权重 4，真人／Teacher 分歧再乘 2，仅以响应 validation 选 epoch，不加载旧
checkpoint。训练器没有 response-test 参数。validation 只比较预声明 10%/20%/30% 覆盖门；失败时 test 字节保持
未读，单测验证只调用一次 validation scanner。test 通过仍只解锁全新 100 墙四座轮换。当前人工标签为 **0**，因此
没有 checkpoint、没有 gate 结果，也没有新强度结论。

100/400 墙执行层也已预注册但没有运行：response gate report 和 checkpoint 均须 SHA-256 精确匹配；包装器只在
classic 普通 pass/chi/pong 合法集合内调用模型，hu、杠、游金、金牌锁、天听、非响应和普通弃牌全部回退 frozen
Teacher。selection 固定 `seed=202634000` 起 100 墙，严格正下界通过后才读取 `202634200` 起 400 墙 terminal。
这补齐了“人工 test gate 能判定却无法实战”的执行缺口，不改变当前 0 标签状态。

## 2026-08-08：两摸听牌 exact-DP 确认 v2（selection 拒绝）

两摸 v1 与首分歧二元随机干预都只有正点估计、置信区间跨零。本轮不调其 2.0 近分范围、5% 优势阈值或 32 场景，
而新增只做减法的确认层：v1 先提议唯一替代，随后只对该动作和 frozen Teacher 动作精确枚举两次公开未见副本的
无放回本家摸牌；第一摸后按第二摸前“已胡或进入听牌”的精确概率选择弃牌。算法不读真实墙、seed、对手暗手或未来
RNG。既有 actor-visible queue 前 100 题的只读诊断中，12 个 v1 proposal 被确认 8 个、否决 4 个，耗时 156.20 秒。

预注册 classic `seed=202636100` 起 100 墙、四座轮换已完成。3,207 次弃牌调用中 v1 proposal 184 次，exact 确认
102、否决 82，实际覆盖 3.18%，向听恶化 0。候选 104/400 胜、1 流局、总分 +170，均分 **+0.425**；paired
95% CI **[−1.4590,+2.3090]**，未通过严格正下界。候选耗时 2,031.86 秒，Teacher baseline 15.82 秒。

状态固定为 `selection_rejected`：不部署、不蒸馏、不产标签、不追加墙或重调同族 utility。TwoDraw 局部牌效族
至此冻结。报告 `artifacts/exact-two-draw-tenpai-reach-v2/selection-report.json` SHA-256 为
`3173fd8e076c6e8f94614c4eb1cba68c613176738bcbba6c1a762fc221065e27`。

## 2026-08-08：首杠机会二元干预 v1（覆盖失败，方向不支持跳过杠）

游金升级的自然覆盖审计发现，小 v4 与 5,000 墙 part-01 的 `advance_tour` 全来自合成课程，自然 Teacher 轨迹为 0，
故没有把合成频率当强度机会。真实 natural train 则有明杠 255、补杠 104、暗杠 44，因此改在每条候选座轨迹首个
Teacher 杠机会，以 0.5/0.5 随机执行杠或 frozen 非杠 fallback，随后恢复 Teacher。三种杠类型在打开结果前固定，
类型选择采用 Bonferroni family α=0.05。

全新 classic `seed=202637000` 起 100 墙得到 65 次干预，低于预注册 100：fallback/Teacher=32/33，100 个墙组
完整，审计问题 0。overall `skip-kan − Teacher-kan` 为 **−3.705**，95% CI **[−7.475,+0.065]**；暗／补／明杠
point estimate 分别为 −0.590/−0.710/−2.405，三类校正区间均跨零。状态
`selection_rejected_no_kan_training_labels`：覆盖不足不能宣称正式显著，但所有方向均不支持跳过杠，因此不追加墙、
不构造少杠候选、不产标签。安全数据 SHA-256
`c64511a1a43251af5fa3fe4485f1a3b20de1405a650703e52898ea74633d18c2`。

## 2026-08-08：公开向听—有效进张 Pareto v2（空策略拒绝）

为避免向听 v1 用大量有效进张交换少一向听，已有 `PublicParetoDeficiencyTeacherAgent` 被固定为只减法确认层：向听
必须严格下降，同时公开下一摸有效进张张数不能低于 frozen。全新 classic `seed=202638000` 起 100 墙中，3,263 次
弃牌产生 70 个 v1 proposal，但 70 个全部被 Pareto 条件拒绝，实际 override 为 0，候选与 Teacher 完全一致。

这说明当前真实访问状态中的 v1 进一向听动作系统性牺牲有效进张；v2 是结构性空策略而非可部署平局。状态
`selection_rejected`，不放宽条件、不产标签。报告 SHA-256
`7a60f963220ea5c536ec7e42e4f5e4feea6ba1e4af571f88e696b1d78092dca7`。

## 2026-08-08：Teacher 完全并列公开进展消歧 v1（selection 拒绝）

源码级复核抚州 commit `53f73d2495c4a49790ab2af7c064906cd5d9fd2b` 后发现，其生产 hybrid 的网络几乎不
推翻 lookahead reference；主要差异来自 reference 最高分并列时的 `progress_static_risk`。据此实现厦门独立翻译：
只在 frozen 最高分完全相等的普通弃牌中，选择向听不高、公开有效进张不低且至少一项严格改善的 Pareto 动作。

10 墙只读覆盖为 78/332（23.49%），随后按冻结协议运行全新 classic `seed=202638300` 起 100 墙。正式覆盖
814/3,318（24.53%），1,742 次最高分完全并列；814 次改动均保持向听不变，累计增加 5,879 张公开有效进张。
候选 103/400 胜、总分 −32、均分 **−0.08**，paired 95% CI **[−3.5633,+3.4033]**。候选耗时 944.51 秒，
Teacher 15.62 秒，慢 60.46 倍。

未通过正下界，且公开进张的大幅增加没有转化为长期收益。本族 `exact score tie + shanten/ukeire Pareto` 冻结：不部署、
不蒸馏、不追加风险或巡数门控。报告 SHA-256
`c4be7590ff52b5efcb52214329ad301482fa02224edc8e5282febdaea43e8adf`。

## 2026-08-08：完全并列盲态二选一数据入口 v1（题库与训练器就绪，0 标签）

exact-tie 自动 progress 候选未过实战门槛，但它暴露了 Teacher 的 52.5% 完全并列率。为获得真正的 tie-break 监督，
新增 pairwise-only 人工审阅：只展示 Teacher 默认牌和公开进展替代牌，左右顺序按 hash 决定；提交前 API 隐藏双方
身份、group、margin、原 item、赛果、wall、seed 和暗手。

从 immutable 600 题 queue 复算：600 题均为 top-2 完全同分，候选分歧 253；按每个原物理局 group 最多一题后固定
227 题／227 groups，source issues 0。queue SHA-256
`0006b2603bcb32332c4d08b76b5b310d3fde046d6402bf41b944dc9467f747bc`。服务已在 `127.0.0.1:51863` 启动，真实
GET 审计显示只返回两个动作，Teacher/candidate/group/outcome/wall/seed 均未泄漏；当前 0/227。

数据门槛固定为 100 confirmed、20 个 candidate preference、75 groups。通过后按 group 80/10/10 切分；fresh
feature-v3 hidden-64 CPU MLP 只对展示动作对做 cross-entropy，未展示动作不是负例，validation 选 epoch，test 训练时
不读。当前没有标签，因此没有训练 checkpoint，也没有新强度结论。

## 2026-08-08：完全并列 source-world 小模型与高支持因果确认（全族拒绝）

为绕开 0 个人工标签，只在 frozen Teacher 最高分完全并列处使用训练期私有模拟分支：同一自然访问隐藏世界中强制
每个并列动作，随后恢复 Teacher，导出仅含 actor-visible state 和终局目标。100 墙 pilot 有 398 状态；扩大到 900 墙
得到 3,550 状态／897 groups／11,501 个分支。feature-v3 五个 hidden-64 MLP 仅在全员一致时覆盖，外层 validation
为 **−1.418**，95% CI **[−3.550,+0.715]**，test 未读。

加入 59 维显式 Teacher 结构分量、精确向听和完整公开余牌计数后，feature-v4 在全新 100 墙为 **+0.075**，95% CI
**[−2.222,+2.371]**。再固定同一暗手、重排四次未来墙：500 墙／1,972 公开决策／25,340 分支，平均标签模型在
97 validation groups 为 **−0.091**，95% CI **[−1.380,+1.198]**，内部排序准确率仅约 51%–52%；final test 未解析。

既有高支持随机弃牌数据中，v2 单点 HT 诊断曾为 +1.920、区间跨零。为排除稀疏 uniform-action 支持，冻结 v2 五个
checkpoint，在全新 `seed=202644000` 起 100 墙第一次一致分歧处 0.5/0.5 二元随机。得到 229 次干预，
candidate/Teacher=113/116；低于预注册 250，且 HT **−5.210**，95% CI **[−11.311,+0.891]**。状态
`selection_rejected_no_deployment`：不做整局 agent、不打开 final test、不继续同族调参。安全数据 SHA-256
`6a4f9aceafba450a0958c99d7c0964d1fd65c8d417e4c10f250a3cd6160bf53e`。完整推理见
[协议](exact_tie_source_world_and_causal_v1_protocol.md)。

## 2026-08-08：Teacher-clone reference-KL 小 MLP PPO v2（selection 拒绝）

旧 margin-5 anchored PPO 的 3,000 个 actor-visible 决策审计显示，iteration-8 residual 最佳替代 gap 最大仅
`0.001572`，与固定 margin 相差约 3,000 倍；严格零实战分歧源于 actor 没有穿透 prior。同期修复 rollout 报告把
所有非流局误计 candidate win 的错误。基于 96.95% Teacher 一致率的 feature-v3 hidden-128 candidate MLP，冻结训练
起点 reference，使用 KL 权重 0.05、policy-head 学习率倍率 10，在三家 Teacher 上按预注册运行 4×1,024 classic 局。

正式 artifact 补齐输入模型、源码树、全部超参数与逐轮 seed 的 provenance；两次完整重跑的四个 `state_dict` 逐张量
相同。45,102 个候选决策动作计数完整。final KL mean `0.0036966`、起点 argmax 分歧 `1.0415%`，mechanics gate
通过。随后唯一打开 seed `202647500..202647599` 的 100 个物理墙、四座轮换：候选 99/400 胜、总分 −54、均分
**−0.135**；paired 95% CI **[−1.9748,+1.7048]**。

状态 `selection_rejected_configuration_frozen_not_deployed`：未解锁 250,000 局，不部署，不在该 100 墙调 KL、倍率、
epoch、规模或挑 checkpoint。结论不是“PPO 永远无效”，而是这组小样本 reference-KL 配置没有越过 Teacher 的可靠证据；
后续若重开 RL，必须从新候选定义和全新墙集开始，不能把本 selection 当调参 validation。

## 2026-08-08：IJCAI-2026 国标数据／工程只读审计（不接入）

在人工 exact-tie 盲评仍为 0/227 时，定向检索公开厦门牌谱，没有发现同时满足 classic 规则、训练授权和完整可重放
历史的来源。进一步只读审计 Hugging Face `Dannibal/ijcai-mahjong-round2`、Botzone 2026 官方竞赛页与
`SuuTTT/IJCAI-mahjong` commit `690848613f34b3ebc0a3547070497e07526c81ea`：它们均面向 MCR 国标麻将，动作空间、
手牌流程、番种和结算与厦门不兼容；原始格式还可能含牌墙、随机种子和玩家标识。代码仓库根目录未发现覆盖自有代码的
整体许可证。

状态为 `research_only_rejected_for_ingestion`：没有下载数据或权重，没有复制代码，没有启动跨规则预训练。仅采纳可独立
实现的实验纪律——三随机种子稳定性、合法动作概率集成、duplicate／matched 评测、计分器黄金测试、小网络优先和扩大
样本后的假阳性复核。这一审计不产生候选 checkpoint，也不改变人工标签门槛；详见
[训练数据来源审计](data_source_audit_2026-08.md)和[开源工程地图](research/open_source/landscape_2026-08.md)。

## 2026-08-08：低分差 top-2 因果 residual v1 mechanics pilot（通过）

新增与旧 fixed tie-break、TwoDraw 和 source-world 分支不同的高支持数据入口：每条候选座轨迹只在第一次 Teacher 前两名
弃牌分差 `<=2.0` 时，以 0.5/0.5 执行 top-2 或 top-1，随后恢复 frozen Teacher。数据记录精确 propensity；动作对必须
能从 actor-visible 导出状态复算。pilot 在结果前固定 25 墙 `202649000..202649024`、四座轮换和行为 seed
`202649025`，仅允许决定工程可行性与正式样本量。

100/100 局均有一次干预，alternative/Teacher=57/43，25 墙组完整，终局候选分差范围 `[-44,108]`，审计问题 0；
93 次完全并列、7 次 margin `(1,2]`。墙级 HT 标准差 **38.1226**。pilot 点估计 `−1.1` 不用于模型或阈值选择。
机械门槛通过后，依据方差固定正式 train 4,000 墙、validation 2,000 墙；另留全新 2,000 terminal seed，只有
validation 通过才采集。为控制磁盘和信息边界，正式数据改为每局一条 `xiamen-low-margin-top2-causal-v1` 瘦身记录，
不保存 seed、墙、他家暗手或无关决策。当前尚未完成正式采集或训练，没有强度结论。

## 2026-08-09：低分差 top-2 因果 residual v1（validation 拒绝；GPU 链路优化）

正式 train 4,000 墙和 validation 2,000 墙均采集完成并通过 actor-visible、完整四座 group、propensity、opaque 顺序和
chunk SHA 审计。train 为 16,000 条记录／15,901 次干预，validation 为 8,000 条／7,945 次干预，train/validation
group 重合为 0。

训练前定位到原流程的主要浪费：feature-v4 在每个 epoch／成员上由 CPU 重复计算，GPU 基本空闲。现改为一次 CPU 构造
39,074,321-byte 特征 cache，全部训练 tensor 常驻 RTX 2080 Ti，三成员复用；validation 也一次构造、三成员复用。监控由
CPU 单核瓶颈变为 GPU P2、约 34% compute utilization，工作显存约 316 MB。低显存占用符合 4.8 万参数小网络的规模，未为
填满 22 GB 而改大网络或引入 Transformer。

三成员 train 内选出的 epoch 为 5／6／22。validation 的 1%／2%／5%／10% 覆盖门墙级点估计依次为 -0.093、+0.1005、
+0.1000、+0.38925；Bonferroni 下界依次为 -0.6743、-0.6192、-0.9351、-0.9393，全部不大于零。机械支持门均通过，
但效果门全部失败。因此状态固定为 `validation_rejected_terminal_not_collected`：terminal 未创建／未读，不部署、不运行
100 墙实战，也不在同一 validation 上调网络或阈值。

训练报告 SHA-256 `b5adced5412d7d83d50297991b53d8618e212c50bbeb10b6871babe08038e916`；validation 报告 SHA-256
`b6873839660d2587e20d4ceab3813e39a39e1a1f508ebd79de88b0c13ad3b715`；feature cache SHA-256
`b6cff35d0fbdf4025d725517fda0b69ab8c494decd6bac8eeed628e0dcc0d2b7`。本轮说明算力链路可以提速，但不能替代低噪声标签和
稳定条件效应；下一模型族必须先提出不同的可识别性假设并使用全新 validation，而不是继续消费本轮 holdout。

## 2026-08-09：低分差 top-2 同墙控制变量 residual v2（降噪成功，策略仍拒绝）

只用 v1 train 发现并验证一个期望为零的 control variate：当前随机动作与同物理墙另外三条座位轮换的终局结果独立，故可用
另外三局均分消除墙难度噪声。v2 训练 target 固定为 `本局分数 + 另外三局均分`，三 fresh hidden-64 成员使用 CUDA；gate
改为三成员最小 effect，防止单 seed 乐观。epoch 7／5／8、四覆盖阈值和 train 最优控制系数均在采集新 validation 前冻结。

全新 `202658000..202659999` 2,000 墙得到 8,000 条／7,946 次干预，分配 3,960/3,986，审计问题 0。独立
validation 上，control 把 1%／2%／5%／10% 门的标准误从 0.204／0.270／0.396／0.489 降为
0.095／0.142／0.244／0.351，说明方差方法确实泛化；调整后点估计为 +0.102／-0.075／+0.119／+0.581，但四个
Bonferroni 下界仍为 -0.135／-0.430／-0.489／-0.296，全部拒绝。

结论是 `wall_control_v2_validation_rejected_terminal_not_collected`：terminal 未创建，不追加墙、不部署、不重调。本轮把
“估计器太吵”和“候选信号太弱”区分开了；低分差 top-2 residual 家族冻结，下一路线必须改变决策信号而非继续扩大模型。
训练报告 SHA-256 `24f80ca8efb55cfdbf916f314e21a5899e53caceabe6e6a66cfd527e7ee3d840`，selection SHA-256
`a593d4e3159872bf215747f5e5dd2bed894e31c90d0daf43bbf7e949ea65c79c`。

## 2026-08-09：低分差 top-2 可解释因果错误地图（无稳定类别）

在 residual v1/v2 都拒绝后，只把已经消耗的 v1 train、v1 validation 和 v2 validation 共 8,000 墙用于探索，不读取两套
terminal。预先冻结 45 个 actor-visible 类别：阶段、margin、庄家、牌类转换、手牌面数、局部连接、摸切、公开剩余、端张、
同花色牌号及有限 exact-tie 二重交叉；估计器固定使用同墙 control `c=-2.5`。

36 类在三个分区都有足够支持，22 类三个点估计同为正，但全族 Bonferroni 后无一正下界。最接近的“公开剩余同面张数相同”
pooled 均值 +1.194、校正下界 -0.393；“不同数牌花色”均值 +0.896、下界 -0.179；“两者均非端张”均值 +0.695、
下界 -0.105。它们最弱分区均不足约 1.04 个标准误，不能从 45 类中事后挑选。

状态 `no_stable_interpretable_category_top2_family_frozen`：不追加类别、不收集新墙、不训练 gate。报告 SHA-256
`736e58abaf534a214a7e55a8da64cf973be01be4a598521cafa86f0441d7b1d2`。这完成了对现有 top-2 随机数据的最后一次
可解释利用；下一可识别信号必须来自盲态真人纠错或结构不同、先经校准的局部 oracle。
