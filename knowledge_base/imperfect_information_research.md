# 不完全信息动作价值：研究笔记

更新时间：2026-08-07。

## 可迁移结论

- [DeepStack（Science 2017）](https://pubmed.ncbi.nlm.nih.gov/28254783/) 将局部递归推理、分解与学习到的直觉价值结合，用于隐私信息博弈；它说明“已知一副暗牌后的 rollout”不能直接替代信息集价值。
- [ReBeL（arXiv:2007.13544）](https://arxiv.org/abs/2007.13544) 将公共状态与 public belief state 作为搜索/学习对象。其两人零和收敛结论不直接适用于四人、一般和的厦门麻将，但“按公开历史维护隐变量分布”的建模方向可迁移。
- [Student of Games（arXiv:2112.03178）](https://arxiv.org/abs/2112.03178) 也以 public belief 与 counterfactual value 网络衔接学习和搜索。对本项目而言，先把信息集目标做对，比直接照搬大型 CFR/MCTS 更现实。

## 项目决策

1. 反事实标签必须近似
   \(Q^\pi(I,a)=\mathbb E_{h\sim p(h\mid I)}[R\mid I,a]\)，其中 `I` 是本家可见的信息集；不得以某一真实暗手 `h` 的结算分当部署真值。
2. 当前 `--belief-resample` 是 `p(h | 公开事实 + 本家手牌)` 的先验抽样，已修正“特定暗世界”泄漏，但尚未充分利用对手弃牌/副露行为。
3. 下一阶段只实现**可审计的近端行为条件化**：对每位对手最近可重建的弃牌决策，在重采样暗手上计算冻结对手策略选择该弃牌的似然，并以粒子重采样加权。不能把这称为完整历史后验。
4. 完整历史后验需要顺序粒子过滤：从发牌起维护粒子，逐个公开动作以相应对手策略的动作概率更新权重，并同时条件化本家的私有摸牌轨迹。这需要扩充 collector 的内部观测轨迹，完成前不修改公开训练特征或网页状态。
5. 四人一般和游戏不宜宣称纳什收敛。所有版本仍以新牌墙、四座轮换、冻结对手池的配对净分置信区间决定是否晋升。

## 验收门槛

- 数据：粒子权重的有效样本量（ESS）、拒绝/跳过率、每状态 belief world 数；均不能泄露墙、他家暗手、种子。
- 离线：按物理牌墙切分后的动作 Q Huber、MAE、argmax；Q 头和 policy 头分别汇报。
- 实战：候选 policy 选牌、Q 选牌和基准各自使用同一组新牌墙做四座轮换；不得使用训练或调参牌墙。

## 2026-08-06：顺序粒子过滤核心（尚未接入 collector）

已新增 `xiamen_mahjong.belief.SequentialParticleBelief`，它接收不透明的运行时粒子和逐个公开
观察的 transition/likelihood 回调，维护标准化后验并安全记录 `ESS`、权重熵、log evidence、
重采样次数、零似然粒子和 proposal failure。粒子、随机种子和任何暗牌均没有序列化接口。

用二类隐藏对手风格的可枚举玩具博弈校准：在观察 `claim`、`pass` 后，粒子后验与精确贝叶斯
结果 `4/13` 一致；低 ESS 时 systematic resampling 仍保留重采样前退化诊断；零总似然时状态
原子回滚。这个结果只验证过滤数学与安全边界，**不代表厦门麻将已经有完整历史后验**。

已开始 collector 内存 trace：每个候选决策快照绑定本家初始手牌/花牌、私有摸牌（含补花与
补杠后的牌）及本家动作；每条记录按公开 action count 定位，暗杠牌面等不会因公开日志脱敏而
丢失。trace 不含他家暗手或墙顺序，且没有 JSON payload/metadata 路径。

在 `core` 档已加入严格 replay oracle：它从开局后的私有状态重新推进，候选只使用自己的 trace，
对手以冻结策略的平滑行为似然加权（有可用 neural/action 分数时用温度 softmax，否则使用确定性
fallback），并逐条比对生成的公开 event。固定 `seed=953` 的 13 个候选快照均精确复现；对手暗杠
作为“发生暗杠”的公开粗粒度事件，对可能牌面求和，绝不把牌面写回公开记录。replay audit 只返回
提议数、接受数、接受率、平均 log-likelihood 与拒绝类别；不会返回粒子本身。

同时验证了一个不可忽略的反例：仅在开局随机重分配未知手牌/牌墙、再拒绝不合法历史的朴素 proposal，
在该局 5 个公开 event 的前缀中只接受 6/200 个粒子，在 13 个 event 的前缀中 0/200。它会迅速
退化，**不得接入 collector 或替换现有 `--belief-resample`**。下一步改为按事件约束的序列 proposal
（在目标弃牌/副露发生时先满足所需手牌与私有摸牌），并以 ESS/接受率门槛校准后才允许产出训练数据。

## 2026-08-07：事件约束修复 proposal（审计结果：拒绝作为训练 belief）

实现了仅限 `core`、只读审计的 `core_public_history_constraint_repair_v0`。它仍先从“本家私有
手牌/摸牌 trace + 公开事实”采样；随后在重放每一个公开事件时，若他家的已观察弃牌、吃、碰或明杠
因随机暗手而不合法，便只在**未知牌墙与非本家暗手**之间守恒交换所需牌。本家手牌、私有摸牌预约、
副露、弃牌和所有导出记录绝不修改；暗杠保持公开面值未知的边缘化。该实现不会返回粒子世界，也没有
接入 `collect_counterfactual_action_value_trajectories`、JSONL 或网页选牌路径。

固定单元夹具（`core`、seed 953、9 个公开 action、本家座位 0、100 个粒子）中，朴素拒绝 proposal
接受率为 **0/100**；修复 proposal 为 **94/100**，平均 **3.38** 次隐藏牌交换。这验证了原先的主导
失败确为“公开对手动作在随机暗手中不合法”，而非 replay 或本家私有 trace 的错误。

但其冻结 Teacher 行为似然的加权 ESS 只有 **3.55/94 = 0.038**。更长前缀的初步探针亦可降至约 1%，
说明高接受率只是把拒绝退化变成了严重的权重退化；而且该交换 proposal 尚未计算自身密度，不能把该
ESS 解释为严格 posterior ESS。因此它**不通过**进入 multi-world 价值数据的门槛，不能声称是 sequential
posterior，也不能替换现有局部 SIR 或网页默认 run4。

下一项研究不是继续扩大该修复采样，而是定义有显式条件密度的逐事件 proposal（含抽牌/本家私有摸牌、
弃牌、公开副露与行为模型温度校准），在可枚举小牌墙上同精确 posterior 对照；只有接受率、proposal
修正后的 ESS 和 posterior 误差均预注册达标后，才考虑接入独立的多 world value 消融。

为避免把“高接受率”误当作正确的条件化，`SequentialParticleBelief` 现允许每个运行时粒子提供初始
重要性权重 `p/q`。以微型物理牌墙 `0,0,1,1`、本家明示持 `2`、他家两张暗手、公开观察“弃 1”为校准：
精确先验为 `P(00)=1/6, P(01)=2/3, P(11)=1/6`，行为似然为 `0, 0.8, 0.6`。故意偏置的合法约束
proposal `q(01)=3/4, q(11)=1/4` 在用 `p/q` 修正后精确恢复 `P(01 | 弃1)=16/19`。这只验证重要性
修正 API 和带重复物理牌的枚举契约；尚未给真实修复 proposal 赋予密度，也不改变上述拒绝结论。

## 2026-08-07：不改写暗牌的 sequential SMC 基线（审计结果：淘汰）

新增 `core_public_history_sequential_smc_v0`，只用于 core audit。它不执行任何 tile repair：每个粒子
从本家可见 setup prior 采样，故初始 `p/q=1`；随后把公开日志按“一个玩家动作 + 该动作自动生成的 draw/result”
分组逐步重放，以增量冻结行为似然更新权重，并在预重采样 ESS 低时使用标准 systematic resampling。这个分组
避免把一次弃牌导致的自动摸牌误判为 prefix mismatch；粒子世界、随机状态和重放游戏均不导出。

同一 core seed 953、9 个公开 action 的夹具中，200 粒子在仅完成 4 个公开 action 后即全部不一致；1024
粒子可走完全程，但最小预重采样 ESS 为 **1/1024**，随后重采样导致最终表面权重重新均匀。由此可见，标准
SMC 解决了“全前缀一次性 rejection”但没有解决稀有公开弃牌/副露造成的极端退化。它也未通过数据门槛，
不会接入 collector；下一步必须是具有显式密度、且在可枚举小牌墙上校准 posterior 误差的条件 proposal，
并用未重采样 ESS 而不是最终均匀权重作门槛。

## 2026-08-06：受限的最新弃牌局部 SIR（默认关闭）

作为完整序列 proposal 前的受控中间步骤，新增 `latest_normal_draw_discard_sir_v1`。它只作用于
本家处于 response、且最后两个公开事件为“对手 normal draw → 同一对手 discard”的状态：
先从已有 actor/public prior 采样当前世界，公开地逆转最后一次弃牌，再以冻结策略对该弃牌的行为似然
加权并重采样。它不查看源局暗手、墙顺序或对手私有摸牌；输出仅含 ESS/一致率等聚合指标。

在 core 的 seeds 930–939 中抽得的 32 个此类 response 状态上，每状态 32 粒子：结构一致率为 1.0。原始
行为似然（幂次 1）平均 ESS 比例 0.156、最低 0.035，过度相信未校准的 Teacher fallback；似然幂次
0.25 后平均为 0.669、最低 0.560。因此 collector 仅在显式启用、幂次 0.25、ESS 比例至少 0.5 时
采样；任何不支持/低 ESS 状态都跳过，绝不伪装为普通 `--belief-resample` 数据。

该机制仍是**单事件局部条件化，非完整 history posterior**；仅可作为独立标记的数据来源，尚未训练、
评测或晋升任何模型。

classic 仅接受能由公开信息精确逆转的普通前缀，并显式跳过游金、天听和早局状态。seeds 930–939 的
34 个合格 classic response 状态上，power=0.25 的平均 ESS 比例为 0.425（最低 0.094）；将幂次降为
0.10 后为 0.583（最低仍有 0.094）。因此经典档实验使用 `power=0.10, minimum_ess_fraction=0.20`，
低 ESS 状态跳过；这只是有审计门槛的局部数据构造，绝不是对复杂地方规则的完整倒放。

首次 classic 采集（40 个物理牌墙、四座轮换、每动作 3 个 belief world、75% Teacher / 25% 冻结 run3）
导出了 124 个 response 决策、822 个动作分支和 372 个条件化 world；所有 world 通过门槛，平均预重采样
ESS 比例为 0.960。它从 run4 微调得到的候选在第一组 200 墙上相对 run4 为 +1.04 ± 0.76，区间
[-0.46, +2.53]；独立的 400 墙终检为 -0.14 ± 0.59，区间 [-1.29, +1.01]。因此局部 SIR 没有证明
强度提升，不能晋升；高 ESS 也只能证明 Monte-Carlo 没有退化，**不能证明行为信息足以改善决策**。

经典档还发现一项必须显式审计的公开约束：重采样对手手牌后，公开河中的荣牌可能触发“强制跟打”，
令已观察的普通弃牌在该粒子中变为不合法。collector 现把这种 `observed_discard_illegal` 视为拒绝，
并在摘要报告平均结构一致粒子率；绝不通过读取源局暗手来保留这些粒子。

该数据的每动作标准误均值仍为 18.55 分，接近 28.47 分的平均动作价值跨度。训练器因此新增可审计的
`--action-value-confidence-z`：反事实软标签可使用 `softmax((Q - z × stderr) / temperature)` 的逐动作
下置信界，而不是仅按照平均标准误下调整条样本。默认 `z=0` 严格保持原目标；该变体会与原始候选使用
同一份冻结数据和新牌墙筛选，结果出来前仍不得视为有效改进。

固定同一份数据的 `z=1.0` 候选在全新 200 墙筛选中为 +1.04 ± 0.75，95% 区间
[-0.44, +2.51]，与均值目标候选几乎相同且仍跨零；它不进入 400 墙终检。该反证表明在仅 124 个
状态、每动作约 18.55 分标准误的条件下，改变标签的风险厌恶形式并不是瓶颈。后续优先增加每状态
world 数和有效 response 覆盖，再重新做独立选择。

## 2026-08-07：面向人类强度的外部路线校验

新一轮文献核对的锚点是 [Suphx](https://arxiv.org/abs/2003.13590)：其公开描述将监督预训练、深度
自博弈强化学习、global reward prediction、仅训练期可见的 oracle guiding 与运行时策略适配结合，报告在
日麻平台上超过绝大多数顶尖人类。它不是厦门麻将的可直接复用实现或强度证明，但明确说明当前项目的
“几百个反事实 response 标签 + 小规模 PPO”与人类级数据规模/训练信号存在数量级差距。

下一条可证伪工程路线据此调整为：先保留 actor 的本家/公开特征与规则合法动作掩码不变；在**仅内存、
仅训练期**的 PPO rollout 中，构造独立的 privileged critic（可读取模拟器完整状态）来降低终局净分
advantage 方差。critic 输入、权重和特征不得进入网页 agent、`policy-value.pt` 的 actor 路径、JSONL 或
replay export；它只可作为训练期基线。该选择还与 [Meowjong](https://arxiv.org/abs/2202.12847) 的
“先监督、再针对主动作自博弈”流程一致。实施前需要先证明：关闭 privileged critic 时与旧 PPO 完全等价，
开启时不会把私有状态写入任何可部署或可导出对象；之后再比较相同 rollout 预算下的 advantage 方差与
未见牌墙强度。

core 冒烟已验证该边界：16 局、8 路 batch、32 隐层 critic、一次 PPO 更新生成 143 个 actor 决策；
保存的 `.pt` 仅含 actor `state_dict`，报告中不出现 `wall`、`opponent_hands` 或 `privileged_features`。
这不是强度实验；critic 参数在训练进程结束后销毁，后续对照将以同一 actor 起点和 rollout 预算检查其
是否实际降低 advantage 标准差并带来独立牌墙增益。

## 2026-08-07：配对动作价值不确定性

反事实 collector 在一个 rollout replicate 内会从同一个 actor-visible belief world 出发，对每个合法
动作各强制一次，再使用同一冻结对手配置继续。因此动作回报并非独立；直接用各动作的边际标准误来判断
“吃/碰/过哪个更好”会浪费 common-random-number 的协方差。新数据格式额外导出
`action_value_gap_stderrs`：以该状态的均值最优动作为锚，记录 `Q(best)-Q(action)` 的配对标准误，仍只
包含聚合分数、绝不包含 world、暗手或牌墙。

真实 classic 一墙、4 个 response 决策、每动作 6 个 world 的校准中，边际动作价值标准误均值为 18.58 分，
配对差值标准误为 12.13 分。训练器可通过 `--action-value-pairwise-confidence-z` 将未显著的差值收缩为
平局；z=0 与旧均值 softmax 完全一致。正式数据必须继续报告两种误差，只有配对误差确实更小且按牌墙
切分的验证集足够大时，才进入候选训练。

两组独立墙共得到 188 个 response 状态（106/53/29 的 train/validation/test）；平均边际误差为
14.33 分、配对 gap 误差 9.96 分，说明协方差修正确实有效。尽管如此，pairwise z=1 的 policy 候选在
22,980,000 起 200 个新牌墙上相对 run4 的结果为 +0.84 ± 0.70，95% 区间 [-0.53, +2.21]。这不足以
进入 400 墙终检：更好的静态反事实标签不能替代足量的 on-policy 学习与长程 credit assignment。

## 2026-08-07：特权 critic 的首个强度筛选

训练期 centralized critic 在 3 × 1,024 局、75% Teacher / 25% 冻结 run3 的 PPO 中保持了既定隔离边界：
actor checkpoint 没有 critic 权重，导出 metadata 也不含墙、对手手牌或 oracle feature 名称。该 run 中的
advantage 标准差从第 1 轮的 0.591 变为第 3 轮的 0.555，但这不是 critic-on/off A/B，不能当作降方差证据。

真正的外部筛选也没有支持继续扩大：三个 PPO iteration 在同一 100 墙组上分别为 +0.545 ± 0.583、
−1.285 ± 0.834、−0.223 ± 0.740，三个 95% 区间均跨零。故不进入 400 墙终检。下一个最小可证伪实验
应在**相同冻结 actor、相同牌墙、相同对手抽样**上仅切换 critic，记录 reward-minus-baseline 的方差；若无
稳定下降，就停止在该 critic 路线投入更多 self-play 预算。

该最小 A/B 已经否定当前路线：512 局独立校准后，在另一组 512 局、5,704 个逐条相同的 actor 决策上，公开
value baseline 的残差 std/MSE 为 0.513/0.263，特权 critic 为 0.591/0.350，MSE **恶化 33.0%**。因此这不是
“方差稍降但强度未升”的情况，而是 critic 本身尚未比公开 value 更准确；暂停该配置的 PPO 扩容。保留
`measure_ppo_baseline_variance.py` 作为后续任何 oracle baseline 的硬性门槛：先通过固定策略 A/B，才允许进入
强度训练与未见牌墙筛选。

## 2026-08-07：VRPO / 动作价值路线的边界

[VRPO](https://arxiv.org/abs/2605.19235) 指出不完全信息自博弈中的方差不只来自 value baseline，还来自后续
随机动作；其 Q-boosting 使用 centralized action-value critic 和 Expected SARSA(λ) 对动作分布取期望。这一
结论与本项目反事实 rollout 的 common-random-number / paired-gap 观察相符，但**不能直接照搬**：论文中的
centralized Q 可见训练全局状态，而厦门网页 actor 只能使用本家与公开信息；刚完成的 oracle critic A/B 也已
证明当前小数据实现甚至不如公开 value baseline。

因此下一步不是宣称实现 VRPO，而是最小可证伪前置门槛：在既有、按物理牌墙隔离的 paired action-value 数据上
单独训练项目已有的**公开信息** `action_value_head`，只看未见墙的 Q MAE / argmax 排序是否优于零初始化 head。
若这个公开 Q 连离线排序都不能稳定改善，就不应把它接入 PPO 或尝试 Expected-SARSA；若通过，才设计 target
network、冻结行为策略和 actor-on-policy 校准的独立实验。

门槛实验已执行且未通过。实现时先发现 `action-value-weight=0` 会意外把 Q regression 的样本权重同时置零，
现已拆成独立的 regression sample weight 并加单测。修复后公开 Q 的隔离 MAE 的确从 30.85 分降到 22.76 分，
但 29 个留出状态的 Q argmax 从 55.2% 下降到 48.3%。数值回归改善却没有可靠的决策排序，不能作为 Q-boosting
或 Expected-SARSA 的输入；当前瓶颈是每个公共状态的有效、按墙隔离的动作价值数据量，而非再换一个 RL loss。

第一轮独立扩容（120 墙、356 个 response 状态）后，Q 的联合留出集达到 103 个状态：MAE 从零 head 的 29.79
分降至 19.85 分，argmax 从 41.7% 提高到 50.5%。但严格按状态配对的差是 +8.7pp，95% CI [-7.5, +25.0]，仍不
足以排除偶然性。因此公开 Q 仍不影响 policy 或网页；下一轮优先继续增加**物理牌墙数**而非重复同一墙的 rollout，
以扩大信息状态覆盖并保持训练/验证/测试独立。

第二独立 120 墙组使 Q 测试增至 146 个状态，离线配对排序终于为 +13.7pp、95% CI [+0.6, +26.8]。为保持行动
分布支持一致，只实现 response-only Q：弃牌/摸牌后仍走原 policy。可是在最终的 200 个全新物理牌墙上，
response-only Q 相对 run4 的净分为 -1.88 ± 1.28，95% CI [-4.39, +0.63]。这比离线 metric 更强的反证：local
latest-discard SIR 的动作价值仍有系统性条件偏差或缺失长程交互，不能通过追加样本修复。暂停该数据构造、Q 选牌和
VRPO 接入；下一轮必须改变训练信号的构造，而不是再增加同一构造的数据。

## 公开后状态输出：只监督已执行动作

福州项目的训练报告提供了一个有用的反例和工程模式：其纯 counterfactual action Q / 原始 PPO 都未成为可部署
结论，而较强候选是 policy、规则 lookahead、预期得分、获胜率和风险的保守混合。其规则并非厦门规则，故本项目
不复用其代码、参数或数值；只采用可证伪的数据边界：终局标签必须先绑定真实执行动作，不能扩写为同一隐藏牌墙
上未执行动作的标签。

当前语料用 `executed_index` 与 Teacher `chosen_index` 分离，行为 index 只引用既有合法动作列表。新
`afterstate_score_head`、`afterstate_win_head` 和 `afterstate_opponent_win_head` 的首轮训练仅更新这些新 head，
保持 run4 编码器与 policy logits 冻结。它在 120 个墙、四座轮换的未见测试决策上把终局净分 MAE 从零预测的
39.66 分降至 34.78 分，两个概率的 Brier 均为 0.172；这足够作为“标签含有公开可学习信息”的前置检查，
却完全不足以比较未执行候选。

因此该 checkpoint 明确为 `diagnostic_only_not_authorized_for_action_selection`。下一项数据实验不是扩大旧的
local-SIR，而是记录温和随机化行为的已知 action propensity、按物理牌墙切分并检查每个候选的支持度；之后仍须把
afterstate 信号限制为小幅、policy-prior 锚定的混合，并由新的 200/400 墙实战门槛决定去留。

### 单点干预替代持续探索

已知 propensity 本身不够：若每一步都做 epsilon 探索，某动作之后的终局回报属于探索策略而非部署的 run4。
因此 collector 采用**每局一个随机决策点**：随机点之前和之后严格运行 run4，只有该点以
`(1-ε) × run4 + ε × Uniform(legal)` 采样，并记录被执行动作的精确条件概率。每局至多一个随机点，故它的
终局结果是该动作在 run4 后缀下的无分支、长程样本；仍只导出本家/公开观察和概率，不导出墙或随机种子。

首轮 240 墙、960 局四座轮换在 train/validation/test 分别获得 419/95/90 个干预点（最小 propensity 约
0.025）。冻结 policy 的公开 afterstate head 在测试上的终局净分 MAE 为 32.38 分，优于零预测 40.47 分；
己方及对手胜率 Brier 均为 0.195，相对于测试胜率常数基线约 0.222 有改善。这仅通过“能否从正确后缀的已执行
动作学到信号”门槛；90 个测试点不能证明全候选 rank，也没有启动任何网页选牌或实战候选。

为避免弃牌占满干预预算，collector 进一步支持 `--intervention-phase response`。它仅把 response 作为单点
干预的序号空间，所有弃牌以及干预后的动作仍用 run4；独立 240 墙组得到 437/56/116 个 train/validation/test
response 干预。测试覆盖吃 54、碰 34、过 23、胡 2 和明杠 3。相应 afterstate 模型的 score MAE 由零预测的
34.28 分降至 30.84 分，己方／对手胜率 Brier 为 0.182/0.183（常数基线约 0.188）。

这些是正确目标策略后缀的 value/risk 校准，而不是候选动作排序。验证集 56 个状态尤其不足以确定融合权重；
下一步应以多个独立 seed 估计预测离散度，并只检查 policy-prior 锚定的保守偏移，最后才让全新 200 墙实战决定
是否值得继续。
