# 厦门麻将深度训练：问题分解、推理链、候选与失败路线

版本：2026-08-07
性质：研究决策文档，不是已完成强度声明
适用规则：本仓库 `classic` 厦门麻将规则档位
部署基线：冻结 `HeuristicTeacherAgent`

> **2026-08-08 资源约束更新：** 本文保留完整路线空间作为研究地图，但不再代表近期执行优先级。
> 当前机器与数据不再训练 Transformer、attention 或完整历史 GRU，也不再扩大同源 Teacher 百万级数据。
> 实际执行以[《厦门麻将 AI 资源受限训练计划》](resource_constrained_training_plan_2026-08-08.md)为准：
> SlowExpertTeacher → 结构化 linear/GBDT/小 MLP residual → 1%–5% 高置信 override → 100/400 墙验证。

## 0. 结论先行

当前最合理的主线不是“再换一个更大的网络”，也不是“把 PPO 多跑几天”，而是一个分层的过渡系统：

```text
规则与评测事实源
  → 找出 Teacher 真正会犯错、且能被独立验证的局部决策
  → 结构化公开状态编码 + Teacher-anchored residual
  → DAgger / 争议状态再标注，修复候选自己的状态分布
  → 有支持度的随机干预或更强局部 oracle，提供 Teacher 之外的改进信号
  → 100 墙快速筛选，正向后才做 400 墙确认
  → 吞吐足够后，再做小步 on-policy RL 和冻结策略联赛
```

这里最重要的判断有五条：

1. **项目现在不是缺少动作标签数量，而是缺少能够超过 Teacher 的标签来源。** 约 90 万个 v4 决策大多来自同一确定性规则 Teacher。交叉熵的最优解是复现 Teacher，不会凭空创造 Teacher 不具备的战略。
2. **厦门麻将的数据问题首先是“独立性、覆盖和可辨识性”，其次才是规模。** 90 万决策来自约 2 万副自然物理墙；同墙内决策强相关，游金、金牌锁、吃碰杠等低频分支又不能用普通多数类准确率衡量。
3. **当前最大的技术瓶颈不是 Transformer 难训，而是目标不够好；同时本机也不具备继续投入大序列模型的资源条件。** 完整公开历史模型只能更稳定地模仿 Teacher；已经完成一次诊断后，Transformer 路线停止。
4. **从规则到纯神经网络之间确实需要中间态。** 抚州/南城公开技术报告中的生产路线本身也是“神经策略 + Teacher + lookahead + 公开风险/进展”的 hybrid，而不是一次跳到纯神经网络。它支持的是工程过渡思想，不是厦门规则或权重的直接迁移。
5. **近期主线应追求“可证伪的小幅真实改进”，而非理论上最完整的全局求解。** 全局 public-belief search、Deep CFR、ReBeL 式求解都值得长期研究，但当前的 belief proposal、Q 排序和模拟吞吐还没有满足其前置条件。

建议把未来投入分成四层：

- **P0：必须先做。** 规则差分测试、评测降方差、吞吐基线、Teacher 失误挖掘。
- **P1：最可能形成首个可部署神经过渡版本。** 结构化 residual、分阶段输出头、单辅助任务、DAgger、局部 oracle 标签。
- **P2：只有 compact hybrid 通过 400 墙后才重新评审。** 轻量 DAgger、固定少量参数的局部更新。
- **P3：只保留文献研究，不在当前机器实施。** PPO/VRPO、联赛、belief search、CFR、PTIE、PSRO/NFSP 和大序列模型。

纯 Teacher 模仿、随机初始化 PPO、单隐藏世界 Q、对已消耗 selection 反复调阈值、直接套用外部麻将权重，均应视为已知高失败率路线。

---

## 1. 证据边界：哪些是事实，哪些是推断

### 1.1 项目内可复核事实

- 网页默认和当前正式基线仍是 `HeuristicTeacherAgent`；没有神经 checkpoint 通过部署门槛。
- v4 语料包含 20,000 副自然 Teacher 物理墙、42,000 条总轨迹、900,589 个决策；890,589 个决策带完整公开历史，另有 10,000 个显式短窗口课程。
- 首轮 `candidate_mlp` 在独立测试上达到 96.95% Teacher 动作一致率，但 100 墙四座实战相对 Teacher 为 `+0.280 ± 0.847`，95% CI `[−1.380,+1.940]`，未通过。
- 已运行的短历史 Transformer、policy residual sequence、普通 PPO、对手池 PPO、fresh self-play PPO、Teacher-anchored residual PPO、单隐藏世界 Q、local-belief Q、logged-outcome selector、原始 DR、多个 SMC proposal、特权 critic、公开风险 Teacher、一步公开前瞻和精确一摸平分裁决，都没有通过预注册的实战或前置门槛。
- fresh PPO 相对 Teacher 灾难性退化；强 Teacher anchor 的 residual PPO 则保持严格零差异。这说明“放开锚点”和“锁死锚点”之间存在尚未解决的信用分配与探索窗口问题。
- 特权 critic 的同轨迹 A/B 中，当前配置没有降低残差方差，反而使残差均方恶化 33.0%；因此不能以“训练期有完整信息”自动推断它有助于 actor。
- direct-response Q 虽能降低某些离线误差，但没有稳定超过冻结 policy 的动作排序；local-SIR Q 即使离线排序门槛曾通过，实战仍为负。
- 常规筛选协议已改为 100 副独立物理墙、每墙四座轮换；只有稳定正向候选才扩大到 400 墙确认。

以上细节和数值以 [`experiment_log.md`](../../experiment_log.md) 为准。

### 1.2 外部一手资料事实

- [Suphx](https://arxiv.org/abs/2003.13590) 面向多人不完全信息麻将，使用 global reward prediction、oracle guiding 和 run-time policy adaptation；这说明强麻将系统通常是多阶段训练系统，而不是单一行为克隆网络。
- [Meowjong](https://arxiv.org/abs/2202.12847) 先监督预训练多个动作网络，再用 Monte-Carlo policy gradient 增强弃牌模型；它提供“按动作阶段拆分 + 监督启动 + 局部 RL”的案例。
- [Tjong](https://doi.org/10.1049/cit2.12298) 使用层次化的动作/牌决策和 Transformer，并先监督学习、再 PPO；论文报告约 50 万训练数据和多 GPU 训练。它证明 Transformer 可以是麻将系统的一部分，但不能证明“原始事件序列 Transformer”适合当前厦门标签和算力。
- [DouZero](https://arxiv.org/abs/2106.06135) 用动作编码、深度 Monte-Carlo 与并行 actors 处理斗地主的巨大变长动作集；其关键可迁移点是动作候选建模和吞吐，而不是斗地主奖励。
- [PerfectDou](https://arxiv.org/abs/2203.16406) 展示 perfect-training/imperfect-execution 的训练课程；其方法要求训练特权信息与部署 actor 严格隔离。
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html) 针对序列模仿中“模型动作改变未来状态分布”的问题，迭代收集候选访问状态并由专家再标注。
- [MahJax](https://arxiv.org/abs/2605.20577) 表明麻将 RL 的环境吞吐可以成为决定性基础设施；论文在 8 张 A100 上报告最高约 1–2M steps/s。这个数量级不能直接要求当前项目达到，但说明先优化环境再扩大 RL 是合理顺序。
- [VRPO](https://arxiv.org/abs/2605.19235) 指出不完全信息自博弈中 GAE 还受到未来随机动作采样方差影响，并以 centralized action-value critic 和 Expected SARSA trace 降方差；它给出后续 PPO 改造方向，但前提是本项目先有合格 Q critic。
- [Big 2 自博弈研究](https://arxiv.org/abs/2605.28863) 在其四人不完全信息环境中发现 PPO 优于几类 value 方法，并观察到适度熵与 current-policy self-play 的有限预算优势；这是相近工程信号，不是厦门麻将结论。
- [CQL](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html) 与 [IQL](https://arxiv.org/abs/2110.06169) 都处理离线 RL 的分布外动作问题；但若数据只覆盖确定性 Teacher，保守算法无法识别大量从未被可靠执行的替代动作。
- [ReBeL](https://arxiv.org/abs/2007.13544)、[Deep CFR](https://arxiv.org/abs/1811.00164) 和 [NFSP](https://arxiv.org/abs/1603.01121) 的主要理论语境是两人零和不完全信息博弈。四人厦门麻将是多人、随机、一般和环境，不能继承其均衡保证。
- [History Filtering](https://arxiv.org/abs/2311.14651) 说明从公开状态生成与历史相容的隐藏世界本身可能很难；这与本项目 proposal 的结构失败和 ESS 退化相符。
- [AIVAT](https://arxiv.org/abs/1612.06915) 展示在已知部分玩家策略时对不完全信息博弈评测降方差的方法；它是未来评测增强候选，不替代同墙四座实战。
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) 的强度来自人类数据、持续联赛和专门暴露主策略漏洞的 exploiters 等组合；这支持“策略种群是后期稳定器”，不支持在弱初始化上直接开大规模联赛。

### 1.3 本文的核心项目推断

以下不是论文已证明的普遍定理，而是基于上述证据对本项目的推断：

- 当前“数据太少”主要指**改进信号太少**，不是 JSONL 行数太少。
- 厦门麻将未能像抚州项目那样顺利过渡，不能归因于“厦门一定更复杂”；更直接的差别是当前 Teacher、数据构造、训练目标和搜索信号没有形成同一条可验证的提升链。
- 第一个超过 Teacher 的模型更可能是 hybrid/residual，而不是纯神经策略。
- 完整历史的价值只通过确定性的公开统计摘要进入模型，例如弃牌计数、最近 4–8 个动作、副露和巡次；不再训练序列网络。
- 若没有人类牌谱或更强 oracle，最有希望的新标签源是：局部精确求解、选择性多世界 rollout、候选—Teacher 分歧状态上的人工/规则复核，以及已知 propensity 的低幅随机干预。

---

## 2. 先定义“更强”，否则训练目标会漂移

### 2.1 主目标

正式目标是：在 `classic` 规则、相同独立物理墙、四座轮换、三名冻结基线/策略池对手下，最大化候选相对基线的每局净分，并报告按物理墙聚合的置信区间。

不能替代它的指标包括：

- Teacher 动作一致率；
- 训练或验证 loss；
- 单一 Q 的 MAE；
- 单个座位胜率；
- 训练自博弈回报；
- 某一个 seed 的单局观感；
- 对随机机器人或明显弱对手的胜率。

### 2.2 次级约束

任何强度候选还必须满足：

1. 合法动作率 100%，规则引擎拥有最终裁决权；
2. actor 不读取墙顺序、对手暗手、未来事件或训练随机种子；
3. 游金、金牌锁、吃碰杠优先级和结算没有回归；
4. 网页推理延迟可接受，慢速观察模式不是策略本身的计算预算；
5. checkpoint 的规则版本、特征契约、训练数据 manifest 和评测协议可追溯；
6. 任何 selection 上的调参不得再次读取同一 selection 并伪装成独立证据。

### 2.3 为什么 100 墙只能做筛选

100 墙能快速拒绝明显负向候选，但许多小幅提升的区间会跨零。项目已有 160 和 400 墙结果方向接近、仍不显著的案例。由此推导：

- 100 墙负向上界小于零：立即淘汰；
- 100 墙正向但区间跨零：只表示“值得研究”，不能上线；
- 100 墙下界大于零：允许进入 400 墙确认；
- 400 墙仍跨零：保持 Teacher，不可用点估计声称更强；
- 若方法只改变 1%–3% 决策，即使真实有效，也可能需要“分歧状态局部评测 + 全局墙评测”两级证据才能提高检验功效。

---

## 3. 厦门麻将训练的根本问题地图

### 3.1 规则复杂性不是唯一难点，规则稀有性更危险

游金、金牌锁、白板/金牌代理、吃碰后强制弃牌和不同阶段响应，会制造低频、非平滑的合法动作边界。普通监督数据会被常见摸打决策淹没；简单过采样又可能让模型把合成课程的分布误当成自然频率。

推导链：

```text
规则事件低频
  → 总体准确率对它们几乎不敏感
  → 网络可能在 97% 总准确率下仍错误处理关键规则
  → 若只加高 synthetic weight，常见策略又可能退化
  → 必须分阶段 head、分桶指标、自然/课程双报告和规则 fail-closed
```

### 3.2 900k 决策的有效样本量远小于 900k

同一物理墙内的多步决策共享初始牌墙、对手行为和终局回报。若按决策随机切分，训练和测试会共享强相关轨迹；当前项目已按物理墙切分，这是正确基础。但模型选择时仍应将“墙数”视为独立样本单位。

进一步推断：

- 对 Teacher 模仿，900k 行足以训练中小模型；
- 对终局 value，独立性更接近 20k 墙，而不是 900k 行；
- 对“某个替代动作是否更好”，有效样本只来自真正覆盖该动作的独立干预/多世界分支，可能只有几百或几千；
- 对游金等低频决策，有效自然样本还要更少。

### 3.3 确定性 Teacher 造成“上限”和“支持集”双重问题

设数据来自 Teacher `π_T`，监督目标为 `a_T=argmax π_T(a|I)`。交叉熵只要求：

```text
π_θ(a_T | I) → 1
```

它没有任何项告诉模型另一个动作 `a'` 在终局上更好。因此：

- 更多 Teacher 数据降低模仿误差；
- 网络的偶然偏离不是有根据的改进，而通常是泛化误差；
- CQL/IQL 能抑制数据外动作过估计，却不能凭空判断从未可靠执行的 `a'`；
- 要超过 Teacher，必须引入外部改进信号：更强规则/搜索、人类、随机干预、或足够稳定的 on-policy 回报。

### 3.4 信息集价值与“真实隐藏世界价值”不同

部署时 actor 只能看到信息集 `I`。正确目标是：

```text
Q^π(I,a) = E[R | I, a, π]
```

若 collector 在某个真实隐藏世界 `h` 上比较动作，得到的是 `Q(h,a)`。把它直接监督给只看 `I` 的模型，会把不可见的偶然牌墙结果当作动作规律。现有单世界 Q 和部分 local-belief 实验正好暴露了这个问题。

### 3.5 多人一般和让“自博弈变强”不再单调

四人麻将中，一个策略相对 Teacher 获利，可能是利用另外两名对手的行为，而不是全局更稳健。策略更新还会同时改变训练分布和对手分布。

后果：

- 不能只用“对最新版自己”训练；
- 不能把两人 payoff matrix 原样用于四人 PSRO；
- 评测需要固定三对手的阵容定义、座位轮换、平均和最差表现；
- 联赛应在有强初始化之后用于暴露漏洞，而不是从随机策略制造信号。

### 3.6 稀疏回报与高方差同时存在

一局结果受发牌、三家动作、响应优先级和特殊规则共同影响。一个早期弃牌对终局的边际贡献很小，普通 REINFORCE/PPO 的 advantage 很容易被整局噪声淹没。

奖励塑形也有风险：向听数、有效牌、危险度或“更快听牌”只是真实净分的代理。已有公开风险 Teacher 随权重增加持续恶化，说明代理权重错误会稳定地优化错误目标，而不是仅仅增加随机噪声。

### 3.7 历史有用，但不是所有事件都需要 Transformer

完整历史可能帮助判断：

- 对手公开弃牌节奏；
- 已见牌和副露结构；
- 当前局面阶段；
- 某玩家近期策略风格；
- 动作是如何到达当前信息集的。

但大量信息可以被无损或近似地聚合成计数、最近事件、轮次和公开威胁特征。若数据只教 Teacher 行动，Transformer 要同时学习规则、聚合器、顺序模式和 Teacher，优化难度高且收益上限不变。

full-history Transformer 消融现已完成并退休：测试 Teacher 一致率为 98.72%，但没有 Teacher 之外的改进目标。当前不再运行 MLP/GRU/Transformer 架构竞赛；历史只通过确定性固定长度摘要进入小模型。

### 3.8 belief 的主要难点是后验正确性，不是粒子数量

从公开历史推断隐藏手牌，必须同时满足牌数守恒、摸打结构、公开副露和对手策略似然。错误 proposal 即使生成很多粒子，也可能：

- 与后续事件结构不相容；
- 权重集中到极少粒子，ESS 崩溃；
- 用错误的独立 tile factor 忽略手牌组合依赖；
- 把 Teacher 的确定性动作转成近零似然，产生极端重要性权重。

现有多个 proposal 的失败说明：现在继续调粒子数、温度或 clip 很可能只是在延缓退化，而不是修正生成分布。

### 3.9 评测噪声会诱导“在 selection 上过拟合”

当每个候选只提高少量决策时，100 墙波动可能大于真实增益。若在同一 selection 上反复试权重、阈值、seed 和 checkpoint，就会选择到噪声最大者。

项目已经采用 selection/terminal 封存，这是应长期保持的研究资产。进一步需要：

- 对候选族预注册，而不是对单个 checkpoint 事后解释；
- 报告一整个族测试了多少组合；
- 使用按墙配对，而非把 4 个座位当 4 个完全独立样本；
- 研究 AIVAT/控制变量，但不替代全局实战。

### 3.10 模拟吞吐决定哪些算法“现实可行”

监督学习只需遍历已有数据；PPO、belief rollout、search、CFR 都会把需求放大为：

```text
墙数 × 四座 × 每局决策 × 合法动作 × 隐藏世界 × rollout 次数
```

若单步 Python 引擎和逐局网络推理吞吐不足，最复杂路线不是“训练更慢”，而是根本无法获得足以压低方差的数据。MahJax 的主要启示正是环境吞吐先于算法规模。

### 3.11 网页可部署性限制了搜索预算

网页 AI 的真实目标还包括可复现和响应延迟。一个离线可跑 30 秒的搜索 agent，不能直接替换网页默认策略。更现实的方式是：

- 离线用慢搜索产生标签；
- 在线由小网络蒸馏搜索偏好；
- 只有置信度低、动作少的局部决策才启用短搜索；
- 任何超时都 fail-safe 回退 Teacher。

---

## 4. 十二条关键演绎推理

### D1：高模仿率不代表实战更强

前提：标签由 Teacher 产生。
推理：模仿率衡量 `π_θ≈π_T`，实战目标衡量 `J(π_θ)>J(π_T)`。两者不是同一个命题。
项目证据：96.95% test accuracy 的 MLP 没有通过 100 墙。
结论：模仿率只用作“是否学会基本规则/行为”的准入指标。

### D2：继续把 Teacher 数据从 20k 墙扩大到 200k 墙，主要降低复制误差

如果规则 Teacher 固定、状态分布固定、目标仍是 CE，那么数据增大只会让网络更接近 Teacher。除非网络因为函数逼近产生有利归纳偏置，但这种收益不可依赖、也无法由标签证明。
结论：新的大规模采集必须增加**来源多样性**，不能只增加同一 Teacher 墙数。

### D3：要超过 Teacher，至少要有一个独立的优势判别器

候选改动作时必须回答：“为什么这个动作比 Teacher 动作好？”可接受答案只能来自：

- 精确局部算法；
- 多隐藏世界、同后续策略的低方差 rollout；
- 已知 propensity 的干预结果；
- 经授权的强人类选择/复核；
- on-policy 配对实战的稳定梯度信号。

“网络分数更高”“attention 看到了历史”“loss 更低”都不是优势判别器。

### D4：Teacher-anchored residual 是必要中间态，但锚点必须可穿透

锚点太弱：模型走入无覆盖状态并灾难退化。
锚点太强：residual 永远跨不过 Teacher margin，得到严格零差异。
因此正确问题不是“要不要 anchor”，而是：

```text
在哪些经独立证据支持的状态允许穿透？
穿透幅度多大？
失败时能否逐状态回退？
```

推荐用置信门控和分歧状态白名单，而不是全局固定 margin。

### D5：结构化小模型比序列模型更适合当前资源阶段

当前任务的 Teacher 模仿数据不小，但改进标签弱、规则特征强、算力有限。显式计数、最近 4–8 个事件的定长编码、Teacher 分数分量加 linear/GBDT/小 MLP，可以避免模型重新发现规则，并把单次训练限制在几十分钟。GRU 和 Transformer 都移出当前候选。

### D6：offline RL 在确定性 Teacher 数据上会面临不可解的支持缺口

即使 IQL 不显式查询数据外动作，它仍只能从数据中存在的动作结果学习“更好动作”。若某信息集只有 Teacher 动作被执行，替代动作没有 outcome，任何 Q 差主要来自函数外推。
结论：CQL/IQL 不是先采数据的替代品；它们应在随机干预/多策略数据形成后再进入。

### D7：单点干预比持续 epsilon 探索更适合建立局部因果标签

持续探索同时改变当前动作和后续策略，终局差不能归因于单个动作。单点干预后恢复冻结 policy，更接近 `Q^{π_T}(I,a)`。
但它仍需要足够 propensity、独立墙、cross-fitting 和低方差估计。现有 DR 尾部爆炸表明“设计上更正确”不等于有限样本可用。

### D8：多任务只有在标签与最终目标存在稳定关系时才有帮助

向听、有效牌、公开风险、胜率、游金可行性都可作辅助标签；但如果辅助 loss 主导共享 encoder，模型可能更擅长代理任务而实战更差。
结论：一次只加一个辅助头，固定 policy 容量，报告其 held-out 指标和实战变化；不做“多头全部堆上”的不可解释实验。

### D9：联赛解决“策略脆弱性”，不解决“初始策略不会打牌”

fresh PPO 已证明随机 actor 在当前预算下无法从稀疏回报启动。联赛把多个弱策略互相对打，不会自动生成强行为。
结论：先有 BC/hybrid 强初始化，再用 league 暴露针对性漏洞。

### D10：belief search 的准入条件是校准与吞吐，而非论文名气

若 posterior 不正确，search 会精确优化一个错误世界模型；若每节点样本太少，max 操作又会选择高噪声动作。
结论：先在 toy profile 与可枚举小状态验证 posterior，再在 classic 只做一层局部 search；否则停止。

### D11：规则 Teacher 的提升应从“错误案例”出发，而非全局权重网格

公开风险权重、一步前瞻、精确一摸裁决已经表明很多直觉规则覆盖率低或破坏原权衡。
更可靠流程是：收集 Teacher 与另一个可信 oracle/人类的分歧 → 分类失败类型 → 只改一个可解释局部 → 在分歧集和全局墙上双重验证。
结论：Teacher 优化进入“错误驱动开发”，停止盲扫全局 heuristic 权重。

### D12：第一版超过 Teacher 的系统很可能仍然保留 Teacher

若学习模块只在 2%–10% 高置信分歧状态介入，其余状态回退 Teacher，那么：

- 规则与常见行为下限得到保留；
- 改进信号集中，评测更容易解释；
- 失败可定位到 override 集；
- 后续可逐步扩大覆盖。

这与抚州公开 hybrid 路线一致，也与本项目现有反证最相容。

---

## 5. 候选路线总表（42 条）

状态说明：`主线` = 近期应投入；`条件` = 前置门槛满足后；`研究` = 长期；`停放` = 保留对照；`拒绝复跑` = 当前固定配置已有直接反证。

| ID | 路线 | 核心改进信号 | 近期状态 | 最大失败风险 |
| --- | --- | --- | --- | --- |
| F1 | 规则差分/性质测试 | 精确规则事实 | 主线 P0 | 测试覆盖不代表策略强度 |
| F2 | 批量/向量化模拟器 | 更多独立 on-policy 数据 | 主线 P0 | 快速实现与权威引擎语义漂移 |
| F3 | 分歧状态回放与可视化 | 定位 Teacher 真错误 | 主线 P0 | 只看精彩样本造成选择偏差 |
| F4 | 配对评测 + 控制变量/AIVAT-lite | 更低评测方差 | 条件 P0 | 错误概率模型引入偏差 |
| T1 | 错误驱动的局部规则 Teacher | 可解释局部 oracle | 主线 P1 | 代理指标损害全局净分 |
| T2 | 选择性一层公开 rollout Teacher | 多世界局部回报 | 条件 P1 | posterior 错误、max 偏差、延迟高 |
| T3 | Teacher ensemble/策略风格池 | 多种规则先验 | 主线 P1 | 弱规则的多数投票仍然弱 |
| T4 | 人类专家分歧复核 | Teacher 外部标签 | 条件 P1 | 授权、质量、规则一致性不足 |
| B1 | 结构化 candidate MLP residual | 小幅可解释偏移 | 主线 P1 | 再次只学会复制 Teacher |
| B2 | 公开历史固定摘要 + 小模型 | 顺序公开行为的确定性压缩 | 主线 P1 | 摘要遗漏真正有用的顺序信号 |
| B3 | 分阶段/层次化动作 heads | 降低混合动作冲突 | 主线 P1 | head 间概率不可比 |
| B4 | 单辅助头多任务学习 | 牌效/规则/风险表征 | 主线 P1 | 优化代理而非净分 |
| B5 | DAgger 状态聚合 | 修复候选分布偏移 | 主线 P1 | Teacher 标签仍有上限 |
| B6 | disagreement-only distillation | 聚焦有意义 override | 主线 P1 | 分歧集带选择偏差 |
| B7 | 置信门控 hybrid | Teacher 安全回退 | 主线 P1 | 门控过严零改动、过松退化 |
| B8 | 搜索策略蒸馏 | 慢 oracle → 快 actor | 条件 P1 | student 复制 search 噪声 |
| D1 | 已知 propensity 单点干预 | `Q^Teacher(I,a)` 局部因果信号 | 条件 P1 | 低 propensity 与终局高方差 |
| D2 | 分阶段定向干预 | 提高吃碰过等支持度 | 条件 P1 | 自然分布覆盖仍有限 |
| D3 | CQL 离线 RL | 保守抑制 OOD Q | 条件 P2 | Teacher 数据支持不足 |
| D4 | IQL/AWR | 保守 advantage BC | 条件 P2 | value 排序不合格 |
| D5 | 分布 Q / quantile value | 风险敏感决策 | 条件 P2 | 分位数标签样本远不足 |
| D6 | OPE 只作筛选器 | 低成本拒绝候选 | 条件 P1 | 将 OPE 误当上线证据 |
| R1 | BC warm-start PPO | 真实 on-policy 净分 | 条件 P2 | 稀疏回报、灾难遗忘 |
| R2 | KL/Teacher 自适应锚定 PPO | 安全探索 | 条件 P2 | 锚点两端：零改动或崩溃 |
| R3 | 当前快照 + 冻结策略混合 | 有限预算课程 | 条件 P2 | 非平稳与策略坍缩 |
| R4 | VRPO/Q-boosting | 降低 stochastic future variance | 研究 P2 | 当前 Q critic 未达门槛 |
| R5 | 小型 league / exploiters | 暴露策略漏洞 | 条件 P2 | 弱初始化、四人阵容爆炸 |
| R6 | NFSP-lite 平均策略 | 稳定自博弈平均行为 | 研究 P3 | 两人零和假设不成立、启动弱 |
| R7 | PSRO-lite 策略种群 | 鲁棒对手分布 | 研究 P3 | 四人元博弈定义不唯一 |
| R8 | population distillation | 压缩策略混合 | 研究 P3 | 蒸馏抹平少数关键策略 |
| S1 | public belief + 一层 search | 信息集局部规划 | 研究 P3 | 历史过滤难、ESS 退化 |
| S2 | 对手公开行为模型 | 更好的 belief/风险 | 条件 P2 | 单 Teacher 过拟合、身份泄漏 |
| S3 | response 局部 CFR/MCCFR | 吃碰过局部遗憾 | 研究 P3 | 局部收益定义与全局错位 |
| S4 | ReBeL/Student-of-Games 式局部求解 | belief + value + search | 研究 P3 | 工程量和理论假设鸿沟大 |
| A1 | raw full-history Transformer | 自动学习长序列模式 | 已完成诊断并退休 | 目标弱、计算贵、只有 Teacher 模仿增益 |
| A2 | structured Transformer hybrid | 显式聚合 + 少量 attention | 当前资源下停止 | 投入产出比低，不再实施 |
| A3 | PTIE/progressive hiding | 特权课程帮助 visible actor | 研究 P3 | 泄露、critic 无增益、蒸馏失败 |
| X1 | fresh tabula-rasa PPO | 纯自博弈回报 | 拒绝复跑 | 已灾难性弱于 Teacher |
| X2 | 单真实世界 Q argmax | 反事实终局回报 | 拒绝复跑 | 目标不是信息集价值 |
| X3 | 未缩减 DR advantage | 因果/OPE 伪标签 | 拒绝复跑 | 低 propensity 尾部爆炸 |
| X4 | 全局 heuristic 权重网格 | 手工代理 | 拒绝同族盲扫 | 多重比较、代理错位 |
| X5 | 直接复制外部麻将模型 | 迁移学习 | 拒绝 | 规则/动作/许可/信息边界不兼容 |

---

## 6. 近期主线路线卡

### F1：规则差分测试与状态机性质测试

**假设。** 训练前先消灭规则和回放的不一致，可避免网络学习不存在的策略差异。

**实施。** 对每种动作阶段生成小规模状态，比较：权威 Python 引擎、网页调用、数据 collector 重放、未来批量/Rust/JAX adapter 的合法动作和结算；增加牌数守恒、响应优先级、同墙同 seed 确定性、座位旋转等性质测试。

**可能成功的原因。** 麻将学习系统的“标签”最终来自规则引擎；一个隐藏的规则分叉错误会污染所有后续训练。

**可能失败的方式。** 测试全绿但策略仍弱；这是基础设施路线，不产生强度信号。

**最小证伪实验。** 同一批 1,000 个回放状态在所有执行路径上动作 mask、状态 hash、终局分完全一致，否则禁止扩规模。

### F2：高吞吐、可验证的批量模拟层

**假设。** 当前反事实、RL 和 search 的样本方差主要需要更多独立环境样本解决，吞吐提升的边际价值高于增加模型宽度。

**实施。** 先不重写权威规则；建立批量 state clone、批量 agent inference、并行墙调度与性能剖析。第二步才评估 Rust/JAX 快路径，并以 F1 差分测试锁定语义。

**成功原因。** 动作 Q、PPO、belief search 的所有可信度都依赖独立 rollout 数；批量推理还可降低 GPU 空转。

**失败方式。** 并行环境共享可变状态、seed 冲突、动作响应竞态；或者吞吐提升后发现真正瓶颈是 Python 规则而非网络。

**最小门槛。** 相同 128 墙串行/批量逐事件相同；零非法动作；报告 env steps/s、games/s、GPU 利用率和内存；至少得到可重复的 3× 吞吐提升后再扩大 RL。

### F3：Teacher 分歧/失误数据集与回放工具

**假设。** Teacher 不是处处都错，首个提升应集中在可解释的少数错误类别。

**数据来源。** Teacher 与以下对象的分歧：局部精确算法、多世界 rollout、经授权人类选择、另一个独立规则 agent、历史候选。每个分歧只保存 actor 可见状态、合法动作、各方案理由和最终审计状态。

**成功原因。** 将“全局超过 Teacher”的稀疏问题变为“某类状态谁更好”的可诊断问题。

**失败方式。** 只挑看起来精彩的案例，形成严重选择偏差；或者 oracle 本身使用隐藏信息。

**防护。** 分歧进入数据集前固定采样规则；保留未筛选分歧总数；任何 oracle 标记其可见字段；最终仍做全局同墙评测。

### T1：错误驱动的局部规则 Teacher 2.0

**假设。** 规则 Teacher 仍有可被精确局部逻辑修复的错误，但不应再用全局代理权重扫描。

**实施步骤。**

1. 从 F3 按失败类型聚类，例如“吃后牌效”“游金保存”“金牌锁响应”“听牌时防守”“残局有效牌”。
2. 每次只为一个类型写纯函数 oracle 和单测。
3. 只有在分歧集上显示正确方向，才组成 `Teacher + local override`。
4. 在 100 墙全局筛选；若 override 率过低，额外报告每次 override 的配对局部结果，但不以局部结果替代全局上线门槛。

**为什么可能成功。** Teacher 有强规则下限，局部修复避免重学整个策略；抚州 hybrid 也保留规则与 lookahead。

**为什么可能失败。** 麻将动作的长期价值高度耦合，局部“更快听牌”可能损害游金、得分上限或防守；已有风险和一步前瞻失败就是反例。

**停止条件。** 同一局部规则两组独立 100 墙均没有正向信号，或全局上界小于零；停止调该规则权重。

### B1+B7：结构化 Teacher-anchored residual + 置信门控 hybrid

**核心形式。**

```text
score(a|I) = teacher_score(a|I)
           + gate(I) * residual_theta(structured_visible_features, a)
```

其中规则引擎先给合法集合；`gate=0` 时严格等于 Teacher。`gate` 不直接由未校准网络自由决定，而由以下白名单组合：

- 训练数据覆盖充分；
- ensemble residual 方差低；
- Teacher 与可信 oracle 有分歧；
- residual margin 超过预注册阈值；
- 当前状态不属于规则高风险/低样本桶。

**与失败 v1-b 的区别。** v1-b 使用全局固定 Teacher margin，训练后没有动作跨过边界。本路线将穿透权限限定在有独立证据的状态，而不是期待稀疏 PPO 自己克服固定 margin。

**可能成功。** 保留 Teacher 常见动作强度，只让网络学习少量 residual；可逐决策解释和回退。

**可能失败。** gate 学成“永不介入”或“在错误的高置信状态介入”；ensemble 一致也可能是共同偏差。

**最小实验。** 先只在一种动作阶段、一个分歧类别上训练；要求 override 率在预注册区间（例如 1%–10%），分歧 held-out 排序优于 Teacher，且 100 墙不为负；否则不扩覆盖。

### B2：结构化聚合 + 定长最近事件的小模型

**固定输入。** 手牌/公开牌计数、副露、金牌、轮次、合法动作、规则阶段、Teacher 分数分量，以及四家最近 4–8 个动作的定长编码。所有统计由规则代码确定性生成，训练和网页共用实现。

模型只比较 linear、GBDT 和 hidden 64/128 的小 MLP，输出 residual 或分阶段动作头。

**可能失败。** 摘要可能遗漏长程顺序关系；但在当前资源下，正确处理是记录这个限制，而不是切换到 GRU/Transformer。

**最小证伪。** 每组新摘要单独加入，单次训练不超过 30 分钟；只有分歧 held-out 提升才保留。总体 Teacher accuracy 的小幅提升不构成采用理由。

### B3：层次化/分阶段动作模型

**建议层次。**

```text
阶段 1：normal discard / response / special claim
阶段 2：pass / hu / chi / pong / kan / tour 等动作类型
阶段 3：具体牌或具体 chi 组合
```

规则引擎继续提供完整合法动作，层次化网络只分解概率计算。

**可能成功。** response 与 discard 的标签、回报、动作数量和历史依赖明显不同；Tjong 与 Meowjong 都采用了动作分解思想。

**失败方式。** 阶段概率相乘后，不同路径分数不可校准；某个 head 样本极少；规则引擎动作与层次编码不一一对应。

**门槛。** 每个合法 flat action 可逆映射到唯一层次路径；概率归一化单测；分阶段准确率、校准和自然频率分别报告。

### B4：一次只加一个辅助头

**候选标签顺序。**

1. 规则事件可行性/动作阶段；
2. 弃牌后向听/有效牌桶；
3. 公开可见牌和剩余张数；
4. Teacher 各评分分量的蒸馏；
5. 终局净分/胜率低权重预测；
6. 风险/游金进度，只有在标签校准后。

**可能成功。** 辅助任务提供密集监督，让 encoder 学到牌形与规则结构。

**失败方式。** 代理标签主导共享表示；低频合成课程使模型误估自然概率；终局 outcome 头只能拟合 executed action，不能授权反事实 argmax。

**实验纪律。** 每次只加一个 head；固定主干参数量；loss 权重只在 validation 调一次；若辅助指标升而 100 墙或分歧集降，删除该头。

### B5：DAgger，而不是无止境 Teacher-only BC

**流程。**

1. 从当前 BC/hybrid 候选出发，在冻结 Teacher/策略池中运行；
2. 记录候选实际访问且与 Teacher 分歧的 actor-visible 状态；
3. Teacher 或更强局部 oracle 给标签；
4. 按物理墙聚合回原数据；
5. 逐轮降低 Teacher 控制比例，但保持安全回退。

**可能成功。** 修复“训练看 Teacher 状态，部署看自己造成的状态”的分布偏移，这是 DAgger 的直接问题设定。

**根本上限。** 若所有再标签仍来自同一个 Teacher，DAgger 主要提高复制稳定性，不能保证超过 Teacher。它必须和 T1/T2/T4 中至少一个更强标签源组合，才能成为“提升”路线。

**停止条件。** 两轮 DAgger 后候选—Teacher 分歧下降但实战仍为零，说明只是在更好地复制 Teacher；停止单独扩大。

### T2+B8：选择性局部 search Teacher 与蒸馏

**目标不是全局 MCTS。** 只在合法动作少、Teacher 分差小、后续 1–2 层可控的状态运行多世界 rollout。搜索输出完整动作分布、均值、置信区间和有效样本数；网络只学习高置信分歧。

**可能成功。** 慢搜索可提供 Teacher 之外的标签，在线 student 保持低延迟。

**失败方式。**

- belief world 不是历史后验；
- 对每个动作取 max 产生 winner's curse；
- 冻结后续 Teacher 使搜索只优化 `Q^Teacher`，可能不适合 student 后续策略；
- search 覆盖过低，无法影响全局；
- search 延迟/吞吐无法扩大。

**最小证伪。** 在一个可枚举 toy profile 上，近似搜索动作排序与精确期望一致；classic 上 search 重复两次的 top action 稳定；高置信分歧对 100 墙有正向信号。任一失败则保持离线诊断。

### T4：经授权的人类专家分歧复核

**假设。** 当前最缺的是 Teacher 外部信号，少量高质量“何切/是否响应”标签可能比更多 Teacher 墙更有价值。

**建议采集。** 不需要一开始收集百万牌谱。先给熟悉厦门规则的人展示 F3 随机抽样分歧状态，要求选择动作、信心和简短理由；隐藏所有墙牌和结果，防止 hindsight label。

**可能成功。** 人类可以指出 Teacher 的系统性盲点，形成局部规则或 residual 数据。

**失败方式。** 专家水平未知、规则口径不同、看到结果后修正选择、多个专家一致性低。

**门槛。** opt-in 授权；规则问卷；盲态标签；至少双人重叠子集；报告一致率；人类训练集与最终真人评测严格隔离。

---

## 7. 条件路线：数据和基础设施通过后再做

### D1+D2：已知 propensity 的单点、分阶段干预

**正确用法。** 每局最多一个动作随机化，随后恢复冻结策略；弃牌、response、特殊动作分别设计行为概率，确保替代动作有支持度。

**为什么仍值得保留。** 现有实验已验证数据契约和 propensity 可记录；失败主要来自有限样本方差与选择识别，而不是概念完全错误。

**为什么不能立即重跑。** 已有原始 DR 的非 Teacher 伪优势标准差达 162–207 分量级，低 propensity 放大尾部；多个 shrinkage 策略在独立 selection 没有正下界。继续换 clipping 数字会在相同问题上过拟合。

**重新准入条件。**

- F2 吞吐允许至少提升一个数量级的独立干预墙；
- 行为设计让关键桶的最小 propensity 和 ESS 达到预注册值；
- reward 改为更低方差、可校准的局部中间目标，或使用更强 control variate；
- 使用全新墙组、cross-fitting 和封存 terminal。

### D3+D4：CQL / IQL / AWR

**适合的数据。** 多个行为策略、已知动作 mask、足够替代动作覆盖、严格按墙切分、真实 executed action 与 outcome 对齐。

**可能成功。** 在已有多策略/干预数据上做保守策略改进，限制 neural policy 远离可支持区域；IQL 的 advantage-weighted BC 与 hybrid 过渡方向相容。

**失败方式。**

- 仍然只有 Teacher 动作，Q 无法辨识；
- 多人非平稳环境让固定 MDP 数据假设偏离；
- 终局 reward 方差使 expectile/advantage 不稳定；
- 保守度过强变回 BC，过弱则 OOD 过估计。

**准入门槛。** 在按墙留出的已覆盖动作上，Q/advantage 排序相对 policy 有正下界；策略 override 率受控；OPE 只作筛选，最终仍过 100/400 墙。

### D5：分布价值与风险敏感选择

**动机。** 相同均值的两个动作可能有不同放铳尾部和高分上限；用 quantile/CDF 比单一均值更接近麻将风险决策。

**可能成功。** 给 hybrid gate 提供“均值改进但尾部不可接受”的过滤器。

**失败方式。** 每个信息集—动作的独立结果太少，分位数比均值更难估；人为选择 CVaR 水平可能再次变成全局风险权重失败。

**最小实验。** 先只预测 executed action 的 return distribution，做 calibration/reliability；没有校准前禁止用于 argmax。

### R1+R2：BC warm-start、KL/Teacher 自适应锚定 PPO

**和已有失败的区别。** 不从随机 actor 开始，也不使用不可穿透固定 margin。actor 从通过模仿/规则分桶门槛的 hybrid 开始；KL 系数根据策略偏移和实战分歧率动态调整；高风险状态仍硬回退 Teacher。

**奖励。** 主奖励仍为规则结算净分；辅助 shaping 只作 critic/task loss，不直接无限累加到真实 reward。按座位和墙标准化 advantage，避免某些结算尺度主导。

**可能成功。** 保留可玩初始化，同时让 on-policy 回报提供 Teacher 之外的信号。

**失败方式。** critic 信号弱；KL 过强无变化、过弱灾难遗忘；对手分布变化导致策略循环；PPO 训练回报与冻结 Teacher 实战脱钩。

**最小实验。** 先用 3 个固定 KL 档、相同 rollout，检查动作 KL、override 率、advantage SNR 和 100 墙；不是先扩到几十个超参。

### R3：current snapshot + frozen pool curriculum

**设计。** 每轮更新前冻结 current policy；对手按固定概率从 Teacher、上轮快照、历史强 checkpoint 和风格 Teacher 中抽取。训练概率写入报告。

**理论/外部启发。** Big 2 近期研究在其有限预算中观察到 current-policy self-play 优势；AlphaStar 则说明长期鲁棒性需要历史和 exploiter 多样性。

**项目风险。** 两个结论不冲突：早期 current snapshot 提供匹配难度，后期历史池防遗忘。但四人阵容组合会快速增长。

**门槛。** 每个候选不仅对三 Teacher，还要对至少三种冻结阵容报告平均/最差表现；不以训练 pool 内胜率晋升。

### R4：VRPO / Q-boosting

**为什么有吸引力。** 它直接针对 stochastic policy 在不完全信息自博弈中的 advantage 方差，理论问题与麻将相符。

**为什么现在不能上。** 当前公开 Q 的动作排序和特权 critic 都没有达到项目门槛；用错误 centralized Q 做 expected backup 会稳定引入偏差。

**准入条件。**

1. 固定 actor 的 Q critic 在未见墙上对已覆盖动作排序显著优于零/公开 value；
2. 同轨迹比较中，Q-boosted advantage 方差下降且均值一致；
3. centralized critic 的完整信息不进入 actor/checkpoint；
4. 普通 PPO 和 VRPO 使用相同 rollout 预算做 A/B。

### R5：小型 league 与 exploiters

**目的。** 不是求 Nash，而是发现主策略对特定风格的脆弱点。

**角色。**

- main：当前 hybrid/BC-PPO 主策略；
- historical：冻结旧版本；
- exploiters：只优化击败 main 的局部策略，但部署不可用；
- rules：保守、激进、牌效优先等可解释 Teacher 变体。

**失败方式。** exploiters 利用模拟器漏洞；main 只对池内过拟合；四人组合成本高；平均提升掩盖某类阵容崩溃。

**防护。** F1 规则测试；固定阵容矩阵；报告 worst-case；任何 exploiter 发现的漏洞先转成回归测试。

### A2：结构化 Transformer hybrid（当前资源下停止）

该路线只保留为文献记录。已完成 full-history Transformer 诊断后，本机不再训练 raw 或 structured attention。即使未来固定摘要出现长历史缺口，也先改进规则摘要和分歧标签，而不是增加序列容量。

---

## 8. 长期研究路线

### S1：public belief + 有限深度 search

**完整目标。** 根据本家私有观察和公开历史维护 `p(hidden histories | information set, opponent policies)`，在该分布上估计动作价值。

**路线分解。**

1. toy profile 全枚举后验；
2. 精确牌数约束 proposal；
3. 逐事件 likelihood / MCMC history filtering；
4. posterior calibration 与 ESS；
5. 一层 response/search；
6. 慢 search 蒸馏；
7. 最后才考虑更深 search。

**可能成功。** 这是理论上最接近正确信息集规划的路线，能生成 Teacher 外部标签。

**可能失败。** 历史生成难；Teacher likelihood 过于尖锐；对手模型错误；多人一般和没有安全 resolving 保证；计算乘法爆炸。

**当前结论。** 研究保留，但现有独立 tile factor 和线性行为 energy proposal 已失败，禁止在 classic 直接扩粒子重跑。

### S2：公开行为对手模型

**目标。** 从弃牌、副露、响应速度/选择（若规则数据中有）预测对手风格或下一步公开动作概率，供 belief 或 risk head 使用。

**可能成功。** 历史的真正增益很可能来自对手行为，而不是牌形本身。

**失败方式。** 训练对手全是同一 Teacher，模型只学座位/阶段；对真人或不同策略失效；使用 persistent identity 造成隐私或泄漏。

**准入。** 先建立至少四种冻结行为策略；按策略版本和墙隔离；跨策略测试；只用公开历史，不用身份 ID。

### S3：response 局部 CFR/MCCFR

**目标。** 只在“吃/碰/杠/过”的短子博弈上研究遗憾式学习，而非求解整个麻将。

**可能成功。** 动作集合小、决策边界清晰，可把 information-set 表示与 regret 目标做成受控实验。

**失败方式。** response 后仍有长局，截断 value 决定结果；四人一般和没有原始 CFR 的全局收敛保证；子博弈边界选择可能改变最优动作。

**最小实验。** 在缩短牌墙/toy profile 可精确枚举子博弈上，tabular CFR 与解析/枚举结果一致；然后才替换 value 为网络。

### S4：ReBeL / Student of Games 式局部求解

**必要组件。** public state、belief、counterfactual value、局部 search、self-play、评测与高吞吐环境。当前项目只有其中部分接口，value/belief 尚未校准。

**可能成功。** 长期上它能统一 search 与学习，避免单世界 determinization。

**失败方式。** 工程量远超当前数据和算力；两人零和理论不能外推；任何一个组件误差都会被 search 放大。

**决策。** 作为 6–12 个月研究方向，不作为下一个 checkpoint 的实现任务。

### R6/R7：NFSP-lite / PSRO-lite

**可迁移部分。** best-response 与 average policy 分离、reservoir、冻结策略池、元分布采样。

**不可迁移部分。** 两人零和均衡/可利用性结论；简单二人 payoff matrix。

**重新准入。** 必须先有：一个达到 Teacher 水平的 neural/hybrid 初始化；稳定的多策略吞吐；四人阵容定义；final average policy 评测协议。

### A3：PTIE / progressive hiding

**目标。** 训练期 oracle 用完整状态学到更低方差结构，再逐步遮蔽为 visible student。

**可能成功。** 特权信息可以帮助学习规则机制和 outcome 表征，Suphx/PerfectDou 提供相近案例。

**失败方式。** 直接 oracle 动作无法由 visible student 复现；蒸馏目标仍条件于真实隐藏世界；泄漏进入 checkpoint；已有 privileged critic 没有降方差，progressive hiding smoke 也未通过。

**正确重新试法。** 只在 toy profile 做字段级遮蔽 curriculum，比较最终 visible student 校准；不允许 full-state actor 进入 classic 或网页。

---

## 9. 高风险和可能失败的路线：为什么失败

### X1：继续纯 Teacher 模仿直到“自然超过 Teacher”

**失败原因。** 目标函数没有 Teacher 之外的偏好信息。网络偏离 Teacher 的部分主要是误差，不是优势。
**保留价值。** 作为初始化、压缩和规则行为基线。
**错误做法。** 用 99% accuracy 或更低 loss 宣称更强。

### X2：从随机网络直接 PPO/self-play

**项目反证。** 固定 v1-a 配置相对 Teacher 约 `−21.75` 分/局，显著灾难性退化。
**失败原因。** 稀疏终局回报、长 horizon、四人非平稳和巨大有效状态空间共同导致启动信号太弱。
**不得做。** 在同一算法上只增加 iteration 或换 learning rate，除非先改变启动信号或奖励识别问题。

### X3：固定强 margin 的 residual PPO

**项目反证。** v1-b 最终与 Teacher 严格零差异。
**失败原因。** PPO 梯度不足以跨过 deterministic Teacher 排序边界；训练看似稳定，实际策略没有变化。
**替代。** 有证据状态上的自适应 gate，而不是全局 margin。

### X4：单真实隐藏世界 counterfactual Q

**项目反证。** 离线反事实 loss 可显著改善，第二独立墙实战仍显著/方向性变差。
**失败原因。** `Q(h,a)` 被当成 `Q(I,a)`；model 只看 `I` 却被迫拟合不可见偶然性。
**不得做。** 通过重复同一 hidden world rollout 伪造独立信息集样本。

### X5：把 executed-action outcome head 用于所有动作 argmax

**项目反证。** logged-action outcome 校准改善，但 response top-2 selector 在 200 墙显著为负。
**失败原因。** 预测执行动作结果不等于识别未执行动作的相对因果效果；函数外推与选择偏差被 argmax 放大。

### X6：未缩减 DR / 低 propensity OPE 标签

**项目反证。** 非 Teacher DR 伪优势标准差比 direct gap 大一个数量级以上，出现数千分极端值；shrinkage 候选仍未通过独立 selection。
**失败原因。** 小 propensity 的逆权重将终局噪声放大；有限样本下无偏不等于低误差。
**不得做。** 在已消耗 selection 上继续调 clipping/threshold。

### X7：raw full-history Transformer 自动解决一切

**最终状态。** 短事件 Transformer 的测试准确率和实战不理想；full-history Transformer 已完成 8 epochs，在 84,538 个测试决策上达到 98.72% Teacher 一致率，但没有 Teacher 之外的改进目标，按资源政策退休且不做实战。
**可能失败原因。**

- 数据目标仍是 Teacher；
- 事件序列中规则结构强，模型需浪费容量重新发现；
- 长尾课程与自然轨迹分布不同；
- 计算预算不足，训练只跑少量 epoch；
- 模型改善 common actions，总体 accuracy 上升但关键分歧不变。

**当前决策。** 不再投入 MLP/GRU/Transformer 架构竞赛。资源转向 SlowExpertTeacher 分歧标签和 linear/GBDT/小 MLP residual；总体强度仍由 100/400 墙判断。

### X8：全局 MCTS / determinization

**失败原因。** 对一个或少数确定隐藏世界做搜索会产生 strategy fusion：同一个可见信息集在不同 hidden world 选择不同动作，但部署 actor 无法知道它处于哪个 world；多人对手策略与 chance 分支又使树巨大。
**可保留形式。** 多 belief world 的一层局部 rollout，输出置信区间并蒸馏；不是全局搜索。

### X9：直接套 Deep CFR / ReBeL / DeepNash

**失败原因。** 理论假设、收益结构、参与人数和游戏树规模不匹配；工程前置组件缺失。
**可保留形式。** 在 toy 或 response 子博弈验证信息集表示和局部 regret。

### X10：全局奖励塑形堆叠

**失败原因。** 向听、危险度、听口、游金进度并非真实净分的势函数；权重改变最优策略。已有风险 Teacher 随权重增加恶化。
**可保留形式。** 辅助预测头或 critic 输入，不直接替代终局 reward；一次只加一个并消融。

### X11：PBT/大规模超参搜索救活不正确目标

**失败原因。** PBT 会更快地找到 selection 噪声和代理 reward 漏洞；当数据契约/Q 目标不正确时，超参优化不能修复估计对象。
**准入。** 只有固定小网格已显示稳定正向、独立 terminal 足够、吞吐成为瓶颈后。

### X12：直接迁移抚州/日麻权重或牌谱

**失败原因。** 规则、计分、牌数、特殊动作、合法性和信息编码不一致；许可证与数据授权也可能不允许。
**可迁移。** 架构模式、训练阶梯、测试思想；若做表征预训练，只能用明确许可数据并完全重建厦门 action head，且必须证明没有负迁移。

---

## 10. 推荐的组合路线，而不是单算法押注

### 组合 A：最现实的首个可部署提升候选

```text
F1 规则/回放一致性
  + F3 Teacher 分歧数据集
  + T1 一个错误驱动的局部 oracle
  + B1 结构化 residual
  + B3 分阶段 head
  + B7 高置信门控回退 Teacher
  + 100/400 墙评测
```

**成功画像。** 只 override 1%–8% 决策，其余完全等于 Teacher；override 集中在一个已验证类别；网页仍可解释和回退。
**失败画像。** override 太少导致无法测量，或局部正确但全局负向。
**优先级。** 最高。

### 组合 B：从稳定模仿过渡到低成本公开历史摘要

```text
v4 数据契约的平衡子样本
  + structured aggregates
  + 最近 4–8 个公开动作定长编码
  + one auxiliary head
  + DAgger candidate states
  + Teacher residual gate
```

**成功画像。** 在相同 Teacher 模仿率下，稀有动作桶、分歧集和跨策略对手泛化提升；随后由局部 oracle 提供真正超越信号。
**失败画像。** 只提高 common action accuracy，实战等同 Teacher。
**优先级。** 高。

### 组合 C：有吞吐后的安全 on-policy 强化学习

```text
通过行为/分歧门槛的 hybrid
  + F2 批量环境
  + adaptive KL Teacher anchor
  + current snapshot / frozen pool
  + distribution/value critic calibration
  + PPO → 条件满足后 VRPO A/B
```

**成功画像。** override 逐步扩大，KL 可控，对多个冻结阵容都有提升。
**失败画像。** 训练回报升但 Teacher 实战不升；KL 两端失控。
**优先级。** 中，必须等组合 A/B 和吞吐。

### 组合 D：高成本、理论更完整的 belief-search 路线

```text
toy 精确 posterior
  + history filtering/MCMC
  + opponent model
  + calibrated public value
  + one-layer belief search
  + search distillation
  + local resolving research
```

**成功画像。** search 在可枚举 toy 上正确，在 classic 重复采样稳定，蒸馏后带来全局增益。
**失败画像。** proposal/ESS/吞吐任一失败，search 只放大偏差。
**优先级。** 长期。

### 组合 E：高质量人类数据路线

```text
opt-in 厦门牌谱/分歧标签
  + rule-version parser
  + blind held-out human test
  + BC / DAgger
  + hybrid residual
  + on-policy league
```

**成功画像。** 终于获得 Teacher 外部且符合本地规则的强行为先验。
**失败画像。** 数据量小、玩家水平混杂、规则版本不一致或真人训练/评测泄漏。
**优先级。** 如果能获得可靠授权，价值很高。

---

## 11. 建议的实际推进阶段与停止门槛

### 阶段 0：冻结当前证据（已完成）

- 当前 Teacher 仍是网页默认。
- full-history Transformer 已完成，test Teacher 一致率 98.72%；状态为资源策略退休，不做 100/400 墙，不接入网页。
- 所有已拒绝 selection 墙保持消耗状态，不重调。

**产物。** 本文、实验注册模板、当前模型/数据 manifest。

### 阶段 1：建立 Teacher 错误地图（1–2 个迭代周期）

- 从现有 20k 墙和可观察牌桌生成均匀抽样的 Teacher 决策回放；
- 记录 action margin、动作阶段、规则标签、是否与候选/局部 oracle 分歧；
- 人工试玩只作规则/行为观察，不直接当强度证明；
- 选择一个覆盖率足够的失败类别。

**停止门槛。** 找不到可重复、可盲态判定的 Teacher 错误类别，则暂停规则 Teacher 优化，转向外部人类数据或吞吐。

### 阶段 2：训练最小 residual hybrid

- linear、GBDT 或 hidden 64/128 的 candidate MLP；
- 一个动作阶段、一个辅助头、一个 override 类别；
- Teacher gate fail-safe；
- DAgger 一轮。

**离线门槛。** 合法率 100%；规则桶无回归；分歧 held-out 排序相对 Teacher 有正向下界；override 率落在预注册区间。
**实战门槛。** 100 墙下界 > 0 才进入 400 墙；否则归档而不调同一 selection。

### 阶段 3：模拟器吞吐和多策略数据

- 批量环境与 GPU batch inference；
- 固定 Teacher、风格 Teacher、历史 hybrid 的策略池；
- 记录实际 behavior probability 和 executed action；
- 单点干预只在新的覆盖协议下重新准入。

**门槛。** 串/并行完全一致；吞吐有可复现数量级提升；按 action bucket 的 ESS/覆盖通过。

### 阶段 4：安全 PPO

- 从通过阶段 2 的 hybrid 开始；
- 小 KL 网格、相同 rollout；
- current snapshot + frozen pool；
- 逐 iteration 固定评测，不挑中间 checkpoint；
- Q critic 未校准前不用 VRPO。

**停止门槛。** 两个独立迭代族均未在 100 墙给出正向信号，或出现 catastrophic forgetting；停止扩大 PPO，回到奖励/数据识别。

### 阶段 5：belief/search 研究

- 只从 toy posterior 开始；
- 先证明历史过滤正确，再谈深度；
- search 先只做 response 或一摸一打；
- 所有 full-state 数据留在 collector 内存。

**停止门槛。** toy 后验误差、classic ESS、重复搜索 top-action 稳定性或吞吐任一未达标，不进入策略训练。

---

## 12. 下一轮最小可执行实验包（资源约束版）

如果只允许启动一个新训练方向，建议如下：

### 实验 E1：linear vs GBDT vs 小 MLP 的 residual 消融

- 从 v4 训练墙按动作阶段和规则类别平衡采样 5万–20万决策；
- 先建立 5,000–20,000 条 SlowExpertTeacher/人类确认的分歧数据；
- 比较 linear、GBDT、hidden 64/128 小 MLP；
- 单次≤30分钟、候选族≤2 GPU小时；
- 只按分歧 held-out、稀有规则保护和 override 稳定性选模型。

**它回答的问题。** Teacher 的一个已验证局部错误能否被小模型稳定修正。
**它不回答的问题。** 总体是否更强；仍须 100/400 墙。

### 实验 E2：Teacher 低 margin 分歧挖掘

- 从独立 1,000 墙抽样所有 Teacher top-2 分差小的决策；
- 按动作阶段和规则类别聚合，不先看终局；
- 使用一个精确局部 oracle 或盲态人类复核；
- 选择覆盖率最高、判断最一致的一类。

**它回答的问题。** 是否存在可教给 residual 的局部 Teacher 盲点。

### 实验 E3：单类别 gated residual

- 只训练 E2 类别；
- 零 residual 严格等于 Teacher；
- ensemble 3 seeds；
- 只在三成员一致且 margin 通过时 override；
- 预注册 override 率和 100 墙。

**它回答的问题。** “规则 → hybrid neural”的中间态是否能产生可测的小幅优势。

### 实验 E4：低成本规则/评测 benchmark

- 16/64/128 并行墙；
- Teacher-only、小 MLP hybrid 两种负载；
- 串/并行 hash parity；
- 输出 games/s、steps/s、CPU 占用和小模型推理延迟。

**它回答的问题。** 100/400 墙和轻量 DAgger 的真实运行成本。

建议执行顺序：`先 E2 建分歧集 → E3 建单类 slow expert → E1 选最小模型 → gated hybrid → E4/100 墙。`

---

## 13. 对“厦门麻将是不是比抚州麻将更复杂”的最终判断

当前证据不足以得出“厦门规则本质上更难训练”的结论。更可靠的分解是：

1. **公开抚州项目的成功系统不是纯神经网络。** 它保留 Teacher、lookahead、风险与进展，是与规则之间的中间态。
2. **厦门项目已经有大量 Teacher 数据，但缺少 Teacher 之外的高质量改进标签。** 这是质量/支持问题，不是简单行数问题。
3. **厦门特殊规则造成更明显的长尾和动作阶段异质性。** 这会放大总体 accuracy 的误导性，并要求分阶段模型与课程审计。
4. **本项目对信息泄露、独立墙和 selection/terminal 的要求更严格。** 很多看似正向的离线方法在严格实战下被否决，这不表示项目“训练不动”，而表示旧方法没有真正证明提升。
5. **两项目没有统一规则、同算力、同对手、同评测的 head-to-head。** 因此不能把迁移速度差直接归因为游戏复杂度。

所以最合理回答是：**不是“厦门太复杂所以神经网络不行”，而是目前缺少一条从规则 Teacher 到独立改进信号，再到安全 neural residual，再到 on-policy RL 的完整证据链。中间态模型是必要的；数据也确实不足，但不足的是多样、可辨识、符合信息集的训练信号。**

---

## 14. 研究纪律清单

每个新候选在运行前必须写清：

```text
hypothesis:
what_changes_vs_previous_failed_route:
actor_visible_fields:
training_only_fields:
behavior_policy_and_propensity:
independent_unit: physical_wall
train_validation_selection_terminal_seeds:
opponent_lineup:
override_rate_target:
offline_primary_metric:
100_wall_gate:
400_wall_confirmation:
stop_condition:
artifacts_and_checkpoint_status:
```

强制原则：

- 新算法名不等于新假设；如果数据、目标和行为分布没变，就可能只是旧失败的换壳。
- 先说明估计对象，再选择网络；先保证标签正确，再增加容量。
- 任何 oracle 的完整信息只在训练进程内存在，部署 actor 的字段必须有自动化否定测试。
- 每个 selection 只使用一次；失败候选不通过改变阈值复活。
- 100 墙用于筛选，不用于包装微弱点估计；400 墙仍跨零就保持基线。
- 外部代码、权重、牌谱先做许可证、规则、数据授权和信息边界审计。

---

## 15. 参考资料与迁移边界

### 麻将与中文牌类

- Li et al., [Suphx: Mastering Mahjong with Deep Reinforcement Learning](https://arxiv.org/abs/2003.13590)。迁移多阶段训练、oracle 隔离和辅助目标思想；不迁移日麻规则/权重。
- Zhao & Holden, [Building a 3-Player Mahjong AI using Deep Reinforcement Learning](https://arxiv.org/abs/2202.12847)。迁移动作分阶段和监督→RL 编排；不迁移三麻规则。
- Li et al., [Tjong: A transformer-based Mahjong AI via hierarchical decision-making and fan backward](https://doi.org/10.1049/cit2.12298)。迁移层次化输出和公平架构消融；不把其 Transformer 结论直接外推到厦门。
- Nishimori et al., [MahJax](https://arxiv.org/abs/2605.20577)。迁移向量化环境和吞吐优先思想；不迁移日麻规则。
- Zha et al., [DouZero](https://arxiv.org/abs/2106.06135)。迁移动作编码、Monte-Carlo actor 和并行系统；不迁移斗地主收益结构。
- Zhang et al., [PerfectDou](https://arxiv.org/abs/2203.16406)。迁移 PTIE 的隔离原则；训练特权信息不得进入 actor。
- [YaxinCui/fuzhou-mahjong-ai 技术报告](https://github.com/YaxinCui/fuzhou-mahjong-ai/blob/main/docs/technical_training_report_20260520.md)。迁移 hybrid 阶梯、DAgger 和评测纪律；规则、源码、权重与牌谱均不直接混用。

### 模仿、离线与在线强化学习

- Ross et al., [DAgger](https://proceedings.mlr.press/v15/ross11a.html)。用于候选访问状态的数据聚合；同一 Teacher 标签仍有上限。
- Kumar et al., [CQL](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)。用于有覆盖数据上的保守 Q；不修复无支持动作。
- Kostrikov et al., [IQL](https://arxiv.org/abs/2110.06169)。用于保守 advantage-weighted BC；前置是已覆盖动作价值可辨识。
- Jiang & Li, [Doubly Robust OPE](https://proceedings.mlr.press/v48/jiang16.html)。用于记录 propensity 后的诊断；有限样本方差和支持集仍是硬门槛。
- Burch et al., [AIVAT](https://arxiv.org/abs/1612.06915)。作为未来评测降方差候选；不替代正式同墙实战。
- Fan & Farina, [VRPO](https://arxiv.org/abs/2605.19235)。只有合格 Q critic 后才做 PPO A/B。
- Patwa, [Self-Play RL under Imperfect Information in Big 2](https://arxiv.org/abs/2605.28863)。作为四人环境的 current self-play/熵消融信号；不外推为厦门结论。

### 不完全信息博弈与搜索

- Brown et al., [Deep CFR](https://arxiv.org/abs/1811.00164)。只在局部/toy 信息集研究；不宣称四人全局收敛。
- Heinrich & Silver, [NFSP](https://arxiv.org/abs/1603.01121)。迁移 average/best-response 分离；不继承两人零和保证。
- Brown et al., [ReBeL](https://arxiv.org/abs/2007.13544)。迁移 public belief 建模；当前先过 history filtering 与 value 校准。
- Schmid et al., [Student of Games](https://arxiv.org/abs/2112.03178)。作为长期 search+learning 参照。
- Solinas et al., [History Filtering in Imperfect Information Games](https://arxiv.org/abs/2311.14651)。用于理解公开历史相容隐藏世界生成的复杂度。
- Vinyals et al., [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z)。迁移“强初始化 + league + exploiters”的职责分离；不照搬规模。

---

## 16. 一句话决策

**先让系统知道 Teacher 在哪里错、为什么错，再训练一个只在这些地方有资格偏离 Teacher 的小模型；等它真的在独立牌墙上赢了，再给它更多自由、更多对手和更复杂的 RL。**
