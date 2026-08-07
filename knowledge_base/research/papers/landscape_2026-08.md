# 棋牌与不完全信息博弈研究地图（2026-08）

本页是选题索引，不是“算法排行榜”。来源优先使用原论文、正式论文页面或原作者/维护者仓库；访问日期为 2026-08-07。
`可迁移等级` 是本项目判断：**A** 可立刻形成可审计小实验，**B** 先补数据/环境后再试，**C** 只作为架构参照。

## 一、麻将、斗地主与相近四人/多人牌类系统

| 资料 | 核心问题/贡献 | 可迁移等级与厦门边界 |
| --- | --- | --- |
| [Suphx](https://arxiv.org/abs/2003.13590) | 多人麻将的 global reward prediction、oracle guiding、run-time adaptation。 | **A：** 多任务标签与训练/部署隔离；日麻规则和得分不能移植。 |
| [Mortal](https://github.com/Equim-chan/Mortal) | Rust + 深度 RL 的立直麻将实战工程。 | **B：** 观察 Rust 规则核、推理/协议边界；代码为 AGPL，不能直接混入本仓库。 |
| [Kanachan](https://github.com/Cryolite/kanachan) | 牌谱标注、行为克隆、课程学习、DQN/离线 RL/Transformer 的麻将框架。 | **A：** 标注器与“annotation-vs-simulation”一致性测试模式；其牌谱和日麻动作语义不可用作厦门标签。 |
| [RiichiEnv](https://github.com/smly/RiichiEnv) | Rust 高吞吐 Gym 风格麻将环境及回放可视化。 | **B：** 环境吞吐和 agent adapter 设计；规则/协议并不等价厦门。 |
| [RLCard](https://arxiv.org/abs/1910.04376) | 卡牌环境、CFR/DQN/NFSP 基准和统一 trajectory 接口。 | **A：** agent/environment 解耦、动作 mask、可重复评测；内置麻将不是厦门实现。 |
| [DouZero](https://arxiv.org/abs/2106.06135) | 三人斗地主以动作编码、深度 Monte Carlo、并行 actor 处理巨大合法动作集。 | **A：** 合法动作候选编码与 actor-learner 并行模式；斗地主的合作阵营收益不同。 |
| [PerfectDou](https://arxiv.org/abs/2203.16406) | perfect-training-imperfect-execution（PTIE）蒸馏。 | **A：** 特权训练信息只能蒸馏/监督、部署只用可见信息；必须先做泄露单测。 |
| [DouZero+](https://arxiv.org/abs/2204.02558) | 对手建模与 coach-guided 学习改进斗地主。 | **B：** 对手风格分桶可作为诊断；不能假设厦门四家存在队友协作。 |
| [DouRN](https://arxiv.org/abs/2403.14102) | 以残差网络改进 DouZero，并进行多角色测试。 | **B：** 多座位/多角色评测优先于单一平均胜率；网络结构本身不构成路线。 |
| [DeltaDou](https://mlanthology.org/ijcai/2019/jiang2019ijcai-deltadou/) | 早期斗地主自博弈深度 RL 系统。 | **C：** 作为中文牌类自博弈的历史基线；不作为当前实现蓝图。 |
| [抚州麻将专项审计](../fuzhou_mahjong/) | 本项目已审计的“Teacher + lookahead + 神经策略”过渡案例。 | **A：** 只迁移其训练阶梯与评测纪律，规则、参数、牌谱不混用。 |

## 二、扑克与一般不完全信息博弈

| 资料 | 核心问题/贡献 | 可迁移等级与厦门边界 |
| --- | --- | --- |
| [CFR](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html) | 在不完全信息扩展式博弈中最小化反事实遗憾。 | **C：** 用于理解 information set；完整四人一般和不具原始两人零和保证。 |
| [MCCFR](https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html) | 以采样扩展 CFR 到大博弈树。 | **C：** 仅在局部 response 子博弈试验；不得声称全局求解。 |
| [CFR+](https://arxiv.org/abs/1407.5042) | CFR 的加速变体。 | **C：** 只作为小型可验证博弈的算法对照。 |
| [Deep CFR](https://arxiv.org/abs/1811.00164) | advantage/平均策略网络近似大规模信息集。 | **B：** 信息集编码、reservoir 数据管理可借鉴；不直接替换摸打策略。 |
| [Single Deep CFR](https://arxiv.org/abs/1901.07621) | 以单网络式策略重建减少一种近似来源。 | **C：** 比较研究素材，先于厦门全局 CFR 的必要性尚未成立。 |
| [NFSP](https://arxiv.org/abs/1603.01121) | best response 与平均策略并行学习。 | **B：** 形成 opponent league 的概念来源；没有四人厦门均衡保证。 |
| [DeepStack](https://arxiv.org/abs/1701.01724) | 递归子博弈求解和深度 value 在 heads-up poker 中结合。 | **C：** 启发“公开历史 + 局部求解”；局部收益与 belief 必须重建。 |
| [Libratus](https://www.science.org/doi/10.1126/science.aao1733) | 大规模两人无限注扑克的抽象/求解范式。 | **C：** 研究分阶段策略与安全子博弈；不迁移其抽象或结论。 |
| [Pluribus](https://www.science.org/doi/10.1126/science.aay2400) | 六人扑克中自博弈与有限前瞻的成功实例。 | **B：** 证明多人不完全信息值得做策略池和局部前瞻；论文亦不提供一般多人理论保证。 |
| [Student of Games](https://arxiv.org/abs/2112.03178) | 融合引导搜索、自博弈和博弈论推理。 | **C：** 中长期“搜索+学习”参照；从其基准到厦门规则有很大工程鸿沟。 |
| [ReBeL](https://arxiv.org/abs/2007.13544) | public state、belief 与子博弈搜索。 | **B：** 先建可审计 public belief；理论语境为两人零和。 |
| [MCCR](https://arxiv.org/abs/1812.07351) | continual resolving 的 Monte Carlo 版本。 | **C：** 用来研究在线局部求解接口；只对两人零和给出论证。 |
| [DecisionHoldem](https://arxiv.org/abs/2201.11580) | 面向多种对手的安全深度受限求解开源实现。 | **C：** 可读“对手范围 + 安全局部搜索”工程化；扑克动作空间不同。 |
| [多人不完全信息对手建模](https://arxiv.org/abs/2212.06027) | 在三人扑克通过重复交互利用对手观测。 | **B：** 对手类型只作为评测分层/受控特征，必须防止把对手暗手泄漏进 actor。 |

## 三、多人自博弈、种群与分布式训练

| 资料 | 核心问题/贡献 | 可迁移等级与厦门边界 |
| --- | --- | --- |
| [DeepNash](https://arxiv.org/abs/2206.15378) | Stratego 的无搜索多智能体自博弈与 R-NaD。 | **C：** 证明隐信息、长局、离散动作可用自博弈；其为两人零和，计算规模不可直接承担。 |
| [PSRO 综述](https://arxiv.org/abs/2403.02227) | 策略种群、response oracle、元博弈的系统化框架。 | **A：** 先做冻结对手池和鲁棒性矩阵；不把四人矩阵误叫 Nash。 |
| [Population Based Training](https://arxiv.org/abs/1711.09846) | 同时进化模型权重与超参数时间表。 | **B：** 在已有稳定训练后再小种群调 loss 权重；数据/评测不稳时不可用 PBT 掩盖问题。 |
| [IMPALA / V-trace](https://arxiv.org/abs/1802.01561) | actor/learner 解耦及策略滞后校正。 | **B：** 当模拟吞吐成为瓶颈时采用；当前先保证规则与数据契约。 |
| [AlphaStar 评述](https://arxiv.org/abs/1902.01724) | 联赛、竞争共演化和多样性训练的分析。 | **B：** 借鉴主策略/漏洞利用者/历史池的职责分离；不能照搬算力和游戏观测。 |
| [自博弈综述](https://arxiv.org/abs/2408.01072) | 归纳多种自博弈设计和棋牌实例。 | **B：** 作为路线检索入口，关键论文仍须回到一手来源。 |
| [OpenSpiel](https://arxiv.org/abs/1908.09453) | 覆盖 n-player、一般和、完美/不完全信息的研究框架。 | **A：** 用它的术语与基准组织小子博弈，不替换厦门规则核。 |
| [PettingZoo](https://arxiv.org/abs/2009.14471) | AEC 形式化顺序多智能体环境 API。 | **A：** 可用于 adapter 的回合/响应语义测试；规则结算仍以本项目引擎为准。 |

## 四、离线 RL、反事实评测与低方差对局评估

| 资料 | 核心问题/贡献 | 可迁移等级与厦门边界 |
| --- | --- | --- |
| [DR OPE](https://proceedings.mlr.press/v48/jiang16.html) | 顺序决策的 doubly robust 离线价值评估。 | **A：** 先记录真实 propensity/动作 mask/行为版本；无支持集时不使用。 |
| [FQE](https://proceedings.mlr.press/v48/thomasa16.html) | 数据高效的离策略策略评估。 | **B：** 可作为 DR 的价值模型对照；依旧受覆盖和隐藏信息错误制约。 |
| [AIVAT](https://arxiv.org/abs/1612.06915) | 用已知策略与启发 value 降低隐信息博弈评测方差。 | **B：** 候选与基线策略概率可追溯后，研究配对墙/控制变量；不得在未知策略上虚构无偏性。 |
| [CQL](https://arxiv.org/abs/2006.04779) | 用保守 Q 抑制离线数据分布外动作的过估计。 | **B：** 仅在海量、有覆盖、动作表示稳定后做离线 Q 对照；不能取代 Teacher 评测。 |
| [IQL](https://arxiv.org/abs/2110.06169) | 不直接查询数据外动作的离线策略改进。 | **B：** 适合“先不偏离行为数据太远”的候选；先验证固定规则数据覆盖。 |
| [distributionally robust DR OPE](https://proceedings.mlr.press/v162/kallus22a.html) | 考虑部署分布偏移的 DR 离线评估/学习。 | **C：** 当策略池带来显著对手分布漂移时再研究。 |
| [Decision Transformer](https://arxiv.org/abs/2106.01345) | 将离线决策视作回报条件序列建模。 | **C：** 可作为带行动历史的表示基线；高分轨迹稀少时不先做主线。 |

## 五、明确不作为当前主线的对照技术

| 资料 | 为什么保留 | 为什么暂不直接采用 |
| --- | --- | --- |
| [AlphaZero](https://arxiv.org/abs/1712.01815) / [MuZero](https://arxiv.org/abs/1911.08265) | 搜索—学习接口和消融设计的参考。 | 其核心成功设定偏完美信息；厦门首先缺公开 belief 与可控局部搜索。 |
| [MADDPG](https://arxiv.org/abs/1706.02275) / [QMIX](https://arxiv.org/abs/1803.11485) | 理解 centralized training / decentralized execution。 | QMIX 等主要针对协作价值分解，四人麻将不是统一团队回报；训练特权信息也不可进入部署 actor。 |
| [MAPPO](https://arxiv.org/abs/2103.01955) | 作为多人 PPO 的对照文献。 | 当前从零 PPO 在本项目稀疏、长程、一般和回报下尚无通过门槛的证据。 |

## 使用流程

1. 先从本页找到相近问题，再读原论文/仓库的实现与限制。
2. 将候选拆成最小可证伪假设，写入 `experiments_and_reproduction/`。
3. 明确 actor 可见字段、数据来源、许可证和规则差异。
4. 只有在冻结基线、四座轮换、物理墙独立的对局中出现正向证据，才提升方法优先级。

因此，本页的 35+ 条来源扩大的是“可研究空间”，不是当前模型已经拥有的能力清单。
