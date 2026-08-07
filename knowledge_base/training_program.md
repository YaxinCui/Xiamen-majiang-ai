# 更强厦门麻将 AI：训练方案与验收门槛

当前资产：经典厦门规则引擎、可解释 Teacher、Teacher 轨迹导出、76 维合法候选动作
MLP 和游金课程。当前 MLP 的 Teacher 动作一致率不能视为实战强度。

## 当前冻结选择（2026-08-07）

网页默认始终是规则 Teacher；目前没有神经 checkpoint 具备部署或下一轮训练起点资格。run3 的早期独立 80 墙
点估计虽为 **+1.34 ± 1.38** 分/局（95% CI **[−1.36, +4.04]**），但随后冻结参数在全新 400 墙终检为
**−0.855 ± 0.645**（**[−2.119, +0.409]**），不通过正向下界且点估计转负。run4 则为
**−2.38 ± 1.13**（**[−4.59, −0.17]**），与 run3 同墙配对差 **−3.72 ± 1.57**（**[−6.79, −0.64]**）。
两者均只保留为历史可复核实验，不能作为默认、数据 Teacher 或新训练的“最强基线”。后续不得再把“更高的
混合轨迹动作一致率”当作选择 checkpoint 的依据；新候选必须先在未参与调参的牌墙上通过正向下界。

## 目标架构

```text
规则引擎 ──合法候选动作──> 策略网络（仅公开信息 + 本家手牌）
    │                              │
    ├──执行、结算、回放──> 对局日志 ┴──> 监督学习 / DAgger / 自博弈 RL
    │                                             │
    └──仅公开状态 + 本家手牌──────────────> 价值与风险辅助头
```

策略网络可以给候选动作排序，但没有能力绕过规则。策略与 value 均只使用同一观察快照，
以保证训练、评测和网页推理一致；未来 oracle 辅助任务只能被蒸馏，不能进入部署输入。

## 阶段与验收

| 阶段 | 要实现的能力 | 通过门槛 | 不通过时的处理 |
| --- | --- | --- | --- |
| A. 可重复评测 | MLP/Teacher/随机策略在固定种子、四座轮换下自动打完；支持固定三人对手阵容。 | 零非法动作、零安全循环超限、每局可回放；先跑 400 个以上配对局。候选与参考必须使用同一阵容标签。 | 先修规则/代理接口，禁止调参。 |
| B. 强监督 | PyTorch 多头策略—价值网络；Teacher 与经授权人类牌谱混合训练。 | 与 Teacher 的保留集动作一致率提高，且 A 的净得分置信区间不劣于 Teacher。 | 检查特征、动作类别与样本覆盖；补游金/杠/响应课程。 |
| C. DAgger | 候选一席对三名冻结 Teacher；Teacher 标注候选实际访问到的状态。 | 同一副牌墙的四座轮换必须在同一数据分区；每轮都在独立种子上评测。 | 提高 Teacher 占比或对罕见规则过采样。 |
| D. 动作价值改进 | 候选一席访问真实部署局面；对每个合法动作反事实 rollout 到结算，学习 `Q^π(s,a)` 的软偏好。 | 四座轮换同组切分；目标只导出本家/公开观察与动作分数；批量/串行在固定分支上精确等价；动作价值留出集不少于 32 条，最终仍在全新牌墙配对。 | 先提高 rollout 吞吐和 belief world 数，再扩大冻结对手池或降低高方差目标权重；禁止把一次分支结果当 oracle。 |
| E. 自博弈 RL | actor-critic / PPO，动作掩码；先单局净得分，后多局累计分差。 | 相对 Teacher **和**固定异质阵容的换座配对评测均须通过正向下界；标准误按同一牌墙的四座轮换聚合。 | 回滚到最佳冻结检查点；检查 reward 与 opponent pool。 |
| F. 规模化 | 纯函数批量环境、GPU rollout、对手池和检查点联赛。 | 吞吐、规则回放和结果统计均可复现。 | 先保持 CPU DAgger，不为 GPU 重写未验证规则。 |

## 模型演进

1. **v1（已完成）**：76 维候选动作 MLP，模仿 Teacher，零依赖。
2. **v1.1（进行中）**：80 维候选动作 MLP，增加本家可见的候选弃牌后牌效与听牌前瞻；
   先通过换座评测验证其是否减小与 Teacher 的差距。
3. **v2（已完成首版）**：PyTorch policy-value 网络。输入含本家手牌计数、四家河、公开副露、
   金/白板代理、花、墙余、座位/庄家、游金与锁定状态、最近动作序列。输出为候选动作
   分数、局内净得分价值、胜率/放铳风险等辅助预测。
4. **v3（已做首轮反证）**：Transformer 与从 MLP 迁移的序列残差均能编码按时间排序的
   公开事件，却未通过独立实战评测。下一次只在更大 online 数据、冻结对手池和严格
   checkpoint 选择下重试；候选动作仍由规则引擎生成。
5. **v4（进行中）**：反事实动作价值蒸馏。私有模拟状态只用于运行分支并生成终局分数，
   导出与 actor 输入仍仅包含本家手牌及公开信息；策略头拟合每个合法动作的软偏好，而非
   单一 Teacher 标签。仅当独立配对收益通过时才接受。
6. **v5**：在通过 v4 验收的冻结候选上，做小规模 self-play PPO 与 frozen-opponent league。

## 数据集与奖励

- **切分**：按完整物理牌墙的 opaque group ID 切分，绝不按单个决策随机切分；训练 JSONL
  不保存可重建牌墙的 seed。候选四座轮换必须共享同一 group。
- **样本权重**：普通弃牌不能淹没胡、吃碰杠、荣牌跟打和游金；为每类动作记录数量与
  准确率。
- **即时奖励**：本局净得分变化、胡牌、放铳、荒庄、游金/三金等均记录，但不手工把
  牌效分数伪装成最终真值。
- **动作价值标签**：每个动作在相同的快照、相同 replicate 的冻结对手下分别结算；保存
  对齐合法动作顺序的 `action_values`。私有牌墙和他家手牌只能存在于 collector 内存中，
  不能写进 JSONL、特征或 source metadata。重复 rollout 的方差必须在实验报告中检查。
- **信息集重采样**：默认反事实分支只适合管线验证；强度实验应启用 belief resample，在固定
  本家手牌和公开事实后重抽未知墙、暗手、非本家花和暗杠牌面。当前为公开先验而非按对手历史
  行为加权的完整后验，因此仍须依赖独立实战评测，而不能声称已得到精确博弈价值。
- **价值可辨识度**：所有合法动作的结算分相同，不能提供策略排序信号；训练默认按动作价值
  跨度下调这类样本。多次 rollout 时，按 action-wise 标准误进一步下调高方差样本，而不把
  噪声较大的单局反事实结果当作确定标签。
- **双重动作目标**：反事实数据同时训练软偏好策略头与逐动作 Q 回归头。策略头学习相对选择
  概率；Q 头回归归一化终局净分，保留收益量级，并单独报告 Huber、MAE 与 argmax 指标。Q 头
  的部署模式必须在全新牌墙的四座轮换评测中单独证明，不能由训练损失推断。
- **长期奖励**：固定局数牌局结束时的累计分差或名次；庄家连庄状态必须留在观察中。
- **防止投机**：评测必须同时报告胡率、放铳率、平均净得分、游金触发/决策正确率和
  非法动作数。

### 人类数据的本地采集边界（2026-08-07）

为使“胜过人类”成为可验证目标，网页现可通过显式 `--human-log` 参数把完成的本地试玩局写到
`local_human_data/`。每条记录只含该玩家的可见状态、当时合法动作、实际选择、公开 action 历史和**本局**
分差；不含牌墙顺序、对手暗手、随机种子、账号、网络标识或时间戳。该目录被 Git 忽略，默认关闭，且
`collector=local_human_opt_in` 与现有 Teacher/DAgger 来源隔离。若对手是显式 checkpoint，记录只写其
内容 SHA-256（不写本地路径），以防把不同候选混进同一强度对照。

这些记录不是自动的“强人类标签”：使用前须确认记录者水平、规则档位、完整性和重复局，并按完整牌局
划分 train/validation/test。`audit_human_trajectories.py` 会先拒绝混合规则/对手、重复局、重放种子与
私有字段；对于结构合格的完整局，报告人类本局平均分差、标准误、正态近似 95% 区间、胡率和流局率。它们
只描述这批记录，不能代替独立对人比赛的统计检验；通过后仍须人工确认数据质量。优先用独立人类留出局报告
动作一致性和最终对局表现；没有足量、多来源、可审计的人类数据时，不能宣称模型已战胜人类。

训练入口也执行这条边界：任何带 `collector=local_human_opt_in` 的文件在默认情况下直接报错，不能因被放进
`--additional-train` 而悄悄混入。只有显式 `--allow-local-human-data` 后，训练器才会对**所有**人类输入合并运行
同一结构审计（最少局数、单一规则/对手、无私有字段、无重复）；仍须由操作者承担人工质量复核。checkpoint report
只保存无路径的聚合审计、人工授权标记和人类样本权重，训练目标标为 `hard_executed_human_action_after_manual_review`。
若任一 split 缺少人类轨迹，训练器同样拒绝，以免把人类训练集与 Teacher 验证集混用后把动作准确率误当成人类泛化；
三份人类 split 仍须由完整牌局、且不得重复的记录构成。
人类终局分数暂不进入 value 回归：这些有限的、单一对手配置的行为记录可以监督行动，但不足以当作稳健的状态价值真值。

### 全历史 belief 的当前工程门槛（2026-08-07）

已验证的 sequential SMC 与“交换未知暗牌”的 repair proposal 都会在多事件公开历史上严重退化，因此不能进入
动作价值 collector。当前只完成一个 audit-only 的正向基元：core 开局本家弃牌后立即发生的他家吃／碰／明杠，
可以把副露所需暗牌作为**初始手牌**条件，以多元超几何分布直接采样并精确计算 `p/q`。它不涉及后续摸牌或弃牌，
更不是完整 posterior。

后续每扩大一个事件类型，都必须先给出 proposal density、在可枚举小牌墙与精确 posterior 对照、报告未重采样
ESS/接受率和不泄露私有世界的回归测试。只有连续事件通过这些门槛，才可新建独立 multi-world value 消融；禁止把
当前局部或 repair audit 直接接入训练、选牌或网页。

其中补花必须作为公开 transition 先行建模：它改变可见花数量，不能仅在 private wall 内悄悄跳过。普通摸牌的
`draw.tiles` 现已按顺序记录公开补花牌面（不记录随后可打出的暗摸牌）。小牌墙先验证 wall-only 条件采样，随后将
未知对手起手牌／花槽与“补花→暗摸底牌→公开弃牌可行”联立为一个 post-setup reference proposal：其组合概率与
条件抽样都通过微型枚举回归。开局端已补上骰位翻金：`gold_indicator_index` 与引擎共用同一环形扫描规则，固定墙的
条件 sampler 及“翻金 + 庄家已知首摸”联合 sampler 均与穷举小牌墙对照；重建测试覆盖首摸补花并验证本家可见状态、
金指示牌和 144 张物理牌守恒。opening 已进一步与“庄家首弃→下家公开补花／暗摸→公开弃牌可行”合成单一
per-particle `p/q` proposal，并在微型牌墙穷举校准。固定 core seed 271 的 256 粒子审计达到 256/256 结构重放，
但冻结 Teacher 行为权重 ESS 只有 **9.55/256（3.7%）**（旧 post-setup reference 为 17.76/256），不接入
collector/Q/网页。普通全历史重放继续拒绝未获专门授权的花牌 transition；补杠／明杠后的 replacement draw 也尚未
形成公开 draw transition。这是保护正确性的限制，不是缺失事件可以忽略的许可。

为排除“确定性 Teacher likelihood 过硬”这一捷径，另训练一个 core Teacher 概率代理（100 训练局、独立 40 局，
top-1 76.84%、交叉熵 0.725）。在同一 64 粒子、8 个公开事件 sequential-SMC smoke 上，原 Teacher 的最终 ESS 为
3.01，代理为 2.86；两者均只条件到第 3 个事件后就因结构不一致全体失败。故当前退化主要来自 proposal 未生成可行
隐藏世界，而非行为概率温度；该代理不得接入 SMC、collector、Q 或网页。下一步仍须先完成有显式密度并通过 toy
exact posterior 的结构 proposal，不能继续调 softmax 温度或扩大行为代理。

### 开局 claim proposal 的精确校准与新瓶颈（2026-08-07）

“开局翻金＋庄家首摸＋下家补花/摸打” proposal 的 toy 测试现不再只检查事件总概率：对每个采样 world 用其
`p/q` 加权后，暗手和翻金后墙尾的边缘分布均须与标记物理牌穷举 posterior 一致。另新增“开局翻金＋首个
吃/碰/明杠所需初始暗手”的联合 proposal；当 claim 后立即弃牌时，弃牌面也作为同一初始手牌多重集约束。
两个条件都在独立可枚举小牌墙上通过总归一化和加权后验边缘回归，因而不存在用 tile swap 或合法性过滤冒充
密度的步骤。

审计现在分开报告 structural ESS（只有显式 `p/q`）和 total ESS（再乘冻结策略行为似然）。固定 core `seed=271`、
256 粒子、audit RNG `202608504` 的“一次对手摸打”前缀为 256/256 结构重放，structural ESS **248.76/256
(97.2%)**，但 total ESS **15.28/256 (6.0%)**。这证明该窄 proposal 的密度本身并非主要退化来源，低 ESS 来自
公开动作选择的行为相容性。

固定 core `seed=2` 的首个 opening claim 在 256 粒子、audit RNG `202608505` 下为 255/256 接受，structural ESS
**246.33/255 (96.6%)**、total ESS **187.88/255 (73.7%)**；它只覆盖“本家首弃→他家 claim”的两动作前缀，尚不构成
可采集 belief。首次把该 prefix 直接截到 claimant 弃牌后得到的 32/256 “接受率”经追踪被判定为**无效实验**：若该弃牌
无人可响应，引擎会在同一原子 transition 自动产生下一次摸牌，旧 recognizer 错在 draw 前截断，因而把额外事件标作
`public_event_mismatch`。现已加入原子边界 guard；这类状态在具备下一张公开摸牌的精确 density 前一律拒绝。

在通过该 guard 的独立 core `seed=67` claim→discard 前缀上，256 粒子、audit RNG `202608507` 为 **256/256** 结构重放，
structural ESS **250.74/256 (97.9%)**，但 total ESS 仅 **33.30/256 (13.0%)**。因此更正后仍可确认：短前缀的结构
proposal 已健康，而动作历史 likelihood 才是此路线的下一瓶颈；不能将先前 224 次边界错误解释成行为不相容。

为使下一步仍保留精确 density，新增 tile-factor 多元超几何 proposal：对手暗手的每一种 tile face 可乘一个正权重，
归一化常数以计数 DP 精确计算并返回条件 prior/proposal 比；独立 toy 牌墙验证其采样分布与加权后原始超几何 posterior。
这允许将未来的线性行为能量模型作为**proposal**，同时仍以冻结行为 likelihood 作目标权重，避免把能量分数误当概率。

第一个、预先固定的低容量 probe 只将“claim 后公开弃牌”同面额的额外副本权重设为 `{1, 1.5, 2, 3}`；结构墙组固定为
core `67, 139, 201, 275`（按是否存在原子 claim→discard 前缀筛选，而未查看 ESS/收益），每项 256 粒子。四项平均
structural ESS fraction 依次为 **97.87%、96.55%、94.41%、88.01%**，total ESS fraction 仅为
**13.34%、13.99%、12.52%、9.79%**；每项均有低 ESS 墙，最高均值也不足以进入现有 20% 的局部 belief 健康门槛。
因此这条“只偏置弃牌同面额”的 action proposal 家族整体否决；不会因 1.5 的微小同组均值差而选择它、重跑或接入任何
collector。保留 DP primitive 作为未来经独立数据训练的、预注册线性能量 proposal 的正确密度基础。

该线性能量后继实验只使用独立 core Teacher 自博弈（seed `202608520`，300 训练局／10,476 决策，100 验证局／3,558
决策）训练 `RulePolicyModel`；验证动作一致率 **67.09%**。对一个固定公开弃牌，线性模型的“弃牌目标 × 暗手面额计数”
项可精确转换为 tile factor，公共项自然抵消；factor 的 DP 校正仍为严格 `p/q`。模型仅作为 audit proposal，SHA-256
`80587fcbc108888b884a39b59b4462040c6d54e618e88aa69093042adc0edfd3`，不是可选牌 checkpoint。

新的 selection 墙组固定为 core `390, 492, 499, 585`，与模型训练及前一因子实验无交集；开启前固定 scale
`{0, 0.25, 0.5, 1}`、log-factor clip `2`、每墙 256 粒子。平均 total ESS fraction 为
**23.58%、31.87%、33.19%、23.95%**，故按预注册的“最大平均 total ESS”选择 scale **0.5**；不得再在这些墙上
调 scale 或 clip。随后只在新 core `seed=707` 的原子 prefix 上做 1,024 粒子终检：接受 1,023/1,024，total ESS
**226.17/1,023 (22.1%)**，但 structural ESS **158.26/1,023 (15.5%)**，未通过 structural 与 total ESS 均至少 20%
的 joint health gate。该线性能量 proposal 因此拒绝；不会进入 SMC、collector、Q、训练、网页或下一轮模型初始化。

因此该路径继续停留在 audit：不接入 SMC、collector、Q、训练或网页。下一项研究必须先为行为相容的隐藏手牌
proposal 给出可计算密度与 toy exact posterior；不能对确定性 Teacher 的“选中动作”直接 rejection 后把未知接受率
当作 `p/q`，也不能以这三个短前缀的高 structural ESS 声称已得到全历史 posterior。

### Teacher 单点干预 response 数据（当前因果来源）

run4-based response outcome ensemble 已被全新实战否决，不能再把 run4 作为改善 Teacher 的桥梁。collector 现支持
`--teacher-base`：候选座位在干预前后均完全执行 `HeuristicTeacherAgent`，每局最多随机选中一个 response 决策，
以 `ε × uniform + (1−ε) × Teacher` 采样并记录该**已执行**动作的精确 propensity。对局另三座仍是 Teacher；JSONL
只保存本家/公开观察、实际动作、propensity 和终局分数，不保存墙、暗手或随机种子。

这可识别“一个受支持 response 替代 + Teacher 后缀”的局部因果回报，但不能直接识别多次连续修改后的策略价值。
`audit_teacher_response_intervention_ope.py` 已实现：它只接收 `base_policy=heuristic_teacher` 的 held-out 轨迹，按
物理墙组汇总 IPS/DR、报告两侧 ESS，并用多个**不接触 test 墙**的 outcome checkpoint 构造保守 LCB 候选。只有单点
策略改动的 IPS 和 DR 保守下界均为正，才允许使用与收集协议一致的“一局至多一次 override”候选进行 200 墙筛选。
不得复用已被否决的 run4 selector、不得把 logged-action 校准或单点 OPE 直接称为完整对局强度。

首轮经典数据（800 物理墙）以全新的固定随机 policy anchor 训练 5 个 outcome 成员，避免把 run3/run4 的权重作为
新的训练起点；只有 outcome encoder 和三类 outcome head 可更新。113 个未接触训练／validation 的 test 墙中有 108
个发生有效随机 response 干预（282 条）。`max_afterstate_score_lcb_or_teacher` 在该 test 的 grouped IPS 为
−1.6428 分/局（95% CI [−10.2839, +6.9983]），DR 为 −0.7897（[−9.8425, +8.2632]）；虽有 target/base ESS
66.1/218.7，双下界均不为正，`ready_for_single_override_game_screen=false`。该 test 组已消耗，禁止再用它调 LCB
阈值、挑 ensemble 成员或重试 selector。下一轮必须预注册新候选，并使用新的 selector 与终检墙组。

第二轮预注册了四层墙组：1600 个全新物理墙先分 train/validation/held-out，再把 held-out 按完整墙组分为 selection
与 terminal；五个 outcome 成员训练时均显式 `--skip-test`，报告亦写 `terminal_test_read=false`。预先固定的
LCB 最小优势网格 `{0, 12, 24, 36}` 只在 200 个 selection 墙组（522 个随机 response）上筛选：0 分阈值的
IPS/DR 95% 下界为 −7.7110/−7.5706（override 35.63%）；12 分为 −1.5236/−1.1817（override 3.45%）；24 与
36 分均退化成无 override、下界 0。所有候选失败，选择器状态为 `selection_rejected_terminal_unread`，terminal
墙组从未读取。不得基于这一 selector 再新增阈值或改模型后重试；下一条路径须是新的历史表示／候选定义及全新四层墙组。

## 实验记录模板

每个实验目录应保存 `config.json`、`metrics.json`、`checkpoint`、`git_commit`、
`rules_profile`、`dataset_manifest.json` 与按种子汇总的配对评测。原始大体量轨迹可置于
Git 忽略目录或对象存储；可发布检查点应小、可加载、并带足够元数据复现。

## 最近执行顺序

1. 实现全自动换座配对评测，并用已提交 MLP 得到第一份对 Teacher 的实战基线。
2. 将 MLP 作为可切换网页 AI 接入，保留 Teacher 回退。
3. 基于评测中 MLP 实际访问的状态做第一轮 DAgger，不直接开始大规模 PPO。
4. DAgger 确认改善后，以冻结 Teacher 为对手执行小步 REINFORCE；仅使用结算净得分，
   并按初始牌墙轮换候选座位。
5. 已验证：批均值 REINFORCE 与 600 局线性 actor-critic 均未在独立配对评测中改善；
   不再继续调这两个候选。
6. 已实现共享表征的 PyTorch policy-value 首版、安全轨迹 v3、候选一席对三 Teacher 的
   online DAgger、neural PPO、变长合法动作 batch、冻结 checkpoint 对手池，以及安全导出的
   反事实动作价值轨迹。首轮大规模 PPO 仍未在独立实战中胜出；当前优先让动作价值数据在
   独立牌墙上证明收益，而不是只扩大同一 Teacher 模仿样本。任何新依赖先在项目虚拟环境内
   验证，不修改系统 Python。
7. 已补入零初始化、向后兼容的 candidate-MlP 动作 Q 头；下一次强度实验必须对 policy 选牌与
   Q 选牌分别做独立配对比较。若两者均未胜出，优先扩展按对手历史行为条件化的 belief 数据，
   不继续仅调节损失权重。
8. 已补入反事实分支的可选批量调度：只批量合并实现 `scores_batch` 的冻结策略推理，规则状态、
   RNG、动作掩码与结算仍逐分支隔离。下一项工程任务是以此吞吐基础实现并校准顺序粒子 belief，
   而不是立即训练更大网络。
