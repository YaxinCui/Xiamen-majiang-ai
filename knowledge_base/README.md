# 厦门麻将 AI 知识库

本目录缓存本项目训练决策所依赖的可追溯知识；它不是模型权重、训练数据或
第三方源码的镜像。每项结论都区分「可直接采用」「仅作方法参考」与「不可复用」，
以免把日麻的规则、数据和结算方式误移植到厦门麻将。

## 文档索引

- [麻将 AI 前沿与开源实现](mahjong_ai_frontier_2026.md)：论文、开源系统、
  许可证边界，以及它们对本项目的实际启示。
- [研究知识库](research/README.md)：按论文、开源工程、强化学习技术、感悟和抚州麻将
  分区维护的长期研究资料；常规评测约定为 100 副物理牌墙。
- [深度训练路线空间](research/technical_routes/xiamen_deep_training_route_space_2026-08.md)：
  从现有反证出发，推演 42 条基础设施、规则、混合模型、离线 RL、自博弈与搜索路线，
  并记录成功前提、失败原因、最小证伪实验和推进顺序。
- [资源受限训练计划](research/technical_routes/resource_constrained_training_plan_2026-08-08.md)：
  当前执行事实源；停止 Transformer 和百万级同源数据扩张，改走 SlowExpertTeacher、固定历史摘要、
  linear/GBDT/小 MLP residual 与 1%–5% 高置信门控。
- [前沿复盘与下一步](frontier_followup_2026-08.md)：补充调研后，将公开工作转换为
  可执行的 belief、数据、离线 RL 与规模化路线。
- [不完全信息动作价值](imperfect_information_research.md)：信息集价值和历史条件化
  belief 的理论边界与项目验收条件。
- [更强 AI 训练方案](training_program.md)：从现有规则 Teacher 基线到自博弈
  强化学习的分阶段训练和验收标准。
- [训练实验日志](experiment_log.md)：已运行实验、独立评测与淘汰结论。
- [吃碰后续牌效 Teacher 协议](meld_continuation_teacher_v1_protocol.md)：响应后强制弃牌的
  专家规则候选、独立筛选与未通过结论。
- [精确向听吃碰响应 Teacher v1](deficiency_meld_teacher_v1_protocol.md)：用五面子精确向听、直接听口和真实响应后
  弃牌的唯一词典序纠正 frozen 吃碰过决策；全新 100 墙未通过，转向独立人工响应纠错。
- [Teacher 响应因果错误地图 v1](teacher_response_causal_error_map_v1_protocol.md)：用既有单点随机干预将
  吃／碰按公开阶段分层；确认削弱碰牌有负向证据，而 `chi→pass` 正点估计在独立 validation 仍跨零，按协议停止。
- [Teacher-clone reference-KL 小 MLP PPO v2](teacher_reference_kl_ppo_v2_protocol.md)：修复 PPO 胜局统计，量化旧
  anchor 的 3,000 倍量纲失配；强小 MLP + reference KL 的 4,096 局 mechanics 通过，但唯一 100 墙强度门槛未过，
  配置已冻结且未扩大训练。
- [低分差 top-2 随机干预与因果 residual v1](low_margin_top2_causal_residual_v1_protocol.md)：每局第一次低分差普通
  弃牌以 0.5/0.5 执行 Teacher top-1/top-2；4,000 train／2,000 validation 墙完成，四门均拒绝且 terminal 未采集。
- [低分差 top-2 同墙控制变量 residual v2](low_margin_top2_wall_control_v2_protocol.md)：利用同墙另外三条座位轮换作
  零均值 control variate；独立 2,000 墙上标准误显著下降，但四个校正下界仍跨零，整个 top-2 residual 家族冻结。
- [低分差 top-2 可解释因果错误地图](low_margin_interpretable_causal_map_v1_protocol.md)：在三个已消耗历史分区上固定扫描
  45 个 actor-visible 类别；虽有 22 类跨分区同号，但全族校正后无确认候选，不再追加 top-2 数据。
- [精确一摸听牌平分裁决 Teacher 协议](exact_one_draw_tenpai_teacher_v1_protocol.md)：只在 Teacher
  原评分近似平分时介入的公开信息规则候选、其独立筛选门槛与边界。
- [公开听牌价值 Teacher v1 预注册协议](public_tenpai_value_teacher_v1_protocol.md)：仅在近分直接听牌时，
  用公开剩余张数与可见立即自摸分做窄门控纠正。
- [公开知识向听 Teacher v1](knowledge_aware_deficiency_teacher_v1_protocol.md)：精确五面子向听、公开进张
  与 100 墙未通过结论。
- [公开向听—有效进张 Pareto Teacher v2](public_pareto_deficiency_teacher_v2_protocol.md)：仅确认 v1 中
  向听严格改善且公开有效进张不下降的近分弃牌，并在全新 100 墙配对筛选。
- [Teacher 完全并列公开进展消歧 v1](public_progress_tiebreak_teacher_v1_protocol.md)：只在 frozen 最高分
  完全并列时，用向听与公开有效进张 Pareto 支配替换牌号顺序，并在全新 100 墙直接证伪。
- [参数化弃牌规则 Teacher v1](parametric_discard_teacher_v1_protocol.md)：CPU 演化校准六个线性权重；
  训练正收益未泛化到独立 validation。
- [两摸听牌可达性 SlowExpert v1](two_draw_tenpai_reach_teacher_v1_protocol.md)：32 个共同分层公开摸牌场景；
  100 墙点估计为正但置信下界未通过。
- [TwoDraw 定向二元干预 v1](two_draw_targeted_intervention_v1_protocol.md)：只在每局首个 TwoDraw/Teacher
  普通弃牌分歧以 0.5/0.5 随机一次，用高支持单点因果数据判断该局部规则是否值得进入 honest gate 学习。
- [两摸听牌精确确认 Teacher v2](exact_two_draw_tenpai_reach_v2_protocol.md)：固定 v1 只负责 proposal，精确无放回
  两步 DP 只负责确认或删除 override；使用全新 100 墙决定是否有资格成为离线标签源。
- [首杠机会定向二元干预 v1](targeted_kan_intervention_v1_protocol.md)：在每条轨迹第一个 Teacher 杠机会以 0.5/0.5
  随机杠或 frozen 非杠 fallback，用三类 Bonferroni 门槛判断无条件杠是否存在可学习盲点。
- [Teacher + 小 MLP 置信门控 v1](confidence_gated_teacher_v1_protocol.md)：把旧小 MLP 限制为约 2% 的
  普通弃牌 residual，并以全新 100/400 墙逐级筛选。
- [32-world 局部 advantage 标签试验](belief_advantage_label_pilot_v1_protocol.md)：用共享信息集 world 检查
  局部 paired 标签能否稳定区分好坏，先过数据门槛再允许训练。
- [精确一摸 public-fix v2 标签试验](exact_one_draw_belief_advantage_v2_protocol.md)：修正他家暗杠牌面泄漏后，
  在唯一保持正点估计的候选源上重新做 32-world paired 标签门槛。
- [真人—Teacher 纠错数据 v1](human_teacher_correction_v1_protocol.md)：在本地 opt-in 真人动作旁保存同状态
  frozen Teacher 参考索引，以聚合门槛筛选轻量 residual 数据；训练与真人终检严格隔离。
- [真人纠错小 MLP + Teacher gate v1](human_teacher_residual_gate_v1_protocol.md)：fresh 小 MLP 只学习真人纠错，
  validation 在 1%/2%/5% 覆盖中选门，真人 test 通过后才允许 100 墙 Teacher 筛选。
- [冻结 AI 对真人强度基准 v1](human_strength_benchmark_v1_protocol.md)：evaluation-only 真人牌局、session-blocked
  统计、三 AI 席合计与单席均值口径，以及“超过本地评测群体”与“超过一般人类”的声明边界。
- [低分歧专家纠错审阅 + 小 MLP v1](human_correction_review_v1_protocol.md)：从 Teacher 近分弃牌状态建立
  actor-visible 本地题库，以失败 SlowExpert 只作盲主动选题、提交前隐藏全部自动答案，再以 group 切分的 confirmed
  标签训练 CPU 小 MLP，并物理隔离 test。
- [完全并列盲态二选一 + pairwise residual v1](exact_tie_pairwise_review_v1_protocol.md)：把 Teacher 默认牌与一个
  完全同分替代牌匿名二选一；固定 227 题／227 groups，只对展示动作对训练 hidden-64 CPU 排序模型。
- [完全并列 source-world／未来墙／因果确认](exact_tie_source_world_and_causal_v1_protocol.md)：900 墙单暗手配对标签、
  feature-v4 结构化小模型、四未来牌序平均和 100 墙高支持二元确认均未过门槛；记录该自动 residual 族为何被冻结。
- [吃／碰／过人工纠错审阅 v1](human_response_review_v1_protocol.md)：独立 400 题 actor-visible response queue；
  失败精确向听候选只排序 219 道分歧难题，人工提交前隐藏全部自动答案，未来只训练独立小型 response residual。
- [外部项目迁移审计](external_transfer_audit_2026-08.md)：对指定抚州／南城项目的
  规则、许可证和可迁移工程方法的边界结论。
- [训练数据来源审计](data_source_audit_2026-08.md)：已核验来源、隐私／规则准入契约
  和当前数据缺口。
- [厦门规则调研](../RULES_RESEARCH.md)：当前经典规则档位的来源和实现边界。

## 使用准则

1. 规则引擎是唯一的合法动作与结算权威；学习策略只能给引擎给出的动作排序。
2. 训练、验证和对局评测按随机种子隔离。模型不可访问牌墙顺序或其他玩家暗牌。
3. 所有「更强」结论须来自同牌墙、换座位的多局评测，而不是模仿准确率或单局
   结果。
4. 仅参考第三方实现的公开思想与接口设计；在采用代码、权重或牌谱前先核查许可、
   来源授权与规则匹配度。

最后更新：2026-08-08。
