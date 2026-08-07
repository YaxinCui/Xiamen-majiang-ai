# 厦门麻将 AI 研究知识库

这是项目研究资料的独立分区，和 `knowledge_base/experiment_log.md` 的“已运行实验记录”分开。
这里记录论文、开源工程、强化学习技术、对本项目的思考，以及抚州/南城麻将项目的迁移审计。

## 分区

- [`papers/`](papers/)：论文与方法卡片。优先记录原论文、官方版本、适用问题和不能直接移植的假设。
- [`open_source/`](open_source/)：开源工程与代码架构。只记录可复用的接口、测试和工程模式，不复制未确认许可的代码、权重或牌谱。
- [`reinforcement_learning/`](reinforcement_learning/)：本项目的强化学习技术路线、数据契约、信息集约束和评测门槛。
- [`reflections/`](reflections/)：基于本项目运行结果形成的判断、反例和下一步假设，明确区分事实与推断。
- [`fuzhou_mahjong/`](fuzhou_mahjong/)：`YaxinCui/fuzhou-mahjong-ai` 的专项审计，记录它为什么有效、哪些方法适合迁移、哪些规则不能迁移。
- [`xiamen_rules_engine/`](xiamen_rules_engine/)：厦门麻将规则、状态机、动作合法性和结算事实源。
- [`data_and_evaluation/`](data_and_evaluation/)：数据来源、切分、指标、物理墙种子和统计解释。
- [`experiments_and_reproduction/`](experiments_and_reproduction/)：预注册协议、命令、版本、失败实验和复现清单。

## 研究记录规范

每条新增知识尽量包含：来源、访问日期、问题设定、结论、适用边界、对厦门麻将的具体动作。来源等级分为：

- **A：** 原论文、官方仓库、项目自身技术报告或可复核实验产物。
- **B：** 官方文档、维护者说明、同行评议后的综述。
- **C：** 基于 A/B 和本项目实验的推断；必须标记为推断，不能写成论文事实。

## 评测约定

从现在起，普通候选筛选使用 `classic` 档连续 **100 副独立物理牌墙**，每副墙轮换四个座位，默认对手为三名
冻结 `HeuristicTeacherAgent`。CLI 的 `scripts/evaluate_policy.py` 默认值已经是 `--hands 100`。

100 墙用于日常迭代和快速筛选；只有候选在 100 墙上显示出稳定正向信号，才考虑额外的 400 墙确认。历史实验中已
使用的 160/400 墙范围不得改写为 100，避免事后改变统计协议。

## 当前总判断（2026-08-07）

抚州项目的公开技术报告显示，其生产 agent 是“神经策略 + lookahead + Teacher + 公开风险/弃牌进展”的 hybrid，
不是直接从规则 Teacher 跳到纯神经网络。对厦门麻将，最可行的中间态仍是：规则引擎正确 → 公开信息 Teacher/搜索
产生高质量且有覆盖的数据 → Teacher-anchored residual 或混合策略 → 独立四座评测 → 再扩大训练。

本知识库不把抚州的模型参数、规则代码或训练牌谱当作厦门项目的可直接输入；规则、版权和隐藏信息边界必须单独审计。

最后更新：2026-08-07。
