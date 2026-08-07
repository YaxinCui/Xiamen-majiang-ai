# 棋牌 AI 开源工程地图（2026-08）

本页只记录接口、测试和训练工程的可借鉴点。未在本仓库完成许可证审计、规则映射和隐藏信息审计前，**任何外部代码、权重、
牌谱都不得复制或混入训练数据**。访问日期为 2026-08-07。

## 第一层：优先阅读的工程模式

| 工程 | 适合学习的部分 | 厦门项目的安全用法 |
| --- | --- | --- |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | extensive-form 状态、chance node、information state、算法/环境分离。 | 写一个只用于小子博弈验证的 adapter；主规则仍由现有规则核决定。 |
| [RLCard](https://github.com/datamllab/rlcard) | `env`/agent 解耦、trajectory、动作合法集合、评测种子与示例算法。 | 借鉴数据契约和 benchmark harness，不复用其麻将规则。 |
| [PettingZoo](https://pettingzoo.farama.org/) | AEC 顺序轮次 API、wrapper/测试生态。 | 只将厦门动作流程映射到 AEC；吃碰杠优先级在本引擎中裁决。 |
| [Kanachan](https://github.com/Cryolite/kanachan) | 牌谱 annotation、训练模块、annotation-vs-simulation 一致性测试。 | 复刻“同一局面双路径结果相同”的测试思想，不能使用日麻牌谱/特征。 |
| [RiichiEnv](https://github.com/smly/RiichiEnv) | Rust 高吞吐 simulation、Gym 适配、回放查看。 | 比较 Rust FFI、批量 rollout 和 replay 设计；不引入其日麻结算。 |
| [DouZero](https://github.com/kwai/DouZero) | 合法动作编码、共享内存 actor、learner checkpoint/恢复。 | 若未来并行训练，先复刻其“动作候选编码”接口，再接厦门 mask。 |
| [PerfectDou 官方代码](https://github.com/netease-games-ai-lab-guangzhou/perfectdou) | PTIE 训练骨架。 | 仅阅读训练/部署特征隔离方式，先写泄漏测试后才能做同类实验。 |

## 第二层：可作行为/部署对照的麻将项目

| 工程 | 价值 | 明确限制 |
| --- | --- | --- |
| [Mortal](https://github.com/Equim-chan/Mortal) | 强实时立直麻将 AI，展示 Rust 推理和 mjai 对接。 | 代码为 AGPL；规则、特征、模型、权重均不是厦门资产。 |
| [Akagi](https://github.com/shinkuan/Akagi) | 实时分析器、对局可视化与统计展示。 | 可启发网页复盘/调试体验；不作为训练策略来源。 |
| [MahjongRepository/mahjong](https://github.com/MahjongRepository/mahjong) | 日麻牌型、向听/和牌等通用算法的工程参考。 | 每一项数学/规则假设须单独验证，不把它当厦门判定器。 |
| [mahjax](https://github.com/nissymori/mahjax) | JAX/GPU 加速麻将 simulator 的近期探索。 | 仅跟踪吞吐架构；当前未完成其规则与许可证审计。 |

## 第三层：本项目的接入门槛

任何候选工程先提交下列最小审计记录，缺一项就只留在知识库：

```text
source_url:
commit_or_release:
license:
copied_or_reimplemented_files:
rule_differences:
actor_visible_fields:
training_only_fields:
test_vectors_compared:
benchmark_command:
```

验收顺序是：许可证 → 编译/依赖隔离 → 厦门规则单测 → 可见性单测 → 固定牌墙回放 → 四座对局评测。工程流行度、星标数和
“能跑起来”都不是接入依据。
