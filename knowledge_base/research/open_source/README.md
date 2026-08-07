# 开源工程代码区

## RLCard

仓库：[datamllab/rlcard](https://github.com/datamllab/rlcard)；论文：[RLCard: A Platform for Reinforcement Learning in Card Games](https://arxiv.org/abs/1910.04376)（A，访问：2026-08-07）。

可借鉴：环境与 agent 解耦、local view、统一 `run()` 轨迹接口、CFR/DQN/NFSP 示例、固定随机种子和多进程评测。
不能直接复制：RLCard 的 Mahjong 规则和动作抽象不是厦门规则，尤其不能覆盖本项目的金牌、游金、花牌、跟字与 16/17 张流程。

## OpenSpiel

仓库：[google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel)；官方说明其覆盖多人、零和/一般和、完美/不完全信息
和 CFR、搜索等算法（A，访问：2026-08-07）。

可借鉴：把麻将状态建模成 extensive-form game，分离 chance、信息集和 action mask；使用公共状态接口测试信息泄露；对局算法和规则引擎
分离。不能直接移植：OpenSpiel 的通用 game API 不会自动解决厦门具体结算与事件优先级。

## PettingZoo

官方文档：[PettingZoo](https://pettingzoo.farama.org/)（B，访问：2026-08-07）。AEC API 适合顺序摸打/响应流程，Parallel API 适合同时动作。
本项目是严格轮流动作并含响应优先级，因此若建立标准适配层，应优先采用 AEC 语义；引擎仍是唯一合法动作和结算权威。

## Deep CFR 参考实现

仓库：[EricSteinberger/Deep-CFR](https://github.com/EricSteinberger/Deep-CFR)（A，访问：2026-08-07）。可读其 reservoir、advantage network 和平均策略组织方式，
但仓库针对扑克信息集；接入厦门前必须先定义公开事件序列、行动 mask 和多人终局 reward。

## 代码迁移准则

开源工程只迁移接口设计、测试方式和算法骨架。任何代码、权重、牌谱和配置在进入本仓库前都要核查许可证、规则是否匹配、是否含隐藏信息，
并写入 `data_source_audit`。抚州项目单列在 [`../fuzhou_mahjong/`](../fuzhou_mahjong/)，不与通用代码区混用。
