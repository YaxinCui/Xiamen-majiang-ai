# TwoDraw 定向二元干预 v1：预注册协议

日期：2026-08-08
状态：20 墙 pilot 通过结构门槛；100 墙 selection 参数已冻结、结果未读

## 为什么不继续训练 v3/v4 相对优势模型

discard shrinkage v3 的九个确定性候选和 stochastic relative v4 的十二个随机候选已在各自独立 selection 上全部
失败。尤其 v4 的 target ESS 为 294–326，说明失败不能再归咎于支持度不足；继续调 C、beta、temperature 或 MLP
只会在已消费 selection 上过拟合。

本实验改进的是**采集分布**，不是网络容量。TwoDraw SlowExpert 曾在独立 100 墙得到 +1.335 分点估计，但区间跨 0；
uniform 干预会把样本浪费在十几种与该假设无关的弃牌上。新行为只在每局第一个普通 TwoDraw／Teacher 分歧处，以
0.5/0.5 执行两者之一，然后永久恢复 Teacher。每个选择都有高且精确的 propensity，目标是判断这个局部规则族是否
值得产生下一轮 honest gate 数据，而不是再次把 SlowExpert 当真值。

该设计遵循 contextual-bandit 实验规划中“小动作集可用均匀采样获得有竞争力的数据效率”的思路，并保留精确 propensity
供 IPS/DR 审计；参考 [Experiment Planning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1e0d9f30c100129259f66660403fb1e2-Abstract-Conference.html)、
[Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601) 和
[Optimal and Adaptive Off-policy Evaluation](https://proceedings.mlr.press/v70/wang17a.html)。这些理论只支持单点
contextual-bandit 估计，不把麻将整局自动化约成 bandit。

## 20 墙 pilot（只定规模，不选策略）

- classic `seed=202635000..202635019`，每墙四座轮换，共 80 条轨迹；行为 RNG `202635020`。
- 每条轨迹最多一次第一个普通 discard 分歧干预；游金、金牌锁、天听、响应、胡、杠和后续决策冻结。
- 安全 JSONL 重载后重新从 actor-visible 状态复现 Teacher 和 TwoDraw；不能复现、propensity 非 0.5、二元集合外动作、
  非随机后缀偏离 Teacher、同局多干预或四座不完整均失败。

结果：55.02 秒；24 次干预，candidate/Teacher 分配 12/12，20 个完整墙组，结构问题 0。按完整墙四座均分的
Horvitz–Thompson candidate−Teacher 为 +6.2 ±4.578，95% CI `[−2.773,+15.173]`。这是规模诊断，不是强度
selection。pilot report SHA-256：
`2f193e813fc9447320502cc0740dbff94b00eec8dc9c1a970eb8125c896c8b28`。

pilot 组间标准差约 20.47 分；若把 6 分／墙预先定义为值得后续建模的材料效应，正态近似约需 45 墙才有机会识别。
按项目常规评测约定并为覆盖波动留余量，正式阶段固定为 100 墙；没有根据 pilot 的单个动作或状态调规则。

## 固定 100 墙 selection

- 全新 classic `seed=202635100..202635199`，四座轮换；行为 RNG `202635300`。
- candidate assignment 固定 0.5；每轨迹最多一次；TwoDraw 参数仍为 score margin 2、最小两摸概率优势 0.05、32
  分层场景。
- 至少 100 次随机干预、100 个完整物理墙，candidate assignment fraction 必须在 `[0.4,0.6]`；所有结构审计为 0。
- 唯一晋级门槛：按物理墙四座等权的未缩减 Horvitz–Thompson candidate−Teacher 95% 下界严格大于 0。
- 不追加 400 墙。通过只允许用**另一批全新墙**设计 honest train/validation gate；selection 行不直接作部署模型标签，
  更不能证明完整多次 override、超过 Teacher 或超过人类。失败则该定向 TwoDraw 数据路线停止。
