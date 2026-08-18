# 低分差 top-2 同墙控制变量因果 residual v2

状态：**全新 2,000 墙 validation 已完成但四门均拒绝；v2 terminal 未采集，整个低分差 top-2 residual 家族冻结。** v1 validation 已经打开且拒绝，
因此禁止将其用于 v2 的模型、门槛、控制系数或停止决策。

## 为什么另开 v2

v1 的四个覆盖门都具备足够的随机支持，但 2,000 墙上的终局方差使 Bonferroni 下界跨零。继续扩大网络不能降低随机终局
本身的噪声。v1 train-only 诊断显示，同一物理墙的四个座位轮换之间存在可利用的共同／负相关难度分量。对当前轨迹 `i`，
定义另外三条轮换轨迹终局候选分数的均值：

`B_gi = mean(Y_gj, j != i)`。

当前轨迹的 0.5/0.5 随机动作 `A_gi` 不参与另外三局，故在固定墙和其他三局后，`B_gi` 与 `A_gi` 独立。对任意在打开
validation 前冻结的系数 `c`：

`E[(I(A=alt)/0.5 - I(A=teacher)/0.5) * (Y - cB)] = E[Y(alt)-Y(teacher)]`。

因此 `B` 只作为训练 target 和随机试验估计器的 control variate；它不是网页部署输入，也不改变待估计的 top-2 相对
Teacher 因果效果。

## train-only 诊断（不作强度结论）

在 v1 train 的 4,000 个墙组上，按 group hash 做五折、每折只用另外四折估计最优 `c`，墙级标准差相对原 HT 的比值为：

| v1 train 覆盖 | 原墙级 SD | 留一控制后 SD | 比值 |
| ---: | ---: | ---: | ---: |
| 全部干预 | 42.8293 | 37.1569 | 0.8676 |
| 1% | 10.2151 | 4.8349 | 0.4733 |
| 2% | 13.3840 | 7.2570 | 0.5422 |
| 5% | 18.7033 | 11.6361 | 0.6221 |
| 10% | 23.1802 | 16.3629 | 0.7059 |

这些数只说明工程路线有方差收益，不允许当作候选强度、覆盖选择或 v2 validation 结果。

## 冻结数据边界

- 训练来源：只读复用 v1 train 的 40 个 chunk，4,000 墙／16,000 条安全记录；每个 SHA 必须和 v1 train 总审计一致。
- 禁止读取：v1 validation、v1 预留 terminal，以及任何 v2 validation／terminal，直到相应阶段被协议解锁。
- v2 validation：全新 classic 2,000 墙，`seed=202658000..202659999`，20 个 100 墙 chunk；行为随机 seed 域
  `202673000+chunk`。
- v2 terminal：全新 classic 2,000 墙，`seed=202660000..202661999`；只有 v2 validation 通过后才允许采集。
- 数据格式、0.5 propensity、每候选座轨迹最多一次干预、95% 干预覆盖、四座完整 group、opaque UUID 排序和 actor-visible
  top-2 复算规则均沿用 v1。v2 validation 不得软链接或复制 v1 validation。

## 冻结模型

- 输入仍为 feature-v4 的 `Teacher || alternative || difference`，612 维；不输入 group、其他三局结果、当前随机 arm、终局分数、
  seed、真实墙或他家暗手。
- 训练 target 对每条 eligible 记录固定为 `(Y_i + B_gi)/40`，等价于 control coefficient `c_train=-1.0`。选择 -1 而不是
  train 最优小数，是为了避免把有限 train 的系数噪声写进监督标签。
- 三个 fresh seed `202674000/1/2`，网络仍为 `612→64→64` GELU encoder 加 `64→32→1` nuisance/effect heads；Huber
  beta 0.5、AdamW lr 1e-3、weight decay 1e-3、batch 256、clip 5、最多 40 epoch。
- 按 opaque group 固定 90/10 train 内 epoch selection；同一墙的四条记录不得跨 fit/holdout。选定 epoch 后在全 train 重训。
- gate score 改为三个成员 effect 的**最小值**，而不是均值；它要求三随机种子都认为 alternative 有优势，以处理 v1 的 seed
  方向不稳定。仍只在 train score 上冻结严格 1%／2%／5%／10% 分位数。
- train 特征一次构造并缓存，训练 tensor 常驻 CUDA；不得为了显存占用扩大模型或候选族。

## 冻结 validation 估计器

对每个冻结 gate，在 train 墙级数据上计算：原 HT 墙贡献 `Z_g`，以及把每条当前结果换成其 `B_gi` 后的零均值墙控制量
`Q_g`。冻结

`c_gate = clip(cov(Z,Q)/var(Q), -4, +4)`。

该系数、checkpoint、gate threshold、全部 train 输入 SHA 和训练报告 SHA 必须在采集 v2 validation 前写入模型报告。
validation 的每墙贡献为 `Z_g - c_gate Q_g`。validation 不能重新估计 `c`。

四个门仍采用双侧 family alpha=0.05 Bonferroni，固定 `z=2.4977054744`。每个门至少 60 次 override、60 个 override
墙组，日志 alternative/Teacher 各至少 20；校正后下界必须严格大于零。多个门通过时选校正下界最高者，再以更低覆盖、
更高阈值破同分。全部失败则 terminal 不创建。

通过只证明“每局第一次低分差普通弃牌最多替换一次”的 simulator policy 值改善；它不直接证明网页、多次 override 或真人强度。

## train-only 冻结结果（v2 validation 采集前）

CUDA 三成员的 train 内选定 epoch 为 7／5／8；全 train wall-controlled factual RMSE 分别为 38.82、39.17、37.82 分。
成员 effect 均值分别为 -1.057、-1.267、-0.251，说明总体 top-2 没有普遍优势，故部署候选严格使用三成员最小 effect 的
最上层覆盖，而不是平均 effect 或全覆盖。

| 目标覆盖 | 严格 minimum-effect 阈值 | train overrides | 冻结 `c_gate` | 原墙级 SD | 调整后 SD | SD 比值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 3.539841 | 159 | -2.835063 | 10.1913 | 4.2984 | 0.4218 |
| 2% | 2.701957 | 318 | -2.609671 | 13.1579 | 7.1935 | 0.5467 |
| 5% | 1.127638 | 795 | -2.399841 | 18.5065 | 11.7350 | 0.6341 |
| 10% | 0.235572 | 1,590 | -2.139708 | 22.8892 | 16.4017 | 0.7166 |

模型报告 SHA-256 为 `24f80ca8efb55cfdbf916f314e21a5899e53caceabe6e6a66cfd527e7ee3d840`；三个 checkpoint
SHA-256 依次为 `6e179b6c9b6152015b309412c708bdca6d6783f7c6c7cbcc03afb48a3102100f`、
`a278fe798ae05cf65755de6161e979d72a028a7278ad02bdeac9b86be22ae30c`、
`f303cd4648612caf3136688a619a5d4cc104000cedac83de4f2ead608b8d56e7`。训练报告明确记录 v1 validation／terminal
均未被 v2 trainer 读取或 hash，v2 validation／terminal 均未采集。

## v2 validation 结果

全新 `202658000..202659999` 的 2,000 墙得到 8,000 条记录、7,946 次干预；alternative/Teacher 为
3,960/3,986，54 条无合格机会，完整墙组 2,000，train group 重合 0，审计问题 0。四个 gate 的机械支持均通过：

| 覆盖 | overrides / groups | alt / Teacher | 原 HT SE | control 后均值 / SE | Bonferroni 区间 | 结论 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1% | 73 / 73 | 41 / 32 | 0.2039 | +0.1022 / 0.0950 | [-0.1351, 0.3396] | 拒绝 |
| 2% | 141 / 141 | 73 / 68 | 0.2704 | -0.0746 / 0.1422 | [-0.4297, 0.2805] | 拒绝 |
| 5% | 401 / 393 | 194 / 207 | 0.3956 | +0.1193 / 0.2437 | [-0.4893, 0.7279] | 拒绝 |
| 10% | 770 / 713 | 387 / 383 | 0.4894 | +0.5806 / 0.3511 | [-0.2962, 1.4574] | 拒绝 |

控制变量在独立 validation 上把 SE 降至原来的约 47%／53%／62%／72%，证明降噪工程泛化；但降噪后的效果仍没有严格
正下界，说明失败主体是候选动作信号弱，而不是单纯估计器方差。状态固定为
`wall_control_v2_validation_rejected_terminal_not_collected`：`202660000..202661999` terminal 不得采集，不部署，
不追加墙，也不在这批 validation 上改成员聚合、覆盖、系数或显著性门槛。

validation split 审计 SHA-256 为 `2773ccedfc00f7629b5daddb39320ddc8e9faa89e6814f5b361ea0e7e05e3b7a`；
selection 报告 SHA-256 为 `a593d4e3159872bf215747f5e5dd2bed894e31c90d0daf43bbf7e949ea65c79c`。
v1/v2 共同表明“第一次低分差普通弃牌时学习 top-2 替换”不足以形成可靠提升，本族到此冻结。
