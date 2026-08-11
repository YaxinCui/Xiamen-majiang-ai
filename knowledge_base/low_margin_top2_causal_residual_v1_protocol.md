# 低分差 top-2 随机干预与因果 residual v1

状态：**正式训练完成，但 2,000 墙 validation 的四个预注册覆盖门全部拒绝；terminal 未采集，模型不部署。** 本协议不是对模型强度的声明。

## 问题

冻结 `HeuristicTeacherAgent` 的普通弃牌分数包含大量并列或低分差状态。直接把 top-2 当作更优标签没有依据；在同一隐藏
世界里分别 rollout 又会把真实牌墙当成 actor 信息。这里改用真实对局中的高支持随机试验：每条候选座轨迹最多改变一个
动作，记录精确 propensity，并用终局分差估计 top-2 相对 top-1 的条件因果效果。

## Pilot（结果前冻结）

- 规则：`classic` / `xiamen-classic-full-v2`。
- 物理墙：25，`seed=202649000..202649024`，每墙四座轮换，共 100 局。
- 行为随机种子：`202649025`。
- 资格：候选座第一次非游金、非锁金、非天听且 Teacher 行动为普通弃牌的状态；冻结 Teacher 的前两项分差必须 `<=2.0`。
- 处理：以 0.5 选择 Teacher top-2，以 0.5 保留 top-1；随后所有动作恢复 frozen Teacher。
- 机械门槛：至少 80 次干预、25 个完整四座墙组、top-2 分配比例 `[0.3,0.7]`、终局分数非退化、所有导出状态能仅凭
  actor-visible 信息复现 top-1/top-2 和行为支持。正式 chunk 允许极少数整局没有合格机会，但必须保存显式
  `no_intervention` 零贡献行，且总体干预覆盖至少 95%。
- pilot 只决定正式样本量和工程是否可行。**不得用其 HT 均值、置信区间或任一分层方向选择规则、阈值或 checkpoint。**

## 数据契约

训练可见字段只允许行动者手牌、公开牌河／副露／花／金、公开历史摘要、合法动作、Teacher top-1/top-2、实际随机动作、
propensity、候选座终局分差和 opaque 物理墙 group。不得包含 seed、真实牌墙、其他玩家暗手、未来动作、账号或逐局可逆来源。
四座轮换必须作为同一个 group 切分；任何不完整墙组、非 0.5 随机行、多个干预或无法复现 top-2 的记录整组拒绝。

安全瘦身记录版本为 `xiamen-low-margin-top2-causal-v1`。有干预的对局只保留随机决策的 actor-visible state、两项动作、
处理分配、0.5 propensity、候选座终局分差及 opaque group；没有合格机会的对局保存 `state/actions/margin=null`、
`executed_arm=none`、propensity 1.0 的零贡献占位行。这样四座墙组不会因“无机会”被条件性删除。完整 public suffix、其他
普通 Teacher 决策和四家终局向量均不进入模型数据文件。每个 100 墙 chunk 单独写入并绑定 SHA-256，拒绝覆盖，便于中断后
从下一个完整 chunk 恢复。chunk 写出前按不可逆的 opaque UUID `record_id` 排序，不保留墙／座位采集顺序，避免通过行号和
公开 seed 范围间接还原隐藏世界。

## Pilot 结果（不作强度选择）

- 100/100 局均出现合格干预；alternative/Teacher 分配为 57/43，25 个四座墙组完整，审计问题 0。
- 93 次为完全并列，7 次分差 `(1,2]`；说明正式路线主要学习“完全并列时何种公开上下文应选第二牌”，不是继续手调
  一个固定 tie-break 规则。
- 候选座终局分差范围 `[-44,108]`，墙级 HT 标准差 **38.1226**。pilot 均值 `−1.1` 及其区间按预注册不用于选择、
  停止或改阈值。
- 报告 SHA-256 `753d06574b7ed0c80f65bb5c0ee7f691bd1646925dde7b143647a1f43051fddd`；安全全轨迹仅供
  pilot 复核，SHA-256 `dc9a4169637d33363b4e90458d815dc15624c358bf8ddeca9fcdbce2c8c18345`。

## 正式数据规模（pilot 后、正式数据前冻结）

pilot 标准差意味着 100–500 墙无法可靠判断 1–2 分级别的小优势。因此固定：

| 分区 | 物理墙 | seed | 100 墙 chunks | 用途 |
| --- | ---: | --- | ---: | --- |
| train | 4,000 | `202650000..202653999` | 40 | 拟合三随机种子 CPU 小模型及内部训练诊断 |
| validation | 2,000 | `202654000..202655999` | 20 | 冻结 ensemble 和覆盖门；不参与梯度 |
| terminal | 2,000 | `202656000..202657999` | 20 | validation 严格过门后才允许采集／读取 |

每个 split 的干预覆盖必须至少 95%，alternative 在实际干预中的总分配比例须在 `[0.45,0.55]`，记录数必须严格等于墙数乘四。
train 与 validation 的随机
行为分别使用独立 chunk seed 域 `202670000+chunk`、`202671000+chunk`，这些 seed 不写入安全记录。pilot 的 25 墙不
并入任一正式分区。

## 冻结的小模型

首选不是 Transformer，而是 CPU 小型因果 residual：输入为 feature-v4 的 top-2、top-1 及其差分；以随机处理中心化后的
outcome regression 学习 conditional effect，不把单局输赢硬转成“正确动作”。固定训练三个随机种子并平均 effect；网络宽度、
epoch、正则、训练内 early-stop 和 validation 覆盖档必须在打开 validation 前写入本协议。

validation 只允许在预注册覆盖档中选择门槛，并按物理墙聚合 inverse-propensity policy-minus-Teacher 贡献。需要足够的实际
override、随机支持平衡以及多重比较校正后的严格正下界；未通过时 terminal 不创建。通过后冻结 checkpoint、ensemble、阈值和
输入 SHA，再采集 terminal；terminal 单一检验 95% 下界仍须严格大于零。

### 模型与优化器冻结值（打开 validation 前）

- feature-v4 每动作 204 维；输入严格为 `top1 || top2 || (top2−top1)`，共 612 维。
- 三个独立成员，seed `202672000/1/2`；每个为共享 `612→64→64` GELU encoder，加独立 `64→32→1`
  nuisance/effect heads，约 4.8 万参数。没有 attention、RNN 或 Transformer。
- treatment 中心化编码：alternative `+0.5`、Teacher `−0.5`；预测事实结果为 `m(x)+(T−0.5)τ(x)`，终局分差除以
  40。loss 为 beta 0.5 的 Huber，AdamW，lr `1e-3`、weight decay `1e-3`、batch 256、梯度裁剪 5，最多 40 epoch。
- train 内按 opaque wall group 固定 90/10 切分，以 factual Huber 最小选择各成员 epoch；随后相同 seed 在全部 train
  group 上只重训所选 epoch。外部 validation 不参与 epoch 或权重拟合。
- 推理时只平均三个成员的 `τ(x)`。`executed_arm`、propensity、终局分差、seed、墙和他家暗手均不属于网络输入。
- 训练硬件固定为当前 RTX 2080 Ti CUDA：actor-visible 特征只在 CPU 计算一次，绑定全部 train 输入 SHA 后缓存为约 40 MB
  tensor；features/treatment/outcome 和内部分组 mask 整体驻留 GPU，三成员复用同一缓存。batch、网络和统计门槛不因硬件迁移
  改变。显存占用不是目标；禁止为了填满 22 GB 显存扩大模型或并行试探更多候选。
- 每个 chunk 的 actor-visible top-2 已在采集落盘后完整重读复算，并由 split 总审计绑定 SHA。训练器和 selector 再次核对
  chunk SHA、opaque 顺序契约与总审计后只做严格 schema 解析，不做第三遍完全相同的手牌复算；数据字节变化会使 SHA 先失败。

### validation 门槛冻结值

ensemble 在 train eligible 行上的严格 effect 分位数冻结 1%／2%／5%／10% 四个覆盖门，validation 不重新定阈值。
每门使用 0.5 propensity 的 policy-minus-Teacher HT 贡献，并把无机会或未覆盖行记为零；四座先平均成物理墙样本。

每个可晋级门必须同时满足：至少 60 次 override、至少 60 个 override 墙组、其中日志 alternative/Teacher 分配各至少
20 次；四个门采用双侧 family α=0.05 Bonferroni，临界值固定 `z=2.4977054744`，校正后下界严格大于零。若多门通过，
选择校正下界最高者（再以更低覆盖、阈值更高作确定性破同分）。全部失败则状态固定为 validation rejected，terminal 不创建。

即使单次干预的 terminal 通过，它也只解锁一个“每局第一次低分差机会最多覆盖一次”的 100 墙部署试验。重复多次覆盖、网页
替换、400 墙确认和真人强度评测都需要各自的新门槛，不能从离线准确率外推。

## 正式训练与 validation 结果

train 的 40 个 chunk 全部通过审计：16,000 条记录、15,901 条实际干预、4,000 个完整物理墙组。三成员分别在 train 内
选择 epoch 5、6、22；其全 train 事实结果 RMSE 为 42.80、42.76、42.11 分。三个成员的平均 effect 并不稳定，其中一个
成员的全 train effect 均值为负；这在打开 validation 前已经提示条件效应信号弱于终局噪声，但没有改变任何预注册门槛。

validation 的 20 个 chunk 同样全部通过数据审计：8,000 条记录、7,945 条实际干预、2,000 个墙组，和 train 的 opaque
group 重合为 0。所有门的覆盖、override 数和 0.5/0.5 日志支持均达到机械要求，但 Bonferroni 下界均未严格大于零：

| train 目标覆盖 | validation overrides / groups | alt / Teacher | 墙级均值 | Bonferroni 区间 | 结论 |
| ---: | ---: | ---: | ---: | --- | --- |
| 1% | 87 / 87 | 42 / 45 | -0.093 | [-0.6743, 0.4883] | 拒绝 |
| 2% | 144 / 144 | 73 / 71 | +0.1005 | [-0.6192, 0.8202] | 拒绝 |
| 5% | 389 / 380 | 213 / 176 | +0.1000 | [-0.9351, 1.1351] | 拒绝 |
| 10% | 760 / 713 | 398 / 362 | +0.38925 | [-0.9393, 1.7178] | 拒绝 |

因此状态固定为 `validation_rejected_terminal_not_collected`。不得在同一 validation 上改网络、挑成员、改覆盖档或降低统计
门槛；预留的 2,000 墙 terminal 没有创建或读取，也不运行 100 墙部署试验。训练报告 SHA-256 为
`b5adced5412d7d83d50297991b53d8618e212c50bbeb10b6871babe08038e916`，validation 报告 SHA-256 为
`b6873839660d2587e20d4ceab3813e39a39e1a1f508ebd79de88b0c13ad3b715`。

## GPU 利用结论

最初训练慢的主要原因不是显存不足，而是每个 epoch、每个成员都在 CPU 重复执行 actor-visible 手牌特征计算，并进行很多
小批量 CPU→GPU 搬运。正式训练前已经改成一次 CPU 特征构造并写入绑定全部输入 SHA 的 tensor cache；39,074,321-byte
cache 的 SHA-256 为 `b6cff35d0fbdf4025d725517fda0b69ab8c494decd6bac8eeed628e0dcc0d2b7`。features、treatment、
outcome 和 group mask 一次放到 GPU，三个成员复用；validation 也只构造一次特征并让三个 checkpoint 复用 GPU tensor。

RTX 2080 Ti 有 22 GB 显存，但 4.8 万参数、16,000 样本的网络只需约 316 MB 工作显存是正常结果。监控中训练从原先 CPU
单核、GPU 近空闲变为 P2、约 34% 计算利用率。这里的目标是减少墙钟时间与数据搬运，不是提高显存占用率。禁止仅为“吃满
显存”改成 Transformer、扩大网络或并行试探更多候选；这些动作会增加过拟合和 multiple-testing 风险，却不会改善随机试验
标签的信噪比。

后续若要继续提高硬件效率，顺序应为：先让独立牌局采集使用多进程 CPU 并行；再把任何新小模型的固定候选／随机种子作为
一个预注册 GPU batch 训练与推理；只有 profiler 证明规则/hand kernel 成为主要且可批处理的热点后，才考虑将麻将状态转移和
牌效计算改写为向量化 Torch/JAX kernel。当前 Python 游戏引擎以分支、对象和集合运算为主，直接把它搬到 CUDA 不会自然加速。
