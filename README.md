# 厦门麻将 AI

本项目提供一个本机运行的 `1 人 vs 3 个 Teacher AI` 厦门麻将网页桌。默认
为“经典厦门”档位，同时保留更轻量的“核心教学”档位；两者均不会把 AI
暗牌或牌墙顺序发送给浏览器。它是后续导出教师轨迹、监督训练、DAgger 和
hybrid 推理的规则正确性基线。

## 运行网页游戏

只需要 Python 3.11+，没有第三方依赖：

```bash
python3 scripts/serve_web_game.py --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。局域网体验可将 `--host` 改为对应
内网地址或 `0.0.0.0`；该版本没有账号与公网访问控制，不应直接暴露到互联网。

牌桌默认展开三位 AI 的调试手牌，右上角的 **隐藏 AI 手牌（调试）** 可切回
正常暗牌视图。点击 **规则与胡牌教学** 可进入 `/guide.html`，查看牌局流程、
五组一对、金牌限制、游金体系、特殊胡法、响应优先级和结算说明。

## 规则档位

网页右上角选择档位，再点“新开一局”生效。

- **经典厦门完整版（默认）**：4 人、144 张；每人 16 张，摸至 17 张按五组牌加一对将胡牌；补花后掷骰翻出真金；白板只代替金牌原牌面；支持跟打、游金 ×4、双游 ×8、三游 ×12、三金倒、抢金、天听和天胡；牌墙留 16 张荒庄。结算使用闲底 8、庄底 16 加水，三家共同付款；庄家胡或荒庄连庄并翻倍庄底。
- **核心教学**：保留 4 人、花、吃碰杠、金牌、标准和和七对；关闭白板金替身、跟打、留牌荒庄与经典底/水制，便于规则教学和测试。

不同平台仍可能采用“三游 ×16”“三金倒只限开局”等房规。本项目选用的默认值和证据边界见 [RULES_RESEARCH.md](RULES_RESEARCH.md)。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

## 规则 Teacher 模仿学习基线

经典规则已经可以导出全席位 Teacher 自博弈轨迹，并训练一个只在**规则引擎
给出的合法动作**中排序的监督策略基线。轨迹对每个决策仅保留该玩家手牌和
公开信息，不包含对手暗牌或牌墙顺序。

```bash
python3 scripts/train_rule_policy.py \
  --profile classic \
  --hands 80 \
  --validation-hands 20 \
  --epochs 10
```

脚本将把 JSONL 轨迹、训练报告和 `rule-policy.json` 检查点写入
`artifacts/rule-policy-classic/`（默认不提交）。仓库额外保留已验证的 MLP
检查点 `artifacts/rule-policy-classic-mlp/rule-policy.json`，以便直接复现当前
策略参数；原始 JSONL 轨迹仍不提交。这是可复现的模仿学习基线；
后续可在相同状态/动作接口上增加 MLP、DAgger 和对局评测，而无需绕过规则
引擎。

经典规则的游金、双游和三游属于低频状态。训练脚本默认会额外生成 136 个由
规则引擎验证、Teacher 标注的游金/双游课程样本，并用独立的游金样本报告
动作一致率。需要关闭课程样本时使用 `--tour-curriculum 0`。

训练脚本默认使用零依赖的轻量 ReLU MLP（v2：80 维状态—动作特征、12 个隐藏
单元）来给合法动作排序；其中 4 维是本家可见的候选弃牌后牌效/听牌前瞻。模型以
深层 ReLU 路径学习组合关系，并保留直接数值通路利用这些前瞻特征；规则仍决定动作
集合。已发布的 v1 76 维检查点保持可加载。需要与线性基线对照时可加
`--model linear`。

训练后的实战强度使用固定牌墙、四座轮换的 Teacher 配对评测，而不是只看动作
一致率：

```bash
python3 scripts/evaluate_policy.py \
  --checkpoint artifacts/rule-policy-classic-mlp/rule-policy.json \
  --profile classic \
  --hands 100
```

评测时模型在每个随机种子下依次担任四个座位、其余三席固定为 Teacher；输出候选
策略的平均单局净得分、标准误、胡牌率和荒庄数。训练路线与前沿调研见
[knowledge_base/](knowledge_base/README.md)。

比较两个候选时应传入 `--reference`，让评测在同一初始牌墙及四座轮换上报告每副牌墙
的配对净分差与 95% 区间；区间跨过 0 的候选不能宣称变强：

```bash
python3 scripts/evaluate_policy.py \
  --checkpoint artifacts/candidate/rule-policy.json \
  --reference artifacts/baseline/rule-policy.json \
  --profile classic --hands 100 --seed 21100000
```

若实战评测显示策略脱离 Teacher 轨迹后变弱，可用 DAgger 在策略实际访问到的
局面上补充 Teacher 标注：

```bash
python3 scripts/train_dagger_policy.py \
  --checkpoint artifacts/rule-policy-classic-mlp/rule-policy.json \
  --profile classic \
  --rounds 3 \
  --rollout-hands 20
```

该脚本把原始轨迹、报告和新检查点写入 `artifacts/dagger-classic/`；新检查点仍应
先经过换座配对评测，不能仅凭训练集指标替换网页对手。

在 DAgger 已稳定后，可用规则引擎结算的单局净得分做带状态价值基线的小步
actor-critic 微调：

```bash
python3 scripts/train_reinforce_policy.py \
  --checkpoint artifacts/dagger-classic-v2-run1/rule-policy-dagger.json \
  --profile classic \
  --iterations 4 \
  --episodes-per-iteration 40
```

每一局中只有候选策略的一席采样动作，其余三席固定为 Teacher，并在不同初始牌墙中
轮换座位。脚本会另存 `state-value-baseline.json`，可用 `--value-checkpoint` 续训；
价值基线只用于训练，不会进入网页推理。它是实验性 on-policy 阶段：必须在与训练种子
隔离的换座配对评测中胜过或至少不劣于 Teacher，才可以发布检查点。

## 轨迹 v3 与 GPU policy-value 候选

研究用的 v3 轨迹将“模型可见数据”和“可复现牌墙”分开：训练 JSONL 只含本家手牌、
公开状态、合法动作、Teacher 标签与终局分数，**不含**牌墙种子、随机行为种子、对手暗牌
或牌墙顺序。完整 replay 种子只有显式指定 `--replay-index` 才会写入用户自行保护的私有
文件，不能交给训练器或提交到仓库。

下面建立混合离线数据并训练 145 维公开候选特征的 policy-value 网络。PyTorch 只安装在
项目 `.venv`，不会改变网页和规则引擎的零依赖运行方式。

```bash
python3 scripts/collect_training_trajectories.py \
  --profile classic --hands 160 --exploration-hands 120 \
  --response-pass-curriculum 80 --tour-curriculum 136 \
  --output-dir artifacts/trajectory-balanced-classic-v3

.venv/bin/python scripts/train_policy_value.py \
  --train artifacts/trajectory-balanced-classic-v3/train.trajectories.jsonl \
  --validation artifacts/trajectory-balanced-classic-v3/validation.trajectories.jsonl \
  --test artifacts/trajectory-balanced-classic-v3/test.trajectories.jsonl \
  --device cuda --output-dir artifacts/policy-value-classic-v1
```

离线数据无法完全覆盖模型犯错后的状态，因此下一轮可用“候选一席 vs 三个冻结 Teacher”
的 DAgger 采集器。每副牌墙轮换候选四座，四个轮换被强制置于同一数据切分；候选实际访问
到的状态由 Teacher 标注，终局净分则是该候选行为分布下的 value 标签。

```bash
.venv/bin/python scripts/collect_candidate_teacher_dagger_trajectories.py \
  --checkpoint artifacts/policy-value-classic-v1/policy-value.pt \
  --profile classic --seed-count 100 \
  --output-dir artifacts/candidate-teacher-dagger-classic-v1

.venv/bin/python scripts/train_policy_value.py \
  --train artifacts/trajectory-balanced-classic-v3/train.trajectories.jsonl \
  --additional-train artifacts/candidate-teacher-dagger-classic-v1/train.trajectories.jsonl \
  --validation artifacts/trajectory-balanced-classic-v3/validation.trajectories.jsonl \
  --additional-validation artifacts/candidate-teacher-dagger-classic-v1/validation.trajectories.jsonl \
  --test artifacts/trajectory-balanced-classic-v3/test.trajectories.jsonl \
  --init-checkpoint artifacts/policy-value-classic-v1/policy-value.pt \
  --device cuda --output-dir artifacts/policy-value-classic-v1-dagger
```

训练脚本按验证集最低合法动作负对数似然选择 checkpoint（同分时取更高准确率），随后才评
一次测试集。是否晋升仍只由独立牌墙、四座轮换的配对净分置信区间决定。

### 反事实动作价值数据（Teacher 之外的策略改进信号）

Teacher / DAgger 的标签只有“Teacher 会选哪一个动作”，因此上限就是 Teacher。本项目还
提供了反事实 rollout 采集：候选实际遇到一个公开局面后，对该局面中**每个规则合法动作**
在私有内存副本中强制执行，再由冻结候选与冻结对手走到结算。导出的 JSONL 只保留本家手牌、
公开状态、合法动作及其对应终局净分 `action_values`；绝不导出用于模拟的牌墙或对手暗牌。

同一条 rollout replicate 中，所有候选动作共享同一组冻结对手，降低动作比较噪声。这个目标
估计的是当前候选与冻结对手混合下的 `Q^π(s,a)`，不是知道暗牌的部署特征，也不是自动可用的
“最优解”；必须重新采集、独立评测，且不能把一次 rollout 的小样本 checkpoint 提升到网页。
建议打开 `--belief-resample`：它会固定本家手牌和全部公开信息，而对未知牌墙、他家暗手、
非本家花牌和对手暗杠牌面重新采样；这避免把某一副真实暗牌分配直接当成玩家可知的价值。
它目前是公开信息先验采样，并不对历史对手动作作完整后验加权，实验报告必须标记该限制。
对于 `core`，或已排除游金/天听/早局特殊状态的 `classic` 档，且本家正响应“对手公开摸牌后立即弃牌”的局面，可额外启用
`--belief-latest-discard-particles 32`：它在当前公开先验粒子上重建该弃牌前状态，按冻结对手
策略选择该弃牌的**保守温和**似然重采样。默认使用似然幂次 `0.25` 与预重采样 ESS 比例门槛
`0.5`；不满足公开前缀、候选合法动作或 ESS 门槛的局面会被跳过。它只是单事件局部条件化，
**不是完整历史 posterior**，默认关闭，不能据此直接晋升网页模型。
采集摘要还会分别报告结构一致粒子率与 ESS：例如经典档的公开“强制跟打”规则会使一部分
重分配后的对手手牌与观察弃牌不相容，这些粒子必须拒绝，不能从源局借用暗手使其通过。
`--rollout-batch-size` 只将独立分支的神经网络推理合并；每条分支仍有独立规则状态与随机数，
并逐步执行原有合法性检查。开发或回归验证时可固定为 `1`，再与批量输出逐项比较。

```bash
.venv/bin/python scripts/collect_action_value_trajectories.py \
  --checkpoint artifacts/policy-value-classic-v1-dagger/policy-value.pt \
  --profile classic --seed-count 200 --samples-per-hand 1 \
  --rollouts-per-action 4 --rollout-batch-size 32 --belief-resample \
  --opponent-checkpoint artifacts/policy-value-classic-v1/policy-value.pt \
  --teacher-opponent-probability 0.75 --device cuda \
  --output-dir artifacts/counterfactual-action-value-classic-v1

.venv/bin/python scripts/train_policy_value.py \
  --train artifacts/trajectory-balanced-classic-v3/train.trajectories.jsonl \
  --additional-train artifacts/counterfactual-action-value-classic-v1/train.trajectories.jsonl \
  --validation artifacts/trajectory-balanced-classic-v3/validation.trajectories.jsonl \
  --additional-validation artifacts/counterfactual-action-value-classic-v1/validation.trajectories.jsonl \
  --test artifacts/trajectory-balanced-classic-v3/test.trajectories.jsonl \
  --additional-test artifacts/counterfactual-action-value-classic-v1/test.trajectories.jsonl \
  --init-checkpoint artifacts/policy-value-classic-v1-dagger/policy-value.pt \
  --action-value-weight 1.0 --action-value-regression-weight 1.0 \
  --action-value-target-scale 80 --action-value-temperature 16 --action-value-margin-scale 16 \
  --checkpoint-selection-source counterfactual_action_value_rollout \
  --checkpoint-selection-metric action_value_huber_loss \
  --minimum-selection-decisions 32 --device cuda \
  --output-dir artifacts/policy-value-action-value-classic-v1
```

局部条件化必须单独收集、单独报告。经典档的冻结 Teacher 行为更尖锐，使用更保守的幂次与门槛：

```bash
.venv/bin/python scripts/collect_action_value_trajectories.py \
  --checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --profile classic --seed-count 200 --rollouts-per-action 4 --belief-resample \
  --belief-latest-discard-particles 32 \
  --belief-latest-discard-likelihood-power 0.10 \
  --belief-latest-discard-min-ess-fraction 0.20 \
  --output-dir artifacts/counterfactual-action-value-classic-local-belief-v1
```

该训练同时做两件事：将 `action_values / temperature` softmax 为软动作偏好，并让独立 Q 头
直接回归每个合法动作的归一化终局净分（`action_values / --action-value-target-scale`）。前者
学习相对偏好，后者保留“好多少”的幅度；二者不能互相替代。Q 头目前只在 `candidate_mlp`
结构中可用，旧 v1/v2 checkpoint 加载时会零初始化该头，原策略输出保持不变。

无论模拟结果多大，所有动作同分的样本没有排序信号，默认以
`--action-value-margin-scale 16` 将其权重压到零；若采集时有多次 rollout，还可用
`--action-value-stderr-scale` 下调高方差目标。若不同合法动作的标准误差异很大，
`--action-value-confidence-z 1.0` 可将软偏好目标改为 `Q - z × stderr` 的逐动作下置信界；默认 `0`
不改变原始均值目标。对于每次在**同一 belief world**内评估的合法动作，采集器还会导出动作差值标准误；
`--action-value-pairwise-confidence-z 1.0` 只保留大于该配对误差的动作价值间隔，通常比绝对 Q 标准误
更适合判断吃/碰/过的排序可靠性。`--checkpoint-selection-source` 防止大量 Teacher
样本掩盖在线动作价值留出集；当训练 Q 头时，`--checkpoint-selection-metric action_value_huber_loss`
应和该来源配套，避免按另一个策略目标选择 epoch。最终是否保留
checkpoint，仍由全新牌墙上的四座轮换配对评测决定。

训练报告会额外输出 `action_value_huber_loss`、回报绝对误差和动作价值 argmax 准确率。默认仍
按 policy 头选牌；只有 Q 的隔离实战评测胜出，才可显式使用
`scripts/evaluate_policy.py --action-selection action_value` 测试 Q 头，绝不可凭离线 Q 指标替换网页 AI。

若要探索超越 Teacher 的方向，可对 policy-value 候选做终局净分 PPO 微调。它只允许一席
候选策略采样，三席始终冻结为 Teacher；策略仍只能从规则引擎的合法动作中选择。该命令
输出的是研究 checkpoint，不会自动替换网页 AI：

```bash
.venv/bin/python scripts/train_torch_ppo.py \
  --checkpoint artifacts/policy-value-classic-v1-dagger/policy-value.pt \
  --profile classic --iterations 3 --episodes-per-iteration 1024 \
  --rollout-batch-size 32 \
  --output-dir artifacts/torch-ppo-classic

.venv/bin/python scripts/evaluate_policy.py \
  --checkpoint artifacts/torch-ppo-classic/policy-value-ppo-iteration-3.pt \
  --reference artifacts/policy-value-classic-v1-dagger/policy-value.pt \
  --profile classic --hands 200 --seed 22000000 --device cuda
```

如需降低训练期终局净分的高方差，可加 `--privileged-critic`。它的 critic 仅在该次 PPO 进程内读取
完整模拟状态作为 advantage baseline；`TeacherDecision`、网页 actor、JSONL、报告和保存的
`policy-value-ppo-iteration-*.pt` 都不会携带其特征或权重。该路径默认关闭，且仍必须通过未见牌墙评测：

```bash
.venv/bin/python scripts/train_torch_ppo.py \
  --checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --profile classic --iterations 3 --episodes-per-iteration 1024 \
  --rollout-batch-size 32 --privileged-critic \
  --privileged-critic-hidden-size 128 --privileged-critic-weight 0.25 \
  --device cuda --output-dir artifacts/torch-ppo-classic-privileged-critic-v1
```

针对 `.pt` checkpoint，采集和评测命令会记录推理设备；同一轮比较必须固定同一设备，避免
CPU/GPU 浮点舍入在临界动作处造成不必要的策略差异。

若要训练对多种历史策略都稳健的候选，可重复传入 `--opponent-checkpoint`，并通过
`--teacher-opponent-probability 0.75` 保留 75% Teacher 对手。批量 rollout 会把候选及
同一冻结 checkpoint 的对手请求分别合并成变长合法动作 batch；规则引擎和结算不会被并行
模型绕过。

除单元测试外，改动网页交互后应启动本地服务并通过真实浏览器完成至少一局。
