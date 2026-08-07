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

### 可选：本地人类对局数据

当前模型还没有“击败人类”的验证证据，因此可在自己试玩时显式开启本地记录，为后续独立的人类行为
评审提供数据：

```bash
python3 scripts/serve_web_game.py \
  --human-log local_human_data/my-training-play.jsonl \
  --human-log-purpose training
```

仅在一局结束后记录：玩家本人可见的手牌与公开状态、当时全部合法动作、实际选择和最终公开结算。
不保存牌墙顺序、三家暗手、随机种子、账号或网络标识；`local_human_data/` 默认被 Git 忽略，也不会自动
混入 Teacher/DAgger 训练。人类数据必须先按牌局切分、检查规则档位和质量，并在独立人类留出局上验证，才可
作为新的训练来源。

`--human-log-purpose` 是强制的来源契约，不能事后靠文件名推断：`training` 只表示可在人工复核后申请用于
行为模仿；`evaluation` 表示独立强度评测，训练器和切分器会直接拒绝它。两种用途必须写入不同的本地 JSONL，
不能混合。

若要让人类玩家与某个实验 checkpoint 对局，必须显式指定它；没有该参数时仍是 Teacher。网页会显示
“EXPLICIT EXPERIMENTAL CHECKPOINT”，避免把未通过离线门槛的模型误认为默认版本：

```bash
.venv/bin/python scripts/serve_web_game.py \
  --ai-checkpoint artifacts/policy-value-classic-v1-run3/policy-value.pt \
  --ai-device cpu \
  --human-log local_human_data/run3-vs-human-evaluation.jsonl \
  --human-log-purpose evaluation
```

这只是受控试玩和数据采集路径，不构成该 checkpoint 胜过人类的证据。run3 仅是历史实验 checkpoint：其后续
独立 400 墙终检也没有通过 Teacher 的正向置信区间门槛，网页默认继续使用 Teacher。

记录达到一定数量后，先运行只读质量审计，而不是直接训练：

```bash
python3 scripts/audit_human_trajectories.py \
  --input local_human_data/run4-vs-human.jsonl \
  --minimum-hands 100
```

审计会拒绝混合规则档位/对手版本、重复牌局、带重放种子或隐私字段的记录，并对结构合格牌局汇总人类本局
平均分差、标准误、正态近似 95% 区间、胡率和流局率。即使结构审计通过，也只表示可以进行人工质量评审；
这些描述性统计不等同于人类强度结论。要声称对人类变强，仍须使用从未用于训练或调参的真人对局留出集。

对于已冻结、已知 SHA-256 身份的候选 AI，只能用 `evaluation` 记录器收集**从未进入训练或选模**的真人局，
再运行独立只读审计：

```bash
python3 scripts/audit_human_match_strength.py \
  --input local_human_data/frozen-ai-human-evaluation.jsonl \
  --minimum-hands 200 --minimum-sessions 10
```

每次启动记录器会生成一个不含姓名、账号或设备信息的随机会话 ID；报告按会话均分而非逐局计数，避免一名玩家
连续对局被误作许多独立人类样本。下界为正也只解锁人工复核：仍需验证每个会话对应预先声明的独立参与者或区块、
参与者同意、招募范围和水平、AI 身份在全程冻结，以及这些局从未用于训练、early stop 或选模，才可能作为
“胜过该真人评测群体”的证据；它不自动证明胜过一般人类。

审计通过后，用下列命令把**完整牌局**稳定拆成互不重叠的 train / validation / test。输出被限制在
Git 忽略的 `local_human_data/` 下，已存在的输出默认拒绝覆盖：

```bash
python3 scripts/split_human_trajectories.py \
  --input local_human_data/run4-vs-human.jsonl \
  --minimum-hands 100 \
  --output-dir local_human_data/run4-split-v1
```

切分只准备结构合格的人类**行为模仿**输入；它既不代表记录者一定是强人类，也不授权训练或模型晋级。

即使把人类 JSONL 直接传给训练器，它也会默认拒绝。人工确认记录者、对手身份、规则档位和独立留出集后，才可
在按完整牌局分开的 train/validation/test 文件上显式启用；训练报告只记录聚合审计与
`<local_human_data>` 占位符，不会写入你的本地路径：

```bash
.venv/bin/python scripts/train_policy_value.py \
  --train local_human_data/train.trajectories.jsonl \
  --validation local_human_data/validation.trajectories.jsonl \
  --test local_human_data/test.trajectories.jsonl \
  --allow-local-human-data --human-minimum-hands 100 --human-weight 1.0 \
  --output-dir artifacts/human-reviewed-policy-experiment
```

此开关只授权行为模仿，**不**证明这些记录代表强人类，且不会让 checkpoint 自动成为网页默认或“胜过人类”的证据。

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

`--action-value-weight` 只控制反事实标签是否改变 policy 偏好；`--action-value-regression-weight` 与
`--action-value-regression-sample-weight` 则独立控制 Q 回归。因此可以将前者设为 `0`，先在冻结的
按牌墙切分数据上校准和验证**公开信息** Q 头，而不会把尚未证实的 Q 排序写入 policy。

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

### 已执行动作的后状态校准（诊断，不选牌）

`TeacherDecision.executed_index` 记录行为策略实际执行的合法动作，和 Teacher 的
`chosen_index` 分离。下面的训练只用该实际动作的终局净分、己方获胜和对手获胜结果；它不会把一条
continuation 扩写成所有候选的 Q 标签，也不会改变 policy logits 或网页默认 AI。

先采集“每局仅一个随机干预、随后回到冻结 run4”的按墙分组语料；`executed_probability` 是可审计目标，
不会作为模型输入：

```bash
.venv/bin/python scripts/collect_candidate_teacher_dagger_trajectories.py \
  --checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --profile classic --seed-count 240 --seed 202608360 \
  --uniform-exploration-probability 0.40 --intervention-max-decisions 16 \
  --train-fraction 0.7 --validation-fraction 0.15 --device cuda \
  --output-dir artifacts/afterstate-intervention-classic-v1-run1
```

```bash
.venv/bin/python scripts/train_afterstate_outcomes.py \
  --train artifacts/afterstate-intervention-classic-v1-run1/train.trajectories.jsonl \
  --validation artifacts/afterstate-intervention-classic-v1-run1/validation.trajectories.jsonl \
  --test artifacts/afterstate-intervention-classic-v1-run1/test.trajectories.jsonl \
  --init-checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --require-known-propensity --only-randomized-actions \
  --output-dir artifacts/policy-value-afterstate-intervention-classic-v1-run1 \
  --device cuda
```

默认只接收 Teacher self-play 和 candidate-vs-Teacher DAgger 的一致后续策略，随机行为轨迹默认排除。
检查点会标记为 `diagnostic_only_not_authorized_for_action_selection`；只有在按墙隔离的校准、行为支持度审计
及新的 200/400 墙配对实战均通过后，才可以测试小幅 policy-prior 混合。

run4 的后状态 selector 已被独立实战否决，不能继续作为新的数据行为基线。若要研究“是否能在 Teacher 的某一个
响应点安全偏离”，可改用 `--teacher-base` 收集因果单点干预：干预前后均严格回到 Teacher，只有随机选中的一个
response 按已知概率采样。它是数据收集入口，不是部署 selector；产物仍须先有独立的校准和离线支持度证据：

```bash
.venv/bin/python scripts/collect_candidate_teacher_dagger_trajectories.py \
  --teacher-base --profile classic --seed-count 800 --seed 202608463 \
  --uniform-exploration-probability 0.40 --intervention-max-decisions 4 \
  --intervention-phase response --train-fraction 0.7 --validation-fraction 0.15 \
  --output-dir artifacts/teacher-response-intervention-v1
```

对 Teacher 数据，禁止使用已否决的 run3/run4 参数作初始化。改用同一个从未训练的随机 policy anchor（仅固定集成
成员间的 logits；不参与选牌），训练多个不同随机种子的可训练 afterstate head：

```bash
.venv/bin/python scripts/train_afterstate_outcomes.py \
  --train artifacts/teacher-response-intervention-v1/train.trajectories.jsonl \
  --validation artifacts/teacher-response-intervention-v1/validation.trajectories.jsonl \
  --test artifacts/teacher-response-intervention-v1/test.trajectories.jsonl \
  --fresh-policy-anchor-seed 202608471 --epochs 48 --batch-size 256 \
  --require-known-propensity --only-randomized-actions --decision-phase response \
  --output-dir artifacts/policy-value-teacher-response-outcome-v1-seed1 --device cuda --seed 1
```

若有多个按物理牌墙隔离的单点干预墙组，可用 `--additional-train`、`--additional-validation` 和
`--additional-test` 只追加同一分区。用训练墙和验证墙训练两个以上、不同随机种子的 afterstate outcome head 后，
只能在**从未参与其训练／选 epoch 的 test 墙组**上运行分组 IPS/DR：

```bash
.venv/bin/python scripts/audit_teacher_response_intervention_ope.py \
  --outcome-checkpoint artifacts/policy-value-teacher-response-outcome-v1-seed1/policy-value.pt \
  --outcome-checkpoint artifacts/policy-value-teacher-response-outcome-v1-seed2/policy-value.pt \
  --data artifacts/teacher-response-intervention-v1/test.trajectories.jsonl \
  --device cuda --output artifacts/teacher-response-intervention-v1/ope-audit.json
```

审计按物理牌墙而非单条 decision 计算置信区间，核对 `ε × uniform + (1−ε) × Teacher` 的精确 propensity，并同时要求
IPS、DR 的 95% 下界为正和两侧 ESS 达标。即使通过，它也只解锁「每局**最多一次** response override、随后回到
Teacher」的全新 200 墙四座轮换筛选；不解锁完整策略、400 墙终检、网页替换或任何对人类强度的声明。

若要从一个有限阈值网格中选择候选，不能把同一 test 墙同时用于选择和终检。先把未读 held-out 池按完整物理墙分成
`selection` 与 `terminal`，训练时传 `--skip-test`，再使用选择器；它在 selector 全部失败时不读取 terminal：

```bash
.venv/bin/python scripts/split_heldout_trajectory_groups.py \
  --input artifacts/teacher-response-intervention-v2/test.trajectories.jsonl \
  --output-dir artifacts/teacher-response-intervention-v2-heldout \
  --selection-fraction .5 --split-salt teacher-response-v2-selector-terminal-v1

.venv/bin/python scripts/select_teacher_response_override.py \
  --outcome-checkpoint artifacts/policy-value-teacher-response-outcome-v2-seed1/policy-value.pt \
  --outcome-checkpoint artifacts/policy-value-teacher-response-outcome-v2-seed2/policy-value.pt \
  --selection-data artifacts/teacher-response-intervention-v2-heldout/selection.trajectories.jsonl \
  --terminal-data artifacts/teacher-response-intervention-v2-heldout/terminal.trajectories.jsonl \
  --minimum-lcb-advantage 0 --minimum-lcb-advantage 12 \
  --minimum-lcb-advantage 24 --minimum-lcb-advantage 36 \
  --output artifacts/teacher-response-intervention-v2-heldout/selection-result.json
```

### response 定向反事实 Q（离线门槛，尚不选牌）

可显式只收集 response 的逐合法动作分支；每个分支的暗牌和牌墙只在规则 collector 内存中使用，导出的输入和
JSONL 仍只有本家手牌、公开状态、合法动作与对齐的终局分数。它是 Monte-Carlo 监督数据，不是网页运行时 oracle：

```bash
.venv/bin/python scripts/collect_action_value_trajectories.py \
  --checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --profile classic --seed-count 240 --samples-per-hand 2 \
  --decision-phase response --rollout-batch-size 32 --device cuda \
  --output-dir artifacts/counterfactual-action-value-response-direct-classic-v1
```

Q-only 校准必须启用独立 Q encoder 和 policy 路径冻结；`--action-value-weight 0` 与 `--value-weight 0` 是强制组合，
不能让 Q loss 或 AdamW weight decay 修改 run4 policy。先用 `audit_response_q_policy.py` 在按墙隔离的留出集比较 Q
和冻结 policy 的最优率／后悔配对区间；两项 95% 下界都为正之前，Q 不能进入真实对局：

```bash
.venv/bin/python scripts/audit_response_q_policy.py \
  --checkpoint artifacts/policy-value-direct-response-q/policy-value.pt \
  --reference artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --data artifacts/counterfactual-action-value-response-direct-classic-v1/test.trajectories.jsonl \
  --device cuda
```

训练器还提供 `--action-value-centered-regression`（去掉同一信息集各动作共享的绝对终局分）和
`--action-value-rank-loss-weight`（独立 Q 的 listwise 排序损失）；它们只用于离线消融，不会自动选牌。若使用
`--policy-top-k` 或 `--minimum-q-advantage-points` 审计选择器，阈值必须只在 validation 墙组确定，并在从未使用的
test 墙组报告一次；test 的平均值、MAE 或准确率都不能绕过两项正向置信下界和后续真实对局验收。

若要探索超越 Teacher 的方向，可对 policy-value 候选做终局净分 PPO 微调。它只允许一席
候选策略采样，三席可按概率混合 Teacher、显式冻结 checkpoint，以及“本轮更新前冻结”的当前
策略快照；策略仍只能从规则引擎的合法动作中选择。快照不与 actor 共享参数或梯度，并在每轮
更新后才刷新。该命令输出的是研究 checkpoint，不会自动替换网页 AI：

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

要检查策略是否只适应了三名 Teacher，可把模型与 Teacher 组成固定三人阵容。候选会在同一副
物理牌墙上轮换四座，阵容内每名对手也会覆盖候选周围的三个相对座位；只有**相同阵容标签**的
结果可做配对分差：

```bash
# 建立 Teacher 在该阵容中的基线
.venv/bin/python scripts/evaluate_policy.py \
  --teacher-candidate \
  --opponent-checkpoint artifacts/policy-value-classic-v1-run3/policy-value.pt \
  --opponent-checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --profile classic --hands 200 --seed 24000000 --device cuda

# 在完全相同的阵容和牌墙上评估新 checkpoint；--reference 同样复用该阵容
.venv/bin/python scripts/evaluate_policy.py \
  --checkpoint artifacts/new-candidate/policy-value.pt \
  --reference artifacts/policy-value-classic-v1-run3/policy-value.pt \
  --opponent-checkpoint artifacts/policy-value-classic-v1-run3/policy-value.pt \
  --opponent-checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --profile classic --hands 200 --seed 24000000 --device cuda
```

规则候选也必须走同一评测接口。`--one-ply-lookahead-teacher-candidate` 是一个只使用本家手牌和公开
河/副露/翻金的一步前瞻筛选器，**不是**网页 AI；其首个独立 20 墙筛选已显著劣于 Teacher，保留该开关仅为
复核负结果：

```bash
.venv/bin/python scripts/evaluate_policy.py \
  --one-ply-lookahead-teacher-candidate \
  --profile classic --hands 20 --seed 202608401
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

在增加此类 PPO 预算前，先运行固定 actor 的 critic-on/off 方差 A/B。该工具用独立墙组校准 critic，随后
在**同一** actor、牌墙、座位和对手抽样下回放两次；只有轨迹逐条一致时才比较 `reward - baseline` 的聚合
残差。它不保存 critic、墙、暗手或私有特征：

```bash
.venv/bin/python scripts/measure_ppo_baseline_variance.py \
  --checkpoint artifacts/policy-value-classic-v1-run4-dagger/policy-value.pt \
  --profile classic --calibration-episodes 512 --evaluation-episodes 512 \
  --rollout-batch-size 32 \
  --opponent-checkpoint artifacts/policy-value-classic-v1-run3/policy-value.pt \
  --teacher-opponent-probability 0.75 --device cuda \
  --output artifacts/ppo-baseline-variance.json
```

针对 `.pt` checkpoint，采集和评测命令会记录推理设备；同一轮比较必须固定同一设备，避免
CPU/GPU 浮点舍入在临界动作处造成不必要的策略差异。

若要训练对多种历史策略都稳健的候选，可重复传入 `--opponent-checkpoint`，并通过
`--teacher-opponent-probability 0.75` 保留 75% Teacher 对手。批量 rollout 会把候选及
同一冻结 checkpoint 的对手请求分别合并成变长合法动作 batch；规则引擎和结算不会被并行
模型绕过。

可额外使用 `--self-play-opponent-probability 0.25`，并相应把 Teacher 概率降为 `0.75` 或更低。
每一轮的 snapshot 均从更新前 actor 复制，报告会分别计数 `current_policy_snapshot`、Teacher 和冻结
checkpoint 对手；它是有待独立评测的训练消融，不是自动晋升条件。

除单元测试外，改动网页交互后应启动本地服务并通过真实浏览器完成至少一局。
