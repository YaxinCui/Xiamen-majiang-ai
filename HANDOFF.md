# 厦门麻将 AI 项目交接

更新时间：2026-08-11
工作分支：`agent/rule-policy-mlp`
GitHub PR：<https://github.com/YaxinCui/Xiamen-majiang-ai/pull/1>

## 一句话状态

当前正式基线仍是 `HeuristicTeacherAgent`。规则、网页验桌、训练数据契约、人工盲审和多条小模型／因果评测链路已经具备，
但截至本次交接，没有任何神经网络或新规则候选通过“独立物理墙、四座轮换、配对 95% 置信区间下界大于 0”的强度门槛。
因此网页默认继续使用规则 Teacher，不能声称模型已超过 Teacher 或普通人类。

## 已完成能力

- 经典厦门与核心教学两种规则档位，规则引擎负责合法动作和结算。
- 网页支持 `1 人 + 3 AI`、四人手动验规则，以及即时／5／10／15 秒 AI 决策间隔。
- 人类训练记录与真人强度评测记录从来源上隔离；`evaluation` 数据会被训练入口拒绝。
- 训练样本支持完整公开历史、按物理墙分组切分、不可混用的 train／validation／test，以及 actor-visible 泄漏检查。
- 已实现小 MLP、Teacher gate、PPO、局部 SlowExpert、随机干预、OPE、同墙控制变量和人工审阅所需的训练／审计工具。
- 已建立论文、开源工程、强化学习、数据与评测、抚州麻将、技术路线和研究反思知识库。

## 当前结论与冻结边界

### 正式模型

- 默认策略：`xiamen_mahjong.agents.HeuristicTeacherAgent`。
- 它是规则评分器，不需要 `.pt` 参数文件。
- Git 中已保留若干历史 checkpoint 供复核，但它们都是实验资产，不是默认部署模型。

### 资源政策

- 当前不再训练 Transformer／attention 大模型。
- full-history Transformer 用 724,243 条训练决策达到 98.72% Teacher 动作一致率，但只证明更会复制 Teacher，
  没有提供超过 Teacher 的强度证据，状态固定为 `diagnostic_only_retired_by_resource_policy`。
- 后续模型限制为结构化 linear／GBDT／小 MLP，近期参数上限约 50 万、checkpoint 目标小于 5 MB；
  模型只学习经独立证据确认的 Teacher 局部错误，gate 外严格回退 Teacher。

### 最近主要实验

- 精确向听候选：100 墙均分 `+0.2675`，95% CI `[−1.4671,+2.0021]`，拒绝。
- 两摸公开听牌 SlowExpert：100 墙均分 `+1.335`，95% CI `[−0.9195,+3.5895]`，拒绝。
- 六权重参数化 Teacher：训练段为正，独立 validation 均分 `−1.53`，拒绝。
- Teacher-clone reference-KL PPO v2：100 墙均分 `−0.135`，95% CI `[−1.9748,+1.7048]`，拒绝。
- exact-tie source-world／future-average 小模型：内部排序准确率约 51%–52%，外部强度或高支持因果确认未通过，整族冻结。
- 低分差 top-2 因果 residual v2：同墙控制变量显著降低标准误，但 1%／2%／5%／10% 四个 gate 的
  Bonferroni 下界仍全部小于 0，terminal 未创建，整族冻结。
- 45 类可解释因果错误地图没有稳定正下界，不能事后挑类别继续调参。

以上失败都不是“代码没跑通”，而是候选在预注册的独立证据门槛上没有证明优于 Teacher。不得在已读 holdout 上改阈值重跑。

## 当前首要阻塞

项目缺的不是更多 Teacher 模仿数据或更大的网络，而是 Teacher 之外、低噪声且不会泄漏隐藏世界的纠错信号。
当前三个本地人工标签文件都还是 0 条：

- 普通弃牌纠错：目标至少 500 个 confirmed、50 个 Teacher 分歧、100 个 group。
- 完全并列匿名二选一：固定 227 题；目标至少 100 个 confirmed、20 个非 Teacher 偏好、75 个 group。
- 吃／碰／过响应纠错：使用独立题库、独立 split 和独立 gate，不能与弃牌标签合并凑门槛。

在相应数据门槛通过前，不启动新 residual 训练。

## 建议的下一步

1. 先完成 `51863` 的 exact-tie 匿名二选一；这是当前人工成本最低、问题最明确的入口。
2. 运行只读审计，确认匿名、去重、group 数量、candidate preference 数量与来源哈希均合格。
3. 只有达到固定门槛，才按 group 做 80/10/10 切分，并训练 fresh hidden-64 CPU 小 MLP pairwise residual。
4. validation 只选择一次低覆盖 gate；未通过时 test 保持未读。通过后再运行全新 100 墙 Teacher screen。
5. 若 pairwise 数据门槛失败，回到更广的 600 题普通弃牌盲审；不要继续扩大 exact-tie 模型或复用已拒绝 holdout。
6. 只有 100 墙配对置信下界大于 0，才做全新 400 墙确认；400 墙仍通过后，才讨论替换网页默认 Teacher。

## 本地运行入口

项目虚拟环境：`/home/ubuntu/Desktop/Xiamen-majiang-ai/.venv`

主麻将桌：

```bash
.venv/bin/python scripts/serve_web_game.py --host 0.0.0.0 --port 51860
```

完全并列匿名二选一：

```bash
.venv/bin/python scripts/serve_exact_tie_pairwise_review.py \
  --host 0.0.0.0 --port 51863 \
  --queue local_human_data/exact-tie-pairwise-review-v1.queue.jsonl \
  --output local_human_data/exact-tie-pairwise-review-v1.labels.jsonl
```

浏览器入口：

- <http://localhost:51860/>：主麻将桌。
- <http://localhost:51863/>：exact-tie 匿名二选一。
- Ubuntu 当前局域网地址曾为 `192.168.1.7`；地址可能变化，应以 `hostname -I` 为准。

服务没有账号或公网访问控制。`0.0.0.0` 只适合可信局域网，不应直接映射到公网。服务进程不会在机器重启后自动恢复。

## exact-tie 审阅完成后的命令

```bash
.venv/bin/python scripts/audit_exact_tie_pairwise_reviews.py \
  --input local_human_data/exact-tie-pairwise-review-v1.labels.jsonl \
  --queue local_human_data/exact-tie-pairwise-review-v1.queue.jsonl

.venv/bin/python scripts/split_exact_tie_pairwise_reviews.py \
  --input local_human_data/exact-tie-pairwise-review-v1.labels.jsonl \
  --queue local_human_data/exact-tie-pairwise-review-v1.queue.jsonl \
  --output-dir local_human_data/exact-tie-pairwise-review-v1-split

.venv/bin/python scripts/train_exact_tie_pairwise_residual_v1.py \
  --review-split-dir local_human_data/exact-tie-pairwise-review-v1-split \
  --output-dir artifacts/exact-tie-pairwise-residual-v1 \
  --device cpu
```

先阅读 [exact-tie 协议](knowledge_base/exact_tie_pairwise_review_v1_protocol.md)，不得省略门槛或改变 split 身份。

## 数据与模型资产边界

- `local_human_data/` 默认被 Git 忽略，包含个人选择，不上传 GitHub。
- `artifacts/` 默认被 Git 忽略；本机约 8.1 GB，绝大部分是可重建的数据、缓存、报告和已拒绝 checkpoint。
- 仓库历史中已经显式跟踪了一组重要复核 checkpoint，包括 run3／run4、Teacher discard/response outcome ensemble
  和早期 PPO stage 参数；本次没有产生一个通过门槛、应新增上传的“当前最强模型”。
- 若未来产生通过全部门槛的 checkpoint，应只显式 `git add -f` 该 checkpoint、训练报告、输入哈希与选择报告，
  不得整体强制加入 `artifacts/`。

## 验证与接续阅读

提交前的最低验证：

```bash
.venv/bin/python -m pytest -q
git diff --check
```

优先阅读：

1. [README](README.md)
2. [训练方案与验收门槛](knowledge_base/training_program.md)
3. [实验日志](knowledge_base/experiment_log.md)
4. [资源受限训练计划](knowledge_base/research/technical_routes/resource_constrained_training_plan_2026-08-08.md)
5. [候选技术路线空间](knowledge_base/research/technical_routes/xiamen_deep_training_route_space_2026-08.md)
