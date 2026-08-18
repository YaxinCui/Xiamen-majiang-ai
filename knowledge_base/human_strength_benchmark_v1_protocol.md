# 冻结 AI 对真人强度基准 v1

日期：2026-08-08
状态：采集与只读审计基础设施就绪；没有候选通过 Teacher 门槛，尚未启动真人终评

## 能证明什么

本基准只回答一个有限问题：一个 SHA-256 已冻结的候选，在 classic 规则下作为三张 AI 座位的同一策略副本，与明确
招募范围内的真人参与者对局时，平均每个 AI 座位是否取得严格正分。它不能由单一参与者自动外推为“打败一般人类”，
也不能证明顶尖水平。

项目的强度声明分三层：

1. **超过规则 Teacher**：全新 100 墙筛选与 400 墙确认均要求按物理墙 paired 95% 下界严格大于 0；
2. **超过本地参与者／预注册群体**：本协议 evaluation-only 对局的每 AI 座位分差 95% 下界严格大于 0，并完成人工
   招募与冻结身份复核；
3. **超过一般或强人类**：必须另有多参与者、水平分层和外部可复核的预注册设计；v1 不提供这一结论。

只有第一层通过的候选才允许进入第二层，避免把真人终评反复当成模型选择集。

## 采集边界

- 启动网页时必须显式指定 `--human-log-purpose evaluation`；记录在源 metadata 中固定，训练器和训练切分器均硬拒绝；
- AI identity 必须包含 checkpoint SHA-256、wrapper 版本和完整 gate margin；同一评测文件只允许一个 identity；
- 记录模式由服务端强制隐藏三家暗手，debug 参数不能绕过；
- 只在完整牌局结束后追加 actor-visible 决策、公开历史和四家本局分差；不保存 seed、牌墙顺序、账号、设备或网络标识；
- 每次服务启动生成随机 session ID。session 只是预注册区块，不自动代表不同真人；参与者映射与同意记录保存在项目外，
  不写入训练仓库或牌局 JSONL；
- 评测牌局永远不得进入训练、early stopping、gate、规则修改或 checkpoint 选择。

## 固定统计

首轮最少 200 个结构合格完整牌局、10 个预注册 session block。先对每个 session 的真人本局分差取均值，再让各 session
等权计算标准误和 normal-approximation 95% 区间，避免长 session 取得更大统计权重。

一桌有一个真人座位和三个完全相同 AI 策略座位；每局四家分差为零和。因此：

```text
AI_team_score = -human_score
AI_per_seat_score = -human_score / 3
```

主报告使用 `ai_per_seat_score_delta_mean` 与 `ai_per_seat_score_delta_95pct_low`；历史
`ai_side_score_delta_*` 字段仅为三 AI 席合计的兼容别名，不能解释成单席强度。数值除以三不改变正负门槛，但避免夸大
幅度。

结构 gate 和正下界同时通过后仍只得到 `ready_for_manual_human_strength_review`。人工必须核对：参与者知情同意、厦门规则
熟悉度、招募范围、session/参与者或预注册区块关系、候选全程冻结、对局未被用于训练/选模。通过后的准确表述只能是：
“该冻结 AI 在该预注册本地评测群体中，每 AI 座位平均分差的 95% 下界为正。”

## 命令

候选先通过 Teacher 400 墙确认，随后才启动独立端口；以下占位符不能在真人结果后更换：

```bash
.venv/bin/python scripts/serve_web_game.py \
  --host 127.0.0.1 --port 51864 \
  --ai-checkpoint artifacts/<frozen-candidate>/policy-value.pt \
  --ai-device cpu \
  --ai-teacher-gate-margin <frozen-margin> \
  --human-log local_human_data/<frozen-id>-human-evaluation.jsonl \
  --human-log-purpose evaluation

.venv/bin/python scripts/audit_human_match_strength.py \
  --input local_human_data/<frozen-id>-human-evaluation.jsonl \
  --minimum-hands 200 --minimum-sessions 10 \
  --output artifacts/<frozen-candidate>/human-strength-audit.json
```

当前没有候选通过 Teacher 400 墙确认，故不能用 rejected PPO v2 或默认 Teacher 启动正式真人终评，也没有“打败人类”
结论。
