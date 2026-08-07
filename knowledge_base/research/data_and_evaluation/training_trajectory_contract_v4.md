# 训练轨迹数据契约 v4

## 目的

让每个训练决策只使用当时该座位合法可知的信息，并让数据量扩大后仍可重放检查。它不是模型输入格式的限制：轻量模型可只读摘要，
序列模型可按游标恢复完整公开历史；两者都不能读取未来事件或私有世界。

## 一条完整物理牌局记录

```text
TrainingTrajectory
  profile / rules_version / rules
  trajectory_id / split_group_id
  source_metadata（安全导出时移除所有 seed 字段）
  public_actions[]              # 全局、按时间追加的公开事件，只存一次
  decisions[]
  outcome                       # 结算标签，不能反向写入 actor 特征
```

`public_actions` 只允许公开事件：摸牌发生、公开补花、弃牌、副露、杠、游金推进、胡牌和结算。对手摸到的牌面、暗手、牌墙顺序
与随机种子均不在其中。

## 每个决策记录

```text
TeacherDecision
  state.hand                    # 仅本家暗手
  state.public_players          # 四家公开河、副露、花数、手牌数量等
  state.public_action_count     # 该决策前，公开事件流中可见的前缀长度
  state.recent_public_actions   # 该前缀末尾 24 个、转换为本家相对座位的事件
  legal_actions / chosen_index  # 规则引擎给出的完整合法动作与 Teacher 标签
  executed_index / probability  # 行为策略实际动作及可审计 propensity（若适用）
  action_values                 # 可选训练目标，绝不是 actor 特征
```

对座位 `p` 的完整历史通过 `trajectory.public_actions[:state.public_action_count]` 得到，再把事件中的绝对座位转换为
`(seat - p) mod 4`。这一规则确保一次决定不会读取发生在它之后的弃牌、响应或结算。

## 特殊课程的边界

真实 Teacher/self-play 牌局使用完整公开历史。游金课程是单决策的构造状态，明确标注
`public_history_scope=recent_window_only`，只授权默认的短窗口训练；金牌锁课程保存其完整构造前缀。训练报告必须分别报告
完整历史和窗口课程数量，不能把课程样本伪装为独立自然对局。

## 审计

在训练前运行：

```bash
.venv/bin/python scripts/audit_training_trajectories.py \
  --input <train.jsonl> --input <validation.jsonl> --input <test.jsonl> \
  --minimum-teacher-selfplay-hands 1000 \
  --minimum-decisions 100000 \
  --minimum-response-decisions 1000 \
  --minimum-tour-decisions 136 \
  --minimum-gold-locked-decisions 136
```

审计器验证：安全导出中没有 replay 种子；决策状态中没有 `wall`、`opponent_hands` 等私有字段；历史游标不越界；最近窗口精确等于
该前缀末尾；以及规定的来源/规则状态覆盖门槛。结构错误直接拒绝训练；覆盖不足只说明还不能启动该轮 GPU 训练。
