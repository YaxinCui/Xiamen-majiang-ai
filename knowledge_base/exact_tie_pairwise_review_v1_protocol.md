# Teacher 完全并列盲态二选一 + pairwise residual v1

日期：2026-08-08
状态：`queue_ready_service_running_zero_labels`

## 为什么改成二选一

自动 exact-tie 公开进张候选在全新 100 墙改变 814/3,318 次弃牌，却没有超过 frozen Teacher。这证明公开进张不是
真值，但也暴露出 Teacher 的根本数据缺口：约一半弃牌最高分完全并列，最终只按牌号排序。

原 600 题审阅要求从整手牌中独立选最优，认知负担高；而当前最可辨识的问题只是“Teacher 默认牌与另一个完全并列牌
谁更好”。本协议把它改成匿名 pairwise 判断，同时保持以下边界：

- 两张展示牌在 frozen Teacher 中最高分完全相同；
- 一张是 Teacher 按牌号得到的默认项，一张是已被实战拒绝的公开进展 Pareto 项；
- 展示顺序由 item hash 决定，Teacher 不固定在左或右；
- 提交前 API 不返回 Teacher/candidate index、group、原 item、margin、赛果、wall、seed 或暗手；
- 候选只负责提出值得问人的对照，不是真值；
- 不确定标签不训练。

## 固定题库

来源是 immutable 600 题弃牌 queue，SHA-256
`9477e7a5099a46616ebbfd217fad1b08500031e21ca1f3f37ceebb4b9ad43ce6`。离线 actor-visible 复算结果：

- 原题 600；top-2 完全并列 600；
- 自动候选与 Teacher 分歧 253；
- 每个原始牌局 group 最多一题后，固定 227 题／227 groups；
- pairwise queue SHA-256 `0006b2603bcb32332c4d08b76b5b310d3fde046d6402bf41b944dc9467f747bc`；
- build report SHA-256 `2731dfeb00ed386dd6874d3f398e9e133f06e2c61f8d8e4e5e1a588cd9aeae9c`；
- source issues 0。

服务运行于 `http://127.0.0.1:51863`，当前 `0/227`。

## 数据门槛

首轮 fixed gate：

- confirmed directional labels ≥100；
- 人类选择非 Teacher 候选 ≥20；
- confirmed groups ≥75；
- queue-binding、重复、隐私和显示位置审计问题为 0。

这是“值得训练 pairwise residual”的门槛，不证明人类选择最优。若人类选择 candidate 少于 20，说明这个自动分歧族没有
足够纠错信号，应停止，不因 Teacher 偏好占多数而训练模仿网络。

## 切分与模型

confirmed 标签按原物理牌局 opaque group 固定 80/10/10 切分。训练器只读取 train/validation；test 在训练阶段只检查
存在，字节保持未读。

模型固定为 fresh feature-v3 candidate MLP、hidden 64、CPU、30 epoch、AdamW。每条样本只把展示的两个动作特征送入
pairwise cross-entropy；未展示合法动作既不是正例也不是负例。没有 value/Q/outcome target，不加载旧 checkpoint，
按 validation pairwise loss 选择 epoch。

即使训练完成也不能直接部署：下一阶段必须冻结 checkpoint SHA 和 validation-selected margin，再打开 pairwise test；
test 通过后只允许在 exact-tie pair 范围内做全新 100 墙 Teacher screen。

## 命令

```bash
.venv/bin/python scripts/build_exact_tie_pairwise_review_queue.py \
  --input-queue local_human_data/human-correction-review-v1.queue.jsonl \
  --output local_human_data/exact-tie-pairwise-review-v1.queue.jsonl \
  --report local_human_data/exact-tie-pairwise-review-v1.report.json

.venv/bin/python scripts/serve_exact_tie_pairwise_review.py \
  --host 127.0.0.1 --port 51863 \
  --queue local_human_data/exact-tie-pairwise-review-v1.queue.jsonl \
  --output local_human_data/exact-tie-pairwise-review-v1.labels.jsonl

.venv/bin/python scripts/audit_exact_tie_pairwise_reviews.py \
  --input local_human_data/exact-tie-pairwise-review-v1.labels.jsonl \
  --queue local_human_data/exact-tie-pairwise-review-v1.queue.jsonl

.venv/bin/python scripts/split_exact_tie_pairwise_reviews.py \
  --input local_human_data/exact-tie-pairwise-review-v1.labels.jsonl \
  --queue local_human_data/exact-tie-pairwise-review-v1.queue.jsonl \
  --output-dir local_human_data/exact-tie-pairwise-review-v1-split

.venv/bin/python scripts/train_exact_tie_pairwise_residual_v1.py \
  --pairwise-split-dir local_human_data/exact-tie-pairwise-review-v1-split \
  --output-dir artifacts/exact-tie-pairwise-residual-v1 --device cpu
```
