# 吃／碰／过人工纠错审阅 v1 协议

日期：2026-08-08
状态：400 题本地不可变题库与网页已就绪；人工标签 0

## 为什么必须独立于弃牌题库

现有 `human-correction-review-v1` 的 600 题只允许普通 discard→discard，训练 gate 也明确禁止用响应、胡、杠、游金
或金牌锁凑数量。该边界不能为方便而修改。另一方面，安全 natural Teacher v4 train 的 34,456 个决策中有 7,312
个满足普通 pass/chi/pong 审阅范围，且 frozen response 极度偏向吃碰。旧响应后形分阈值族和新精确向听词典序都未通过
实战，说明该阶段缺少可靠的自动价值来源，适合单独询问熟悉厦门规则的人。

响应 queue、label、group salt、文件名、审计门槛和未来 response head 均使用新版本；绝不把它们并入 immutable 弃牌
queue，也不让一个门槛替另一个门槛。

## 题库构造与主动选题

来源仅为 `profile=classic`、`collector=teacher_self_play` 的 actor-visible 安全轨迹。每题要求：

- phase 为 response；合法集合只含一个 pass 以及至少一个 chi/pong；
- 排除 hu、ming_kan、游金、金牌锁与本家天听状态；
- 本家手牌、金牌、牌河、公开副露、剩余墙数量和 bounded 最近公开动作可见；
- wall 顺序、seed、对手暗手、赛果、源路径、原 trajectory/group ID 不导出；
- 每个物理牌局 opaque group 最多两题。

固定 Teacher v4 train 扫描 789 局、34,456 个决策，得到 7,312 个合格响应；从公开快照复现 frozen Teacher 的不一致为
**0**。已拒绝的 `DeficiencyMeldTeacherAgent` 只作 acquisition function：它与 Teacher 分歧 221 题，group cap 后保留
219 题，并排在 queue 前部；其余由 opaque hash 稳定排序。最终 400 题覆盖 319 个 group，Teacher 参考动作为 pong 220、
chi 176、pass 4。

queue SHA-256 为 `653e1063d3d46603abf6dd2f43850a9844723f3f19c34681921395848301062d`，构造报告 SHA-256 为
`397ec6e7495212d6c5459537009b009c00af15cb91cbbe4f36eb2f5e4fd4ddaa`。两者位于 Git 忽略的
`local_human_data/`。失败候选的分歧只决定询问顺序，不写入 queue 字段或人工 label，不是真值。

## 盲审、追加写入与快捷操作

本机服务监听 `127.0.0.1:51862`。GET `/api/review` 在人作答前只返回 actor-visible state、合法动作和聚合进度；不返回
Teacher index、SlowExpert index、group、priority 或源身份。网页显示待响应弃牌，并提供“过／碰／吃”按钮；←/→
切换选择、Enter confirmed、U uncertain、S 跳过。

服务端成功追加 label 后才返回 Teacher 与失败 SlowExpert 对照。label 只保存人工动作、置信度、原合法集合、Teacher
参考和 immutable queue digest；不保存 acquisition 动作。重启逐行验证 item/group/profile/rules/state/actions/reference/digest，
不属于 queue、重复 item 或被改动行都会拒绝。`uncertain` 保存采集事实但不进入训练；skip 只在当前会话生效。

## 数据门槛与未来模型边界

首轮固定门槛：

- 300 个 confirmed 响应判断；
- 至少 50 个 confirmed 人类／Teacher 分歧；
- 至少 100 个物理牌局 group；
- 无无效、重复、scope 外或 queue 绑定错误。

通过后按 opaque 物理 group 做确定性 80/10/10 切分，同局绝不跨 split。test 必须物理隔离于训练和 validation 选门。
主动排序使前 219 题不是总体随机样本，所以早期人类分歧率不能外推 Teacher 总体错误率。

未来只允许训练 CPU 小型 response residual/head；普通弃牌、胡、杠、游金和特殊状态继续 frozen。人工标签只有行为 target，
`value_target=None`，不得伪造终局收益。validation/test 必须同时要求 response override 人工精度与按 group 等权增益的
正向置信下界；通过后仍须全新 100 墙四座轮换。没有真人对局前不能声称打败人类。

## 固定小 MLP 训练与物理未读 test

训练接口已经实现，但 0 标签状态下不会运行。split 行必须由独立加载器逐行验证：只接受 confirmed、classic、普通
pass/chi/pong，拒绝重复 item、非行为 Q/value/seed/执行概率字段、越界动作、私有字段和错误来源元数据。弃牌 review 与
response review 使用两个不同 source 和两个不同许可开关，不能混淆门槛。

固定 v1 在总题库先通过 300/50/100 审计后，还要求 train+validation 至少 240 confirmed、35 个 Teacher 分歧和
80 groups。训练配置预注册为 fresh seed `202633000`、feature v3 candidate MLP、hidden 128、CPU、12 epoch；在安全
Teacher v4 语料上混合响应 review，响应基础权重 4，人工／Teacher 分歧再乘 2，只按
`local_human_response_review_opt_in` validation policy loss 选择 epoch。它不加载旧 checkpoint，也没有 response-test
命令行参数。

validation gate 只在预声明 10%/20%/30% override coverage 中选严格 logit margin；至少 5 个 override、15 groups，
override 人工精度 Wilson 95% 下界和按 group 等权的 gated−Teacher 准确率 95% 下界都必须严格大于 0（前者相对
50%）。没有 winner 时 test 文件字节保持未读。validation 通过后才以冻结 checkpoint SHA 和 margin 打开 test；test
同样至少 5 override、15 groups、覆盖不超过 35% 且两项下界为正。通过只解锁全新 100 墙 Teacher 实战，不证明强于
Teacher 或人类。

```bash
.venv/bin/python scripts/train_human_response_review_residual_v1.py \
  --review-split-dir local_human_data/human-response-review-v1-split \
  --output-dir artifacts/human-response-review-residual-v1 --device cpu

sha256sum artifacts/human-response-review-residual-v1/policy-value.pt

.venv/bin/python scripts/select_human_response_review_residual_gate_v1.py \
  --checkpoint artifacts/human-response-review-residual-v1/policy-value.pt \
  --checkpoint-sha256 <冻结检查点的 SHA-256> \
  --validation-input local_human_data/human-response-review-v1-split/validation.response-review.jsonl \
  --test-input local_human_data/human-response-review-v1-split/test.response-review.jsonl \
  --output artifacts/human-response-review-residual-v1/gate-selection.json
```

只有上述输出状态为 `response_review_test_gate_passed_ready_for_100_wall_teacher_screen`，才冻结 gate report 自身 SHA，
运行下列实战。执行包装器只允许普通 classic pass/chi/pong；任一 hu、杠、游金、金牌锁、天听、非 response 或混合合法
集合都不调用模型。selection 固定 `202634000` 起 100 墙，正下界通过才读取 `202634200` 起 400 墙 terminal：

```bash
sha256sum artifacts/human-response-review-residual-v1/gate-selection.json

.venv/bin/python scripts/select_human_response_review_teacher_screen_v1.py \
  --checkpoint artifacts/human-response-review-residual-v1/policy-value.pt \
  --checkpoint-sha256 <冻结检查点 SHA-256> \
  --gate-report artifacts/human-response-review-residual-v1/gate-selection.json \
  --gate-report-sha256 <冻结 gate report SHA-256> \
  --output artifacts/human-response-review-residual-v1/teacher-screen.json \
  --device cpu
```
