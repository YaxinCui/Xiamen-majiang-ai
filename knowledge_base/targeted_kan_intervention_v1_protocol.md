# 首杠机会定向二元干预 v1：预注册协议

日期：2026-08-08
状态：100 墙覆盖门槛未通过且方向不支持跳过杠；不追加、不产标签

## 为什么改查杠

TwoDraw 近似、100 墙首分歧随机干预和 exact-DP 确认都只有正点估计、置信下界跨零，该弃牌代理族已冻结。游金升级
看似有明显赔率问题，但 v4 小数据和 5,000 墙 part-01 中所有升级样本都是合成课程，自然 Teacher 自博弈覆盖为 0，
不能用课程频率伪装自然强度机会。

相反，789 条 natural Teacher train 轨迹中真实执行明杠 255、补杠 104、暗杠 44。冻结 Teacher 对明杠在墙大于
18 时几乎无条件优先，对补／暗杠也在可摸补牌时无条件执行。杠带来水数和补牌，也可能改变牌形、暴露信息与摸牌节奏；
局部启发式无法预先断言“杠”或“不杠”更好，适合用已知 propensity 的一次性随机干预直接测量。

## 固定行为与 estimand

`TargetedKanInterventionBehavior` 在每条候选座轨迹的**第一个 frozen Teacher 杠机会**：

- 暗／补杠的非杠 fallback 是同状态 frozen Teacher 的最佳合法弃牌；
- 明杠的 fallback 是从合法响应删除所有 `ming_kan` 后重新调用 frozen Teacher，通常得到碰或过；
- 以 0.5/0.5 只随机一次 Teacher 杠与非杠 fallback，随后整局恢复 Teacher；
- 无杠机会轨迹和所有非随机前后缀的 executed action 必须等于 Teacher；
- JSONL 保存 actor-visible 决策、精确 executed propensity 和终局候选座分数，不保存 wall、seed、对手暗手或行为 RNG。

主 estimand 是“首次 Teacher 杠改为 frozen 非杠 fallback、其后 Teacher”相对全 Teacher 的每物理墙四座平均分差。另在
预注册的 `an_kan/add_kan/ming_kan` 三类上分别估计；三类选择采用 Bonferroni 两侧 family α=0.05，即每类
98.333% 区间，不能看结果后合并或拆分类型。

## 固定 100 墙与门槛

- classic `seed=202637000..202637099`，每墙四座轮换；行为 RNG `202637200`；
- 至少 100 次总干预、100 个完整四座墙组；总体 fallback 分配率在 `[0.4,0.6]`；
- 某杠类至少 30 次干预、该类 fallback 分配率 `[0.3,0.7]`，且其 Bonferroni 98.333% 的
  `skip-kan − Teacher-kan` 每墙区间下界严格大于 0，才可进入新的类型限域规则候选；
- overall 95% 区间只作描述，不参与选型；没有类型通过即不产标签、不构造候选；
- 通过也必须另用全新 100 墙评测确定性类型候选；本数据不能直接部署或证明人类强度。

固定入口：

```bash
.venv/bin/python scripts/select_targeted_kan_intervention_v1.py \
  --output-dir artifacts/targeted-kan-intervention-v1
```

## 正式结果

固定 100 墙、400 条候选座轨迹已完成，运行 16.56 秒；100 个四座墙组完整、行为分配 32 次 fallback／33 次
Teacher，比例 49.23%，审计问题为 0。但只有 **65** 条轨迹遇到首杠机会，未达到预注册总干预至少 100 的结构门槛，
状态为 `selection_rejected_no_kan_training_labels`。

方向也不支持“减少杠”：overall `skip-kan − Teacher-kan` 为 **−3.705** 分／墙，95% CI
`[−7.475,+0.065]`。分类型 point estimate 全为负：暗杠 −0.590（7 次）、补杠 −0.710（18 次）、明杠
−2.405（40 次）；三项 Bonferroni 98.333% 区间均跨零，且暗／补杠不足 30 次类型门槛。不能把 overall 上界接近
0 解释为正式“Teacher 杠显著更强”，因为总覆盖门槛失败；但它足以否决以减少杠为优化方向，不追加墙、不改 fallback、
不合并类型、不产训练标签。安全 JSONL SHA-256 为
`c64511a1a43251af5fa3fe4485f1a3b20de1405a650703e52898ea74633d18c2`。
