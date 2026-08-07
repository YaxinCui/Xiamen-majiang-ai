# 实验与复现区

每个候选实验先写协议，再运行脚本；实验结束后追加结果，不修改历史段落中的 seed、参数和门槛。

## 最小协议模板

```text
candidate:
baseline:
rules_profile:
training_data:
actor_visible_fields:
privileged_fields:
selection_hands: 100
selection_seed:
terminal_hands: 400
terminal_seed:
promotion_gate:
status:
```

## 复现清单

- Git commit、Python 环境、依赖版本和运行设备。
- 规则 profile、候选/对手身份、动作选择头和 checkpoint SHA-256。
- 物理墙 seed 范围、座位轮换方式、是否包含流局。
- 原始聚合 JSON 与报告中的置信区间计算方法。
- 失败候选的拒绝原因，以及禁止重跑的 seed 范围。

## 当前约定

日常开发使用 100 墙；历史 160/400 墙协议只读。任何“优化成功”结论必须能从 commit、脚本、seed 和原始聚合结果重建。
