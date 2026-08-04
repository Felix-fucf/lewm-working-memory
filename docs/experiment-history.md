# 实验发展历程

本文档只记录可由代码、checkpoint 哈希、run record 或 audit JSON 支持的状态。`planned` 不等于已执行，失败实验也不会因展示需要被删除。

## 证据状态

- `confirmed_complete`：训练完成，存在最终 checkpoint SHA256、run record 与审计结果。
- `confirmed_partial`：确实启动过，但只留下 2–5 个 epoch 或异常退出记录。
- `superseded`：曾经有效，但被后续数据或方法审计推翻，不能再作为当前结论。
- `planned`：有方案或配置，没有运行 artifact。

## 1. 早期 OOT / Leaky-RNN 探索

`e100` 完成 100 epochs；primitive-v1 E15 与 main25 也留下完整或可审计的训练产物。这一阶段建立了遮挡任务、递归状态和 action alignment，但实现仍允许严格瓶颈之外的信息路径，因此不能作为 corrected baseline。公开仓库保留方法源代码、摘要和最终 checkpoint 哈希；逐 epoch 权重只列入待清理清单。

状态：`confirmed_complete`，但对当前主结论为 `superseded`。

## 2. 表示塌缩与 normalization 排查

primitive-v1 E15、main25 完整运行；randh0、zeroh0、peak、SIGReg 只运行 2–5 epochs。审计显示：低训练 loss、高 effective rank 或线性 probe 可解码，单独都不足以证明模型在遮挡后因果地使用了对象记忆。

状态：main25 为 `confirmed_complete`；randh0、zeroh0、peak、SIGReg 为 `confirmed_partial`。

## 3. 严格递归瓶颈 C0–R3

C0、R1、R2、R3 均完成 25 epochs，采用同一严格递归瓶颈。R3 的 hidden-shuffle 相对增幅达到 6.79%，初看像阳性结果；但 matched-future direction alignment 为 -0.00094，位置 probe 也为负，无法证明预测沿正确未来方向变化。

状态：`confirmed_complete`；“R3 已证明工作记忆”的解释为 `contradicted`。

证据：[summary.csv](../results/summary.csv)、[R3 audit](../results/manifests/audits/R3/validation_own/)。

## 4. Dfinal 缺陷与 V3 数据修正

旧 Dfinal 通过了当时的 acceptance 条件，但事后 audit 发现：99.9875% 的 episode 位于“当前输入相同、future 不同”的冲突组。也就是说，旧数据允许模型在无法从当前可见信息辨识未来的条件下被评价，R3 的解释必须撤回。

V3 将第一段遮挡固定为 H4，第二段遮挡设置五个均衡 reveal 时刻，随机 onset 的 target weight 置零，并统一 reveal 权重；同时保留 paired comparison 和边界检查。

状态：旧 Dfinal 结论 `superseded`；V3 数据生成与 audit 为 `confirmed_complete`。

证据：[Dfinal audit](../results/manifests/dataset-audits/dfinal.audit.json)、[V3 audits](../results/manifests/dataset-audits/)。

## 5. V3 N1–N3 正式筛选

N1、N2、N3 均完成 25 epochs；N4/N5 没有启动。N1 对 reset/shuffle 很敏感，但 no-memory 增幅只有 0.14%，prediction effective rank 仅 1.56，且 x/y probe 为负。N2/N3 恢复了 prediction rank，但 shuffle 增幅只有约 0.26%–0.30%，final NMSE 接近均值基线，位置语义和 matched-future direction 仍未恢复。

重新从现存 raw audit 运行 selector，输出 `selected=null`。

状态：N1–N3 `confirmed_complete`；N4/N5 `planned_not_run`；“存在优胜模型”被证据否定。

证据：[selection_decision.json](../results/manifests/selection_decision.json)、[provenance](../results/manifests/selection_decision.provenance.json)。

## 6. 冻结的下一步

下一步只做 visible-frame representation gate：先验证 encoder 在可见帧能否编码 x/y、angle 与 velocity，再决定是否引入 projector、预训练 encoder 或新的 prediction objective。

状态：`planned`，尚无运行 artifact。
