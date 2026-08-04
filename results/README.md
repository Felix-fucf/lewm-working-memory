# Results index

- `summary.csv`：C0–R3 与 N1–N3 的统一证据表；README 中的实验数值均来自此表。
- `figures/`：由同一 summary 表生成的 SVG 图。
- `manifests/audits/`：每个主运行的精选 raw audit。
- `manifests/v3-selector-input/`：可直接重建 V3 selection 的两套 validation audit。
- `manifests/run-records/`：去除机器绝对路径后的已核验 run record。
- `manifests/dataset-audits/`：Dfinal 与 V3 数据审计。
- `manifests/PUBLIC_ARTIFACTS.sha256`：所有公开结果文件的 SHA256。

所有 checkpoint 与数据集均为 remote-only，不在此目录中。
