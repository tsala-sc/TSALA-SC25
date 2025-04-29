# TSALA: Temporal System and Application Log Analysis

This repository contains the source code, models, and evaluation scripts for **TSALA**, a TCN-based performance prediction framework for large-scale HPC systems.

> **TSALA: Improving Performance Prediction in Large-Scale Systems through Temporal System and Application Log Analysis**  
> Submitted to SC'25.

---

## Overview

TSALA predicts HPC application performance (runtime, write throughput, read throughput) by analyzing system logs. It integrates:

- Temporal convolutional networks (TCN)
- Trigonometric encoding for periodicity
- Application-name embeddings
- Overlapping job sequences

It was evaluated on two real HPC systems (TOP500), achieving up to:
- Runtime: R² = 0.88
- Write Throughput: R² = 0.94
- Read Throughput: R² = 0.92

---

## Key Contributions

- **C1**: A novel TCN-based framework capturing temporal patterns and application context.
- **C2**: A preprocessing pipeline extracting features from integrated system logs.
- **C3**: Overall evaluation on two heterogeneous HPC systems.
- **C4**: Evaluation showing impact of temporal features, embeddings, and TCN-based model.

---

## Artifact Contents

| File / Folder | Description |
|---------------|-------------|
| `src/`        | Source code for preprocessing, training, evaluation |
| `models/`     | Pretrained TSALA models (`.h5`) |
| `results/`    | Prediction outputs, R²/RMSE values, baseline comparisons |
| `write_volatility.pdf` | Write throughput volatility visualization |


---

## Requirements

- Python 3.9.19  
- TensorFlow 2.9.0  
- numpy, pandas, scikit-learn, matplotlib  
- keras-tcn

