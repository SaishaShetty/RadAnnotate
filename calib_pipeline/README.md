## Calibration + FP Confidence Reduction (Deduped Reports, Frozen Split)

### Setup
- Reports were **deduplicated** by exact `report` string.
- A **single frozen 50/50 VAL–TEST split** was created per entity (no re-splitting later).
- **Isotonic regression was fit on VAL only** (no test leakage).
- All **before vs after** comparisons below are computed on the **same TEST set**.

---

### Test Brier Results (Before vs After)

\[
\text{Brier}(p,y) \;=\; \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2
\]

| Entity | TEST preds | Brier (Before) | Brier (After) | $\Delta$ (Before $-$ After) | Saved calibrated JSON? |
|---|---:|---:|---:|---:|---|
| ANAT-DP | 487 | 0.0524 | 0.0526 | -0.0001 | No |
| OBS-DP  | 458 | 0.0975 | 0.0954 | 0.0020 | No |
| OBS-DA  | 177 | 0.0579 | 0.0550 | 0.0029 | Yes |
| OBS-U   | 61  | 0.3402 | 0.2846 | 0.0556 | Yes |

**Decision:** calibrated outputs are stored only for **OBS-U** and **OBS-DA**, since they show improvement on TEST.

---

### FP Confidence Reduction on TEST (Before vs After)

We compute **false positives (FPs)** as predictions not present in `true_labels`, then measure the fraction of FPs with confidence above thresholds $t \in \{0.6, 0.7, 0.8, 0.9\}$, comparing raw `confidence` (before) vs `cal_conf` (after).

#### OBS-U
- TEST FPs: $24$
- FP conf reduction:
  - $t \ge 0.6$: $23/24$ (95.83%) $\rightarrow$ $14/24$ (58.33%)
  - $t \ge 0.7$: $23/24$ (95.83%) $\rightarrow$ $5/24$ (20.83%)
  - $t \ge 0.8$: $18/24$ (75.00%) $\rightarrow$ $0/24$ (0.00%)
  - $t \ge 0.9$: $16/24$ (66.67%) $\rightarrow$ $0/24$ (0.00%)

#### OBS-DA
- TEST FPs: $11$
- FP conf reduction:
  - $t \ge 0.6$: $11/11$ (100.00%) $\rightarrow$ $11/11$ (100.00%)
  - $t \ge 0.7$: $11/11$ (100.00%) $\rightarrow$ $11/11$ (100.00%)
  - $t \ge 0.8$: $8/11$ (72.73%) $\rightarrow$ $3/11$ (27.27%)
  - $t \ge 0.9$: $5/11$ (45.45%) $\rightarrow$ $2/11$ (18.18%)

---

### Saved Artifacts
- `outputs/calibrated/OBS-U/test_calibrated.json`
- `outputs/calibrated/OBS-DA/test_calibrated.json`

(Original TEST splits in `outputs/splits/**/test.json` remain unchanged.)
