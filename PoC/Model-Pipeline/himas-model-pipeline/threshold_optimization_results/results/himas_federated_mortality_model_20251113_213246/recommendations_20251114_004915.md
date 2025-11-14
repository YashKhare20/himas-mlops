# HIMAS Threshold Optimization Report

**Date:** 2025-11-14 00:49:15

## Threshold Methods Comparison

| Method | Threshold | Recall | Precision | F1 | FN (Missed) | FP (Alarms) |
|--------|-----------|--------|-----------|-----|-------------|-------------|
| Youden Index | 0.645 | 85.0% | 50.0% | 0.630 | 224 | 1266 |
| F1 Maximum | 0.862 | 71.5% | 64.0% | 0.676 | 425 | 598 |
| Target Recall 85% | 0.639 | 85.0% | 49.8% | 0.628 | 224 | 1277 |
| Target Recall 80% | 0.805 | 80.0% | 57.0% | 0.666 | 298 | 898 |
| Cost-Sensitive (10:1) | 0.625 | 85.2% | 49.3% | 0.625 | 220 | 1305 |
| Current (0.5) | 0.500 | 86.9% | 44.1% | 0.585 | 195 | 1643 |

## Recommended Threshold (Youden's Index)

**0.645**

- **Recall:** 85.0% (catches 1,266 of 1,490 deaths)
- **Precision:** 50.0% (1,266 false alarms)
- **F1 Score:** 0.630

## Implementation

Update `pyproject.toml`:

```toml
[tool.himas.model]
prediction-threshold = 0.6447
```
