# HIMAS Threshold Optimization Report

**Date:** 2025-11-14 01:17:12

## Threshold Methods Comparison

| Method | Threshold | Recall | Precision | F1 | FN (Missed) | FP (Alarms) |
|--------|-----------|--------|-----------|-----|-------------|-------------|
| Youden Index | 0.723 | 84.4% | 51.6% | 0.640 | 233 | 1179 |
| F1 Maximum | 0.869 | 74.2% | 61.8% | 0.674 | 385 | 684 |
| Target Recall 85% | 0.667 | 85.0% | 49.5% | 0.626 | 224 | 1290 |
| Target Recall 80% | 0.820 | 80.0% | 56.8% | 0.665 | 298 | 905 |
| Cost-Sensitive (10:1) | 0.658 | 85.2% | 49.4% | 0.625 | 220 | 1302 |
| Current (0.5) | 0.500 | 86.5% | 43.7% | 0.581 | 201 | 1660 |

## Recommended Threshold (Youden's Index)

**0.723**

- **Recall:** 84.4% (catches 1,257 of 1,490 deaths)
- **Precision:** 51.6% (1,179 false alarms)
- **F1 Score:** 0.640

## Implementation

Update `pyproject.toml`:

```toml
[tool.himas.model]
prediction-threshold = 0.7234
```
