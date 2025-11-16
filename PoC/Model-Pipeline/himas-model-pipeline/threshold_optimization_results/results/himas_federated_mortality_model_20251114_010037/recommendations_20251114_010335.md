# HIMAS Threshold Optimization Report

**Date:** 2025-11-14 01:03:35

## Threshold Methods Comparison

| Method | Threshold | Recall | Precision | F1 | FN (Missed) | FP (Alarms) |
|--------|-----------|--------|-----------|-----|-------------|-------------|
| Youden Index | 0.467 | 85.0% | 51.5% | 0.642 | 224 | 1190 |
| F1 Maximum | 0.726 | 77.8% | 59.8% | 0.676 | 331 | 778 |
| Target Recall 85% | 0.460 | 85.0% | 51.2% | 0.639 | 224 | 1205 |
| Target Recall 80% | 0.663 | 80.0% | 57.6% | 0.670 | 298 | 878 |
| Cost-Sensitive (10:1) | 0.342 | 87.9% | 45.7% | 0.601 | 181 | 1554 |
| Current (0.5) | 0.500 | 84.0% | 53.0% | 0.650 | 239 | 1108 |

## Recommended Threshold (Youden's Index)

**0.467**

- **Recall:** 85.0% (catches 1,266 of 1,490 deaths)
- **Precision:** 51.5% (1,190 false alarms)
- **F1 Score:** 0.642

## Implementation

Update `pyproject.toml`:

```toml
[tool.himas.model]
prediction-threshold = 0.4669
```
