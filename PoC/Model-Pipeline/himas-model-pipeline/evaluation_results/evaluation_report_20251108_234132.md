# HIMAS Federated Model - Test Set Evaluation Report

**Evaluation Date:** 2025-11-08 23:41:32

**Model Path:** `models/himas_federated_mortality_model.keras`

---

## Executive Summary

The federated learning model was evaluated on test data from three hospitals, comprising a total of 12674 patient ICU stays. The model achieved an overall accuracy of 88.36% with an ROC AUC of 0.915, demonstrating strong discriminative ability for ICU mortality prediction.

## Performance by Hospital

### Hospital A

- **Test Samples:** 5,059
- **Mortality Prevalence:** 11.72%
- **Accuracy:** 88.52%
- **Precision:** 50.64%
- **Recall (Sensitivity):** 79.60%
- **Specificity:** 89.70%
- **F1 Score:** 0.619
- **ROC AUC:** 0.912
- **Average Precision:** 0.569

**Confusion Matrix:**
- True Negatives: 4,006
- False Positives: 460
- False Negatives: 121
- True Positives: 472

### Hospital B

- **Test Samples:** 4,378
- **Mortality Prevalence:** 12.29%
- **Accuracy:** 88.47%
- **Precision:** 51.86%
- **Recall (Sensitivity):** 85.50%
- **Specificity:** 88.88%
- **F1 Score:** 0.646
- **ROC AUC:** 0.929
- **Average Precision:** 0.663

**Confusion Matrix:**
- True Negatives: 3,413
- False Positives: 427
- False Negatives: 78
- True Positives: 460

### Hospital C

- **Test Samples:** 3,237
- **Mortality Prevalence:** 11.09%
- **Accuracy:** 87.98%
- **Precision:** 47.27%
- **Recall (Sensitivity):** 72.42%
- **Specificity:** 89.92%
- **F1 Score:** 0.572
- **ROC AUC:** 0.902
- **Average Precision:** 0.541

**Confusion Matrix:**
- True Negatives: 2,588
- False Positives: 290
- False Negatives: 99
- True Positives: 260

## Aggregated Performance Across All Hospitals

- **Total Test Samples:** 12,674
- **Overall Mortality Prevalence:** 11.76%
- **Accuracy:** 88.36%
- **Precision (PPV):** 50.32%
- **Recall (Sensitivity):** 80.00%
- **Specificity:** 89.48%
- **NPV:** 97.11%
- **F1 Score:** 0.618
- **ROC AUC:** 0.915
- **Average Precision:** 0.592

## Clinical Interpretation

The model demonstrates strong performance with a recall of 80.00%, indicating it successfully identifies 80.00% of patients who will experience ICU mortality. The precision of 50.32% suggests that when the model predicts mortality, it is correct 50.32% of the time. The high specificity of 89.48% indicates the model rarely generates false alarms for patients who will survive.

The ROC AUC of 0.915 demonstrates excellent discriminative ability, substantially exceeding the performance of random prediction (AUC = 0.5). This suggests the federated learning approach successfully learned meaningful patterns across the three hospitals without requiring centralized patient data.

## Visualizations

The following visualizations are available in the `figures/` directory:

1. **ROC Curves** - Receiver Operating Characteristic curves for each hospital
2. **Precision-Recall Curves** - Performance across different decision thresholds
3. **Confusion Matrices** - Classification outcomes for each hospital
4. **Metrics Comparison** - Side-by-side comparison of key performance indicators
5. **Prediction Distribution** - Distribution of predicted probabilities by outcome

---

*Report generated automatically by HIMAS Model Evaluation System*
