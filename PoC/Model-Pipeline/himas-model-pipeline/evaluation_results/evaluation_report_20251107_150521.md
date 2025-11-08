# HIMAS Federated Model - Test Set Evaluation Report

**Evaluation Date:** 2025-11-07 15:05:21

**Model Path:** `models/himas_federated_mortality_model.keras`

---

## Executive Summary

The federated learning model was evaluated on test data from three hospitals, comprising a total of 12674 patient ICU stays. The model achieved an overall accuracy of 90.85% with an ROC AUC of 0.914, demonstrating strong discriminative ability for ICU mortality prediction.

## Performance by Hospital

### Hospital A

- **Test Samples:** 5,059
- **Mortality Prevalence:** 11.72%
- **Accuracy:** 90.69%
- **Precision:** 59.65%
- **Recall (Sensitivity):** 63.58%
- **Specificity:** 94.29%
- **F1 Score:** 0.616
- **ROC AUC:** 0.910
- **Average Precision:** 0.547

**Confusion Matrix:**
- True Negatives: 4,211
- False Positives: 255
- False Negatives: 216
- True Positives: 377

### Hospital B

- **Test Samples:** 4,378
- **Mortality Prevalence:** 12.29%
- **Accuracy:** 91.09%
- **Precision:** 60.88%
- **Recall (Sensitivity):** 76.95%
- **Specificity:** 93.07%
- **F1 Score:** 0.680
- **ROC AUC:** 0.929
- **Average Precision:** 0.649

**Confusion Matrix:**
- True Negatives: 3,574
- False Positives: 266
- False Negatives: 124
- True Positives: 414

### Hospital C

- **Test Samples:** 3,237
- **Mortality Prevalence:** 11.09%
- **Accuracy:** 90.76%
- **Precision:** 60.07%
- **Recall (Sensitivity):** 49.86%
- **Specificity:** 95.87%
- **F1 Score:** 0.545
- **ROC AUC:** 0.904
- **Average Precision:** 0.541

**Confusion Matrix:**
- True Negatives: 2,759
- False Positives: 119
- False Negatives: 180
- True Positives: 179

## Aggregated Performance Across All Hospitals

- **Total Test Samples:** 12,674
- **Overall Mortality Prevalence:** 11.76%
- **Accuracy:** 90.85%
- **Precision (PPV):** 60.25%
- **Recall (Sensitivity):** 65.10%
- **Specificity:** 94.28%
- **NPV:** 95.30%
- **F1 Score:** 0.626
- **ROC AUC:** 0.914
- **Average Precision:** 0.572

## Clinical Interpretation

The model demonstrates strong performance with a recall of 65.10%, indicating it successfully identifies 65.10% of patients who will experience ICU mortality. The precision of 60.25% suggests that when the model predicts mortality, it is correct 60.25% of the time. The high specificity of 94.28% indicates the model rarely generates false alarms for patients who will survive.

The ROC AUC of 0.914 demonstrates excellent discriminative ability, substantially exceeding the performance of random prediction (AUC = 0.5). This suggests the federated learning approach successfully learned meaningful patterns across the three hospitals without requiring centralized patient data.

## Visualizations

The following visualizations are available in the `figures/` directory:

1. **ROC Curves** - Receiver Operating Characteristic curves for each hospital
2. **Precision-Recall Curves** - Performance across different decision thresholds
3. **Confusion Matrices** - Classification outcomes for each hospital
4. **Metrics Comparison** - Side-by-side comparison of key performance indicators
5. **Prediction Distribution** - Distribution of predicted probabilities by outcome

---

*Report generated automatically by HIMAS Model Evaluation System*
