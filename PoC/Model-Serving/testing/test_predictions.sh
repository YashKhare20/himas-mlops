#!/bin/bash

echo "🚀 Generating diverse prediction traffic with COMPLETE features..."
echo "Press Ctrl+C to stop"
echo ""

count=0

# Complete realistic ICU patient scenarios with ALL 23 features
patients=(
  # Young healthy patient
  '{"age_at_admission": 28, "los_icu_hours": 24, "los_icu_days": 1, "los_hospital_days": 3, "los_hospital_hours": 72, "n_distinct_icu_units": 1, "is_mixed_icu": 0, "n_icu_transfers": 0, "n_total_transfers": 0, "ed_admission_flag": 1, "emergency_admission_flag": 1, "hours_admit_to_icu": 2, "early_icu_score": 0.3, "weekend_admission": 0, "night_admission": 0, "icu_type": "MICU", "first_careunit": "Medical Intensive Care Unit (MICU)", "admission_type": "EMERGENCY", "admission_location": "EMERGENCY ROOM", "insurance": "Private", "gender": "M", "race": "WHITE", "marital_status": "SINGLE"}'
  
  # Elderly critical patient - HIGH RISK
  '{"age_at_admission": 82, "los_icu_hours": 168, "los_icu_days": 7, "los_hospital_days": 14, "los_hospital_hours": 336, "n_distinct_icu_units": 2, "is_mixed_icu": 1, "n_icu_transfers": 2, "n_total_transfers": 3, "ed_admission_flag": 1, "emergency_admission_flag": 1, "hours_admit_to_icu": 1, "early_icu_score": 0.8, "weekend_admission": 1, "night_admission": 1, "icu_type": "SICU", "first_careunit": "Surgical Intensive Care Unit (SICU)", "admission_type": "EMERGENCY", "admission_location": "TRANSFER FROM HOSPITAL", "insurance": "Medicare", "gender": "F", "race": "WHITE", "marital_status": "WIDOWED"}'
  
  # Middle-aged elective surgery - LOW RISK
  '{"age_at_admission": 55, "los_icu_hours": 48, "los_icu_days": 2, "los_hospital_days": 5, "los_hospital_hours": 120, "n_distinct_icu_units": 1, "is_mixed_icu": 0, "n_icu_transfers": 0, "n_total_transfers": 0, "ed_admission_flag": 0, "emergency_admission_flag": 0, "hours_admit_to_icu": 0, "early_icu_score": 0.2, "weekend_admission": 0, "night_admission": 0, "icu_type": "CSRU", "first_careunit": "Cardiac Surgery Recovery Unit (CSRU)", "admission_type": "ELECTIVE", "admission_location": "CLINIC REFERRAL", "insurance": "Private", "gender": "M", "race": "BLACK", "marital_status": "MARRIED"}'
)

while true; do
  patient=${patients[$RANDOM % ${#patients[@]}]}
  ((count++))
  
  echo "[$count] Making prediction..."
  
  response=$(curl -s -X POST \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    -H "Content-Type: application/json" \
    -d "$patient" \
    https://himas-prediction-service-1089649594993.us-central1.run.app/predict)
  
  echo "   Response: $response"
  
  sleep 3
done
