#!/bin/bash

echo "🚀 Generating diverse prediction traffic..."
echo "Press Ctrl+C to stop"
echo ""

count=0

# Diverse realistic ICU patient scenarios
patients=(
  # Young healthy patient, short stay
  '{"age_at_admission": 28, "los_icu_hours": 24, "los_icu_days": 1, "los_hospital_days": 3, "los_hospital_hours": 72, "n_distinct_icu_units": 1, "is_mixed_icu": 0, "n_icu_transfers": 0, "n_total_transfers": 0, "ed_admission_flag": 1, "emergency_admission_flag": 1, "hours_admit_to_icu": 2, "early_icu_score": 0.3, "weekend_admission": 0, "night_admission": 0, "icu_type": "MICU", "first_careunit": "Medical", "admission_type": "EMERGENCY", "admission_location": "EMERGENCY ROOM", "insurance": "Private", "gender": "M", "race": "WHITE", "marital_status": "SINGLE"}'
  
  # Elderly critical patient, long stay
  '{"age_at_admission": 82, "los_icu_hours": 168, "los_icu_days": 7, "los_hospital_days": 14, "los_hospital_hours": 336, "n_distinct_icu_units": 2, "is_mixed_icu": 1, "n_icu_transfers": 2, "n_total_transfers": 3, "ed_admission_flag": 1, "emergency_admission_flag": 1, "hours_admit_to_icu": 1, "early_icu_score": 0.8, "weekend_admission": 1, "night_admission": 1, "icu_type": "SICU", "first_careunit": "Surgical", "admission_type": "EMERGENCY", "admission_location": "TRANSFER FROM HOSP", "insurance": "Medicare", "gender": "F", "race": "WHITE", "marital_status": "WIDOWED"}'
  
  # Middle-aged elective surgery
  '{"age_at_admission": 55, "los_icu_hours": 48, "los_icu_days": 2, "los_hospital_days": 5, "los_hospital_hours": 120, "n_distinct_icu_units": 1, "is_mixed_icu": 0, "n_icu_transfers": 0, "n_total_transfers": 0, "ed_admission_flag": 0, "emergency_admission_flag": 0, "hours_admit_to_icu": 0, "early_icu_score": 0.2, "weekend_admission": 0, "night_admission": 0, "icu_type": "CSRU", "first_careunit": "Cardiac Surgery", "admission_type": "ELECTIVE", "admission_location": "CLINIC REFERRAL", "insurance": "Private", "gender": "M", "race": "BLACK", "marital_status": "MARRIED"}'
  
  # Moderate severity case
  '{"age_at_admission": 65, "los_icu_hours": 96, "los_icu_days": 4, "los_hospital_days": 8, "los_hospital_hours": 192, "n_distinct_icu_units": 1, "is_mixed_icu": 0, "n_icu_transfers": 1, "n_total_transfers": 1, "ed_admission_flag": 1, "emergency_admission_flag": 1, "hours_admit_to_icu": 3, "early_icu_score": 0.6, "weekend_admission": 0, "night_admission": 0, "icu_type": "MICU", "first_careunit": "Medical", "admission_type": "EMERGENCY", "admission_location": "EMERGENCY ROOM", "insurance": "Medicare", "gender": "F", "race": "HISPANIC", "marital_status": "MARRIED"}'
  
  # Young trauma case
  '{"age_at_admission": 32, "los_icu_hours": 72, "los_icu_days": 3, "los_hospital_days": 6, "los_hospital_hours": 144, "n_distinct_icu_units": 1, "is_mixed_icu": 0, "n_icu_transfers": 0, "n_total_transfers": 1, "ed_admission_flag": 1, "emergency_admission_flag": 1, "hours_admit_to_icu": 0.5, "early_icu_score": 0.7, "weekend_admission": 1, "night_admission": 1, "icu_type": "TSICU", "first_careunit": "Trauma", "admission_type": "EMERGENCY", "admission_location": "EMERGENCY ROOM", "insurance": "Medicaid", "gender": "M", "race": "ASIAN", "marital_status": "SINGLE"}'
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
