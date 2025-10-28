-- ============================================================================
-- FEATURE LAYER: Clinical Features for Mortality Prediction
-- ============================================================================
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.curated_demo.clinical_features` AS
SELECT ie.stay_id,
  ie.subject_id,
  ie.hadm_id,
  ie.icu_intime,
  ie.icu_outtime,
  ie.los_icu_hours,
  ie.los_icu_days,
  ie.icu_type,
  ie.first_careunit,
  ie.n_distinct_icu_units,
  ie.is_mixed_icu,
  ha.age_at_admission,
  ha.los_hospital_days,
  ha.los_hospital_hours,
  ha.admission_type,
  ha.admission_location,
  ha.insurance,
  p.gender,
  p.ethnicity,
  p.marital_status,
  -- Feature: Number of ICU transfers
  (
    SELECT COUNT(*)
    FROM `erudite-carving-472018-r5.curated_demo.fact_transfers` t
    WHERE t.hadm_id = ie.hadm_id
      AND t.care_type = 'ICU'
  ) AS n_icu_transfers,
  -- Feature: Total number of transfers
  (
    SELECT COUNT(*)
    FROM `erudite-carving-472018-r5.curated_demo.fact_transfers` t
    WHERE t.hadm_id = ie.hadm_id
  ) AS n_total_transfers,
  -- Feature: ED admission flag
  CASE
    WHEN ha.admission_location LIKE '%EMERGENCY%' THEN 1
    ELSE 0
  END AS ed_admission_flag,
  -- Feature: Emergency admission type
  CASE
    WHEN ha.admission_type = 'EMERGENCY' THEN 1
    ELSE 0
  END AS emergency_admission_flag,
  -- Feature: Time from hospital admission to ICU (hours)
  TIMESTAMP_DIFF(ie.icu_intime, ha.admittime, HOUR) AS hours_admit_to_icu,
  -- Feature: Early ICU admission score (proxy for severity)
  CASE
    WHEN TIMESTAMP_DIFF(ie.icu_intime, ha.admittime, HOUR) < 6 THEN 3 -- Very early (<6h)
    WHEN TIMESTAMP_DIFF(ie.icu_intime, ha.admittime, HOUR) < 24 THEN 2 -- Early (<24h)
    WHEN TIMESTAMP_DIFF(ie.icu_intime, ha.admittime, HOUR) < 48 THEN 1 -- Delayed (<48h)
    ELSE 0 -- Late (>48h)
  END AS early_icu_score,
  -- Feature: Weekend admission
  CASE
    WHEN EXTRACT(
      DAYOFWEEK
      FROM ha.admittime
    ) IN (1, 7) THEN 1
    ELSE 0
  END AS weekend_admission,
  -- Feature: Night admission (6pm - 6am)
  CASE
    WHEN EXTRACT(
      HOUR
      FROM ha.admittime
    ) >= 18
    OR EXTRACT(
      HOUR
      FROM ha.admittime
    ) < 6 THEN 1
    ELSE 0
  END AS night_admission
FROM `erudite-carving-472018-r5.curated_demo.fact_icu_stay` ie
  INNER JOIN `erudite-carving-472018-r5.curated_demo.fact_hospital_admission` ha ON ie.hadm_id = ha.hadm_id
  INNER JOIN `erudite-carving-472018-r5.curated_demo.dim_patient` p ON ie.subject_id = p.subject_id;