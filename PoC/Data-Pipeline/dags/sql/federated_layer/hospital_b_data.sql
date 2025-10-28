-- ----------------------------------------------------------------------------
-- Hospital B: ICU Mortality Prediction (35% of patients - Random)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.federated_demo.hospital_b_data` AS
SELECT cf.*,
  ha.hospital_expire_flag,
  ha.deathtime,
  p.death_date,
  psa.assigned_hospital,
  psa.data_split,
  -- LABEL: ICU Mortality (same as Hospital A)
  CASE
    WHEN ha.hospital_expire_flag = 1 THEN 1
    ELSE 0
  END AS icu_mortality_label
FROM `erudite-carving-472018-r5.curated_demo.clinical_features` cf
  INNER JOIN `erudite-carving-472018-r5.curated_demo.fact_hospital_admission` ha ON cf.hadm_id = ha.hadm_id
  INNER JOIN `erudite-carving-472018-r5.curated_demo.dim_patient` p ON cf.subject_id = p.subject_id
  INNER JOIN `erudite-carving-472018-r5.curated_demo.patient_split_assignment` psa ON cf.subject_id = psa.subject_id
WHERE psa.assigned_hospital = 'hospital_b';