-- ----------------------------------------------------------------------------
-- Fact: Hospital Admissions
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.curated_demo.fact_hospital_admission` AS
SELECT adm.hadm_id,
  adm.subject_id,
  adm.admittime,
  adm.dischtime,
  adm.deathtime,
  adm.edregtime,
  adm.edouttime,
  adm.admission_type,
  adm.admission_location,
  adm.discharge_location,
  adm.insurance,
  adm.language,
  adm.marital_status,
  adm.ethnicity,
  adm.hospital_expire_flag,
  -- Calculate length of stay
  TIMESTAMP_DIFF(adm.dischtime, adm.admittime, DAY) AS los_hospital_days,
  TIMESTAMP_DIFF(adm.dischtime, adm.admittime, HOUR) AS los_hospital_hours,
  -- Calculate age at admission using anchor_year
  p.anchor_age + EXTRACT(
    YEAR
    FROM adm.admittime
  ) - p.anchor_year AS age_at_admission
FROM `physionet-data.mimic_demo_core.admissions` adm
  INNER JOIN `physionet-data.mimic_demo_core.patients` p ON adm.subject_id = p.subject_id;