-- ============================================================================
-- PATIENT-LEVEL SPLIT ASSIGNMENT
-- ============================================================================
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.curated_demo.patient_split_assignment` AS WITH patient_hashes AS (
    SELECT subject_id,
      anchor_age,
      anchor_year,
      anchor_year_group,
      gender,
      dod,
      -- Hospital assignment hash (0-99)
      MOD(
        ABS(FARM_FINGERPRINT(CAST(subject_id AS STRING))),
        100
      ) AS hospital_hash,
      -- Split assignment hash (0-99)
      MOD(
        ABS(
          FARM_FINGERPRINT(CONCAT('split_', CAST(subject_id AS STRING)))
        ),
        100
      ) AS split_hash
    FROM `physionet-data.mimic_demo_core.patients`
  )
SELECT subject_id,
  anchor_age,
  anchor_year,
  anchor_year_group,
  gender,
  dod,
  CASE
    WHEN dod IS NOT NULL THEN 1
    ELSE 0
  END AS is_deceased,
  -- Hospital assignment (40%, 35%, 25%)
  CASE
    WHEN hospital_hash < 40 THEN 'hospital_a'
    WHEN hospital_hash < 75 THEN 'hospital_b'
    ELSE 'hospital_c'
  END AS assigned_hospital,
  -- Split assignment (70%, 15%, 15%)
  CASE
    WHEN split_hash < 70 THEN 'train'
    WHEN split_hash < 85 THEN 'validation'
    ELSE 'test'
  END AS data_split
FROM patient_hashes;