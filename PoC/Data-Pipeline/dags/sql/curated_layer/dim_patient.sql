-- ----------------------------------------------------------------------------
-- Dimension: Patient Demographics
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.curated_demo.dim_patient` AS
SELECT p.subject_id,
  p.gender,
  p.anchor_age,
  p.anchor_year,
  p.anchor_year_group,
  p.dod AS death_date,
  CASE
    WHEN p.dod IS NOT NULL THEN 1
    ELSE 0
  END AS is_deceased,
  -- Demographics from admissions (first admission)
  a.ethnicity,
  a.marital_status,
  a.language,
  a.insurance
FROM `erudite-carving-472018-r5.raw_demo.patients` p
  LEFT JOIN (
    SELECT subject_id,
      ethnicity,
      marital_status,
      language,
      insurance
    FROM `erudite-carving-472018-r5.raw_demo.admissions` QUALIFY ROW_NUMBER() OVER (
        PARTITION BY subject_id
        ORDER BY admittime
      ) = 1
  ) a ON p.subject_id = a.subject_id;