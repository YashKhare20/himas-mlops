-- ----------------------------------------------------------------------------
-- Fact: ICU Stays (Derived from Transfers)
-- Handles patients who move through multiple ICU types
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.curated_demo.fact_icu_stay` AS WITH icu_transfers AS (
    SELECT subject_id,
      hadm_id,
      transfer_id,
      careunit,
      eventtype,
      intime,
      outtime,
      -- Identify ICU care units (explicit matching for MIMIC demo)
      CASE
        WHEN careunit IN (
          'Trauma SICU (TSICU)',
          'Coronary Care Unit (CCU)',
          'Medical Intensive Care Unit (MICU)',
          'Surgical Intensive Care Unit (SICU)',
          'Cardiac Vascular Intensive Care Unit (CVICU)',
          'Medical/Surgical Intensive Care Unit (MICU/SICU)',
          'Neuro Surgical Intensive Care Unit (Neuro SICU)'
        ) THEN 1 -- Fallback: any careunit with "ICU" in the name
        WHEN careunit LIKE '%ICU%' THEN 1
        ELSE 0
      END AS is_icu,
      ROW_NUMBER() OVER (
        PARTITION BY subject_id,
        hadm_id
        ORDER BY intime
      ) AS rn
    FROM `erudite-carving-472018-r5.raw_demo.transfers`
    WHERE careunit IS NOT NULL
  ),
  icu_only AS (
    -- Filter to ICU transfers only
    SELECT subject_id,
      hadm_id,
      transfer_id,
      careunit,
      intime,
      outtime
    FROM icu_transfers
    WHERE is_icu = 1
  ),
  icu_grouped AS (
    SELECT subject_id,
      hadm_id,
      MIN(intime) AS icu_intime,
      MAX(outtime) AS icu_outtime,
      MIN(transfer_id) AS first_icu_transfer_id,
      -- Aggregate all ICU units visited
      STRING_AGG(
        DISTINCT careunit
        ORDER BY careunit
      ) AS icu_units,
      -- Count distinct ICU types
      COUNT(DISTINCT careunit) AS n_distinct_icu_units,
      -- Get first ICU (by time)
      ARRAY_AGG(
        careunit
        ORDER BY intime
        LIMIT 1
      ) [OFFSET(0)] AS first_icu_by_time
    FROM icu_only
    GROUP BY subject_id,
      hadm_id
  )
SELECT ROW_NUMBER() OVER (
    ORDER BY subject_id,
      hadm_id
  ) AS stay_id,
  subject_id,
  hadm_id,
  first_icu_transfer_id,
  icu_intime,
  icu_outtime,
  TIMESTAMP_DIFF(icu_outtime, icu_intime, HOUR) AS los_icu_hours,
  TIMESTAMP_DIFF(icu_outtime, icu_intime, DAY) AS los_icu_days,
  icu_units,
  first_icu_by_time AS first_careunit,
  n_distinct_icu_units,
  -- ICU type classification - handles multiple ICU types
  CASE
    -- Check for mixed ICU scenarios (2+ different types)
    WHEN n_distinct_icu_units >= 3 THEN 'Mixed ICU (3+ Units)'
    WHEN n_distinct_icu_units = 2 THEN 'Mixed ICU (2 Units)' -- Single ICU type (n_distinct_icu_units = 1)
    WHEN icu_units LIKE '%Medical Intensive Care Unit (MICU)%'
    OR icu_units = 'Medical/Surgical Intensive Care Unit (MICU/SICU)' THEN 'Medical ICU'
    WHEN icu_units LIKE '%Surgical Intensive Care Unit (SICU)%'
    OR icu_units = 'Trauma SICU (TSICU)' THEN 'Surgical ICU'
    WHEN icu_units LIKE '%Coronary Care Unit (CCU)%'
    OR icu_units = 'Cardiac Vascular Intensive Care Unit (CVICU)' THEN 'Cardiac ICU'
    WHEN icu_units = 'Neuro Surgical Intensive Care Unit (Neuro SICU)' THEN 'Neuro ICU'
    ELSE 'Other ICU'
  END AS icu_type,
  -- Flag for mixed ICU (useful for ML feature)
  CASE
    WHEN n_distinct_icu_units > 1 THEN 1
    ELSE 0
  END AS is_mixed_icu
FROM icu_grouped;