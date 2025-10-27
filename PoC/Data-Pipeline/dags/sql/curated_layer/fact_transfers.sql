-- ----------------------------------------------------------------------------
-- Fact: All Transfers
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.curated_demo.fact_transfers` AS
SELECT transfer_id,
  subject_id,
  hadm_id,
  eventtype,
  careunit,
  intime,
  outtime,
  TIMESTAMP_DIFF(outtime, intime, HOUR) AS los_hours,
  -- Care unit type classification based on actual MIMIC demo units
  CASE
    -- ICU units
    WHEN careunit IN (
      'Trauma SICU (TSICU)',
      'Coronary Care Unit (CCU)',
      'Medical Intensive Care Unit (MICU)',
      'Surgical Intensive Care Unit (SICU)',
      'Cardiac Vascular Intensive Care Unit (CVICU)',
      'Medical/Surgical Intensive Care Unit (MICU/SICU)',
      'Neuro Surgical Intensive Care Unit (Neuro SICU)'
    ) THEN 'ICU' -- Emergency Department
    WHEN careunit IN (
      'Emergency Department',
      'Emergency Department Observation'
    ) THEN 'Emergency' -- Step-down / Intermediate units
    WHEN careunit LIKE '%Intermediate%'
    OR careunit LIKE '%Stepdown%' THEN 'Step-down' -- PACU
    WHEN careunit = 'PACU' THEN 'PACU' -- Observation
    WHEN careunit = 'Observation' THEN 'Observation' -- Unknown
    WHEN careunit = 'Unknown'
    OR careunit IS NULL THEN 'Unknown' -- Regular ward
    ELSE 'Ward'
  END AS care_type
FROM `physionet-data.mimic_demo_core.transfers`;