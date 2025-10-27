-- ============================================================================
-- Data Leakage Check
-- Filters by data_split column to check overlaps
-- All overlap columns MUST be 0
-- ============================================================================
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.verification_demo.data_leakage_check` AS
SELECT 'Hospital A' AS hospital,
  -- Count patients in each split
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_a_data`
    WHERE data_split = 'train'
  ) AS train_patients,
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_a_data`
    WHERE data_split = 'validation'
  ) AS val_patients,
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_a_data`
    WHERE data_split = 'test'
  ) AS test_patients,
  -- Check train-validation overlap
  (
    SELECT COUNT(DISTINCT t.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_a_data` t
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_a_data` v ON t.subject_id = v.subject_id
    WHERE t.data_split = 'train'
      AND v.data_split = 'validation'
  ) AS train_val_overlap,
  -- Check train-test overlap
  (
    SELECT COUNT(DISTINCT t.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_a_data` t
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_a_data` te ON t.subject_id = te.subject_id
    WHERE t.data_split = 'train'
      AND te.data_split = 'test'
  ) AS train_test_overlap,
  -- Check validation-test overlap
  (
    SELECT COUNT(DISTINCT v.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_a_data` v
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_a_data` te ON v.subject_id = te.subject_id
    WHERE v.data_split = 'validation'
      AND te.data_split = 'test'
  ) AS val_test_overlap
UNION ALL
SELECT 'Hospital B',
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_b_data`
    WHERE data_split = 'train'
  ),
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_b_data`
    WHERE data_split = 'validation'
  ),
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_b_data`
    WHERE data_split = 'test'
  ),
  (
    SELECT COUNT(DISTINCT t.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_b_data` t
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_b_data` v ON t.subject_id = v.subject_id
    WHERE t.data_split = 'train'
      AND v.data_split = 'validation'
  ),
  (
    SELECT COUNT(DISTINCT t.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_b_data` t
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_b_data` te ON t.subject_id = te.subject_id
    WHERE t.data_split = 'train'
      AND te.data_split = 'test'
  ),
  (
    SELECT COUNT(DISTINCT v.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_b_data` v
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_b_data` te ON v.subject_id = te.subject_id
    WHERE v.data_split = 'validation'
      AND te.data_split = 'test'
  )
UNION ALL
SELECT 'Hospital C',
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_c_data`
    WHERE data_split = 'train'
  ),
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_c_data`
    WHERE data_split = 'validation'
  ),
  (
    SELECT COUNT(DISTINCT subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_c_data`
    WHERE data_split = 'test'
  ),
  (
    SELECT COUNT(DISTINCT t.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_c_data` t
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_c_data` v ON t.subject_id = v.subject_id
    WHERE t.data_split = 'train'
      AND v.data_split = 'validation'
  ),
  (
    SELECT COUNT(DISTINCT t.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_c_data` t
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_c_data` te ON t.subject_id = te.subject_id
    WHERE t.data_split = 'train'
      AND te.data_split = 'test'
  ),
  (
    SELECT COUNT(DISTINCT v.subject_id)
    FROM `erudite-carving-472018-r5.federated_demo.hospital_c_data` v
      INNER JOIN `erudite-carving-472018-r5.federated_demo.hospital_c_data` te ON v.subject_id = te.subject_id
    WHERE v.data_split = 'validation'
      AND te.data_split = 'test'
  );