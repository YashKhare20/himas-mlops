-- ============================================================================
-- Dataset statistics for monitoring
-- ============================================================================
CREATE OR REPLACE TABLE `erudite-carving-472018-r5.verification_demo.dataset_statistics` AS
SELECT 'Hospital A' AS hospital,
    data_split,
    COUNT(*) AS n_icu_stays,
    COUNT(DISTINCT subject_id) AS n_patients,
    SUM(icu_mortality_label) AS n_deaths,
    ROUND(AVG(icu_mortality_label), 4) AS mortality_rate,
    ROUND(AVG(age_at_admission), 1) AS mean_age,
    ROUND(AVG(los_icu_hours), 1) AS mean_icu_los_hours
FROM `erudite-carving-472018-r5.federated_demo.hospital_a_data`
GROUP BY data_split
UNION ALL
SELECT 'Hospital B',
    data_split,
    COUNT(*),
    COUNT(DISTINCT subject_id),
    SUM(icu_mortality_label),
    ROUND(AVG(icu_mortality_label), 4),
    ROUND(AVG(age_at_admission), 1),
    ROUND(AVG(los_icu_hours), 1)
FROM `erudite-carving-472018-r5.federated_demo.hospital_b_data`
GROUP BY data_split
UNION ALL
SELECT 'Hospital C',
    data_split,
    COUNT(*),
    COUNT(DISTINCT subject_id),
    SUM(icu_mortality_label),
    ROUND(AVG(icu_mortality_label), 4),
    ROUND(AVG(age_at_admission), 1),
    ROUND(AVG(los_icu_hours), 1)
FROM `erudite-carving-472018-r5.federated_demo.hospital_c_data`
GROUP BY data_split
ORDER BY hospital,
    data_split;