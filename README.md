# Oral microbiome and vascular health in HPP

Demonstration code for **The oral microbiome maps distinct functional and structural dimensions of vascular health**. The scripts illustrate the analysis of 34 vascular phenotypes across strain, gene-family and pathway layers, followed by genus-level comparison with NHANES.

## Files

| File | Analysis |
| --- | --- |
| `phenotypes.py` | Phenotype names, vascular domains and descriptive orientations |
| `preprocess.py` | Abundance preprocessing, phenotype cleaning and oral-health scores |
| `models.py` | Primary/sensitivity OLS, bilateral models and Bonferroni correction |
| `summaries.py` | Feature ranking, bilateral/postural patterns and NHANES comparison |
| `paired_correlations.py` | Paired Spearman correlations, bootstrap CIs and eye-effect concordance |
| `run_demo.py` | Run the HPP workflow and export CSV summaries |
| `nhanes_validation.R` | NHANES survey-weighted regression |

## Inputs

Place the study tables in `data/`. HPP inputs contain one row per participant, with measurements from the same visit. The study's participant exclusions have already been applied. The code uses the available observations in each model.

- `strain.csv`, `gene_family.csv` (or `gene_family.arrow`), `pathway.csv`, `genus.csv`: `participant_id` and the complete layer's raw abundance columns.
- `vascular_health.csv`, `blood_pressure.csv`, `carotid_ultrasound.csv`, `fundus.csv`: `participant_id` and the phenotype columns defined in `phenotypes.py`.
- `covariates.csv`: `participant_id`, `age`, `sex`, `smoking`. The descriptive table uses 1=male and 0=female. When sampling date and birth year are supplied, age is their calendar-year difference.
- `oral_health.csv`: `participant_id`, `bleeding_gums` (0–3), `abscess_or_gingivitis` (0–2), `teeth_mobility_lately` (0–3), using the recorded ordinal codes.
- `paired_eyes.csv`, when available: `participant_id` and `left_<phenotype>` / `right_<phenotype>` for the twelve retinal traits. Primary means are calculated from paired raw values; single-eye values are cleaned for supplementary analyses. Without this file, `fundus.csv` supplies the verified bilateral means.

Strain identifiers include their `g__` genus annotation for external comparison. The gene-family abundance input represents the complete layer, so its PPM normalization uses the complete row total. Study participant data are accessed through the HPP research environment.

## Analysis

Features detected in ≥200 samples are retained. Zeros are replaced with half the first percentile of positive abundance, followed by sample-level PPM normalization and log10 transformation. Phenotype cleaning estimates mean and sample SD within the shortest 95% interval, excludes values beyond 8 SD, and clips the remainder at ±5 SD.

Primary OLS models adjust for age, categorical sex and smoking, with at least 50 complete observations. Bonferroni correction covers all valid two-sided coefficient P values within each layer across all 34 phenotypes. Additional bilateral models use the same paired sample within each feature/trait and jointly correct all features × five traits × six models. Oral-health models use the primary correction procedure after adding the two ordinal scores.

Feature ranking uses association count, summed −log10(adjusted P), smallest adjusted P and identifier. Postural persistence uses the primary supine and standing-stage significance indicators. Bootstrap correlations use 1,000 paired draws, refitting covariate models in each draw. Rows represent participants in the supplied study tables.

## Run

Use Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python run_demo.py data outputs
```

Outputs include primary and oral-health model results, bilateral models, prioritized-feature heatmap source, sex-specific phenotype summaries and NHANES candidates.

For NHANES, use metadata with `sample_id`, `age`, `sex`, `smoking_status` (Never/Former/Current), `sbp`, `pulse`, `weight_4yr`, `psu`, `strata`; the abundance table contains `sample_id` and all raw genus counts. Supplied `BPXSY1–4` columns are averaged across positive readings; supplied `WTMEC2YR` is divided by two for the pooled four-year weights.

```bash
Rscript nhanes_validation.R outputs/NHANES_candidates.csv data/NHANES_metadata.csv data/NHANES_genus_counts.csv outputs/NHANES_results.csv
```

Install the R package `survey` before running. NHANES uses Bonferroni correction jointly across all valid two-sided P values for unique candidate genus–endpoint pairs pooled across both HPP strategies. Each shared genus–endpoint pair contributes one test. External support uses adjusted P<0.05; direction concordance follows the HPP candidate signs. For mapped genera, all significant HPP strains must share the external sign.

`compare_nhanes` joins the candidate and corrected external results. `nhanes_replication_summary` reports significant and direction-concordant counts for each strategy and endpoint.

Tests use synthetic data to check the statistical logic. The manuscript's numerical results are generated from the study data. Code is released under the [MIT License](LICENSE).

```bash
python -m pip install pytest
python -m pytest -q
```
