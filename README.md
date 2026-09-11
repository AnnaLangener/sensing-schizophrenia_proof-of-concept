# Personalized Sensing Schizophrenia

## Workflow

1. **Feature cleaning**: `FeatureCreation_SensingSchiophrenia_new.ipynb` prepares features for model training.
2. **Main models**: Model 1, Model 2, and Model 3 are represented by the result folders `Model1_Frozen_Combined_no_retrain_single`, `Model2_LoRA_Combined_no_retrain_single`, and `Model3_reg_quantile_no_retrain_single`. Relevant notebooks include `Seed_MainModel_Chronos2_Adapter.ipynb`, `LOCAL_MainModel2_LoRa_Combined_Head.ipynb`, and `MainModel3_Reg.ipynb`.
3. **Baseline models**: Chronos and random-forest baselines are implemented in the `Baseline_Chronos*.ipynb` and `BaselineRF_Chronos2_TryOut.ipynb` notebooks.
4. **Shared model outputs**: `results_true_pred/model_and_baseline_results.csv` contains participant-level summary metrics. Per-participant held-out predictions are stored as `results_<participant>_<outcome>.pth` within each model result folder.
5. **DTW results**: Run `Rscript precompute_dtw_cache.R` to create `dtw_cache/*.rds` files from the held-out `.pth` predictions.
6. **Paper figures**: Run `create_main_paper_figures.R`. Figure 1 reads the summary CSV; Figure 2 reads the summary CSV and Model 1 held-out prediction files.
7. **Result explorer**: `Result_Explorer_App.Rmd` reads the summary CSV and prediction files for interactive model comparisons. It reads DTW cache files for symptom-level DTW summaries and can create a missing participant-level DTW cache on demand.


