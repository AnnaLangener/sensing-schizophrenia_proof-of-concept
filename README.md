# SensingIndividualized — Code

Research notebooks and R scripts for predicting negative, cognitive, and positive schizophrenia symptom scores from passive smartphone sensing and ecological momentary assessment (EMA). The workflow prepares participant data, fits individualized Chronos-2 models and baselines with rolling nested cross-validation, and compares held-out predictions.


## Code guide

| File | Purpose |
| --- | --- |
| `FeatureCreation_SensingSchiophrenia.ipynb` | Cleans sensor streams, constructs EMA symptom scores, and exports participant time-series bundles. |
| `MainModel1_Combined_Head.ipynb` | Model 1: frozen Chronos-2 embeddings of combined sensor inputs with a trainable prediction head. |
| `MainModel2_LoRa_Combined_Head.ipynb` | Model 2: Chronos-2 adaptation with low-rank adaptation (LoRA) and a combined-input prediction head. |
| `LOCAL_MainModel2_LoRa_Combined_Head.ipynb` | Local Model 2 variant; selects Apple MPS, CUDA, or CPU, in that order of availability. |
| `MainModel3_Reg.ipynb` | Model 3: Chronos-2 REG-token embeddings with a head predicting the 0.1, 0.5, and 0.9 quantiles. |
| `Baseline_Final.ipynb` | Runs train-mean, last-observation, random-forest, and Chronos-embedding/random-forest baselines; combines baseline and main-model metrics. |
| `Overview_Results.ipynb` | Summarizes saved predictions, inspects participant trajectories and selected hyperparameters, and combines results with baseline metrics. |
| `precompute_dtw_cache.R` | Precomputes bounded dynamic time warping (DTW) comparisons and permutation-based summaries from held-out predictions. |
| `create_main_paper_figures.R` | Generates performance-comparison and participant-trajectory figures. |
| `Result_Explorer_App.Rmd` | Runs an interactive Shiny R Markdown result explorer with model comparisons, participant predictions, and DTW summaries. |


## Environment and paths

Use Jupyter locally or Google Colab for the Python notebooks. The main Colab notebooks mount Google Drive and load `amazon/chronos-2`; initial model loading requires access to its pretrained weights. Review the pipeline device settings before running locally: the Colab setup selects CUDA, while the local Model 2 notebook provides device selection.

Dependencies referenced by the active workflow include:

- **Python:** NumPy, pandas, PyTorch, SciPy, scikit-learn, Matplotlib, seaborn, and IPython/Jupyter. Model notebooks install `chronos-forecasting>=2.0`; Model 2 also installs `peft` and `torchao>=0.16.0`. `Overview_Results.ipynb` uses `itables`.
- **R:** `rmarkdown`, `shiny`, `reticulate`, `dplyr`, `tibble`, `tidyr`, `ggplot2`, `stringr`, `purrr`, `patchwork`, `hrbrthemes`, `tidytext`, and `ggh4x` across the scripts and explorer. Reticulate needs a Python environment with PyTorch and NumPy to read prediction files.

Before execution, update the following settings for your machine:

| Component | Settings to review |
| --- | --- |
| Feature creation | `base_dir`, `participant_id`, and the full output path passed to `np.savez`. |
| Main models | `BASE_DIR`, pipeline/device settings, `participant_ids`, `outcomes`, and the training configuration. Remove or skip Colab-specific mounting cells for local runs. |
| Baselines | `LOCAL_DATA_ROOT`, `COLAB_DATA_ROOT`, embedding roots, `PRECOMPUTED_EMBEDDING_ROOT`, `OUTPUT_ROOT`, and the later `BASELINE_RESULTS_PATH`. |
| Results overview | `BASE_DIR` and the baseline CSV path in the comparison section. |
| DTW precomputation | `RESULTS_ROOT` environment variable; caches are written beside the script in `Code/dtw_cache/`. |
| Result explorer | `BASE_DIR`, `APP_DIR`, and `DTW_CACHE_DIR`; configure `RETICULATE_PYTHON` before Python initializes. |
| Paper figures | `BASE_DIR`, `APP_DIR`, `PYTHON_PATH`, and the absolute paths in the final `ggsave` calls. |

The explorer currently points its cache directory to a separate local project folder. Set its `APP_DIR` to this `Code` directory, or explicitly align `DTW_CACHE_DIR` with the precomputation output. In the figure script, changing `APP_DIR` alone does not change the hard-coded figure destinations.

## Input data

Feature creation reads four CSV files from `cleaned_out/<participant_id>/`:

```text
location_sample.csv
acceleration_sample.csv
battery_level_sample.csv
ema_response.csv
```

The model notebooks read `deeplearning_data/pred/participant_<id>_4h.npz`. Each participant bundle contains:

| Key | Contents |
| --- | --- |
| `datetimes` | EMA timestamps used to order samples. |
| `labels` | Three columns in order: negative, cognitive, positive symptoms. |
| `accel`, `battery`, `gps`, `speed` | Separate sensor-stream arrays. |
| `combined` | Combined acceleration magnitude, x/y/z, battery percentage, latitude, longitude, and speed sequences. |

The current main-model loops use `combined` as input. They sort samples by timestamp, round labels and divide by 100, and remove the first questionnaire. The baseline configuration also sets `drop_first_sample=True`; preserve this alignment for comparisons.

**Feature-window settings need to match the model inputs.** The active feature-export cell currently uses `LOOKBACK = "24H"`, excludes the four-hour EMA recall window, and writes `participant_<id>_24h.npz`. To create the four-hour inputs expected by the model notebooks, change both `LOOKBACK` and the export filename suffix to four hours. Alternatively, retain the 24-hour window and consistently update all downstream loaders. Renaming a file alone does not change its feature window.

## Running the workflow

1. **Prepare features, if needed.** Run `FeatureCreation_SensingSchiophrenia.ipynb` in order for each participant after aligning the paths and feature window. Existing compatible `.npz` files let you start with modeling.
2. **Fit the desired main models.** Run the selected main-model notebook from top to bottom. Review `participant_ids` separately in each notebook: the saved lists differ. Check `outer_cfg`, `inner_cfg`, `hyperparams_grid`, `num_seed_runs`, `AUG_TECHNIQUES`, and `FINAL_VARIANTS` before starting.
3. **Fit baselines.** Run `Baseline_Final.ipynb`, setting `RUN_CONFIG` to the intended participants and models. The six configured baselines are Train Mean, Last Observation, RF Only, Chronos Embeddings REG + RF, Chronos Embeddings Mean + RF, and Chronos Embeddings Flatten PCA + RF. Embedding-based baselines require compatible precomputed embeddings. Restrict `model_names` to run a subset.
4. **Build comparison tables.** Run the baseline notebook's “Compare with main results” section after setting `BASELINE_RESULTS_PATH` to the CSV just generated. Use `Overview_Results.ipynb` for additional summaries and inspection. Its comparison section can also write the combined summary.
5. **Explore and plot.** With result paths and the R/Python environment configured, run the commands below from `Code/`.

```sh
# Optional: precompute DTW caches for the three main models.
# Export RESULTS_ROOT first if the default project path is different.
Rscript precompute_dtw_cache.R

# Generate the paper figures after editing their output paths.
Rscript create_main_paper_figures.R

# Launch the interactive result explorer.
Rscript -e 'rmarkdown::run("Result_Explorer_App.Rmd")'
```

The checked-in main-model configuration uses an outer training window of 135 observations, test windows of 3, and a step of 3; inner validation uses 21 observations with a step of 114. These sizes count observations, not calendar days. Participants need enough retained observations for at least one outer split.

Only `no_retrain_single` is active in `FINAL_VARIANTS`, with augmentation set to `none`. Other variant definitions are commented out. The active variant evaluates the selected inner-validation model without retraining on the full outer training window. Use `FINAL_VARIANTS` in the current notebooks as the configuration reference; older documentation describes different switches.

## Outputs

```text
SensingIndividualized/
├── deeplearning_data/pred/       # Participant input bundles
├── embeddings_pred/             # Cached embeddings; layout varies by model
├── final_models_<model>_<variant>/
│   └── participant_<id>/<outcome>/outer<k>/...
└── results_true_pred/
    ├── Model1_Frozen_Combined_no_retrain_single/
    ├── Model2_LoRA_Combined_no_retrain_single/
    ├── Model3_reg_quantile_no_retrain_single/
    ├── results_summary_table.csv
    └── model_and_baseline_results.csv
```

Each main-model result folder contains `results_<participant>_<outcome>.pth`, with a list of outer-split dictionaries. These include held-out targets and predictions (`y_test`, `y_pred`), split indices, and model-selection metadata; additional fields vary by model. Model checkpoints are stored separately under `final_models_*`.

`Baseline_Final.ipynb` writes `baseline_outputs/baseline_results.csv` relative to the notebook's **working directory**, plus an in-progress CSV used for resuming. Its later comparison cell currently reads an independently hard-coded path, so align that path with `OUTPUT_ROOT`. When changing a baseline experiment, review the resume setting and output directory to avoid reusing an earlier run's progress.

`Overview_Results.ipynb` writes `results_true_pred/results_summary_table.csv`. The shared `model_and_baseline_results.csv` supplies participant-level comparisons for the R workflow. The figure script currently saves `Results1.png` and `Results2.png` to absolute local paths. The explorer reads saved DTW caches and can compute a missing participant-level cache on demand.

The project-level `../README_model_output_structure.md` provides additional background on split metadata and checkpoints. Its `main_*` / `combined_only_*` folder names and run-switch examples describe an older layout; use the model names and `FINAL_VARIANTS` documented here for the current notebooks.

## Reproducing a run

Record the notebook version, package versions, participant list, outcome list, feature window, split settings, seed settings, and enabled variants together with the results. Keep model and baseline preprocessing and splits aligned. Start with one participant and one outcome to check paths, data availability, and outputs before expanding the run. This README was checked against the source files; the training and analysis workflow was not executed as part of the documentation update.
