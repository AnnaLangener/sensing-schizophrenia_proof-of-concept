#!/usr/bin/env Rscript


library(dplyr)
library(ggplot2)
library(hrbrthemes)
library(reticulate)
library(tidyr)

APP_DIR <- "/Users/f007qrc/Library/CloudStorage/GoogleDrive-anna.m.langener@dartmouth.edu/My Drive/Darmouth Drive/6_PersonalizedSensingSchizophrenia"
BASE_DIR <- paste0(
  "/Users/f007qrc/Library/CloudStorage/",
  "GoogleDrive-anna.m.langener@dartmouth.edu/",
  ".shortcut-targets-by-id/",
  "1_T-CzrkNtF9QAhex6qrEOA0ZQIL7g5u5/",
  "SensingIndividualized"
)

RESULTS_ROOT <- file.path(BASE_DIR, "results_true_pred")
SUMMARY_PATH <- file.path(RESULTS_ROOT, "model_and_baseline_results.csv")
PYTHON_PATH <- "/Users/f007qrc/.virtualenvs/r-reticulate/bin/python"
OUTCOMES <- c("negative", "cognitive", "positive")

SELECTION_MODEL <- "Model1_Frozen_Combined_no_retrain_single"
PLOT_MODEL_DIRS <- c(
  "Model 1" = "Model1_Frozen_Combined_no_retrain_single"
  # "Model 2" = "Model2_LoRA_Combined_no_retrain_single",
  # "Model 3" = "Model3_reg_quantile_no_retrain_single"
)
EXCLUDED_SELECTION_PARTICIPANTS <- c("00011", "00092")

ALL_MODEL_LABELS <- c(
  "Model1_Frozen_Combined_no_retrain_single" = "Model 1 (Frozen)",
  "Model2_LoRA_Combined_no_retrain_single" = "Model 2 (LoRA)",
  "Model3_reg_quantile_no_retrain_single" = "Model 3 (Quantile)",
  "Chronos Embeddings Flatten PCA + RF" = "Chronos PCA + RF",
  "Chronos Embeddings Mean + RF" = "Chronos Mean + RF",
  "Chronos Embeddings REG + RF" = "Chronos REG + RF",
  "Last Observation" = "Last observation",
  "RF Only" = "RF only",
  "Train Mean" = "Train mean"
)


Sys.setenv(RETICULATE_PYTHON = PYTHON_PATH)


safe_cor <- function(x, y) {
  keep <- complete.cases(x, y)
  x <- x[keep]
  y <- y[keep]
  if (length(x) < 2 || sd(x) == 0 || sd(y) == 0) NA_real_ else cor(x, y)
}

outcome_label <- function(outcome) {
  recode(outcome, negative = "Negative", cognitive = "Cognitive", positive = "Positive")
}

save_figure <- function(figure, output_stub, width, height) {
  ggsave(paste0(output_stub, ".png"), figure, width = width, height = height,
         units = "in", dpi = 300)
  ggsave(paste0(output_stub, ".pdf"), figure, width = width, height = height,
         units = "in")
}


# =============================================================================
# Figure 1: performance distributions
# =============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(tidytext)
library(hrbrthemes)
library(dplyr)
library(tidyr)
library(ggplot2)
library(hrbrthemes)
library(ggh4x)

# ============================================================
# Participant colors
# ============================================================
participant_colors <- c(
  "#F04E37",  # strong Aperol orange
  "#FF9A6E",  # peach coral
  "#D97A2B",  # burnt orange
  "#EED9A0",  # pale sand
  "#C9A45A",  # deeper sandy gold
  "#89D0EA",  # light sky blue
  "#4C9BC8",  # coastal blue
  "#2D6D98"   # deep ocean blue
)


# ============================================================
# Identify baseline models
# ============================================================

model_lookup <- data.frame(
  model = names(ALL_MODEL_LABELS),
  model_label = unname(ALL_MODEL_LABELS),
  stringsAsFactors = FALSE
) %>%
  mutate(
    search_text = tolower(
      paste(model, model_label)
    )
  )


train_candidates <- model_lookup %>%
  filter(
    grepl("train", search_text),
    grepl("mean", search_text)
  )


last_candidates <- model_lookup %>%
  filter(
    grepl("last", search_text),
    grepl("observ", search_text)
  )


if (nrow(train_candidates) == 0) {
  stop("Could not identify Train Mean model.")
}

if (nrow(last_candidates) == 0) {
  stop("Could not identify Last Observation model.")
}


TRAIN_MEAN_MODEL <- train_candidates$model[1]
LAST_MODEL <- last_candidates$model[1]


cat(
  "Train Mean model:", TRAIN_MEAN_MODEL, "\n",
  "Last Observation model:", LAST_MODEL, "\n"
)


# ============================================================
# Read raw results
# ============================================================

raw_results <- read.csv(
  SUMMARY_PATH,
  stringsAsFactors = FALSE
) %>%
  filter(
    model %in% names(ALL_MODEL_LABELS),
    !participant_id %in% c(11, 92)
  ) %>%
  mutate(
    model_label = unname(
      ALL_MODEL_LABELS[model]
    ),
    
    symptom = factor(
      recode(
        tolower(target),
        negative = "Negative",
        cognitive = "Cognitive",
        positive = "Positive"
      ),
      levels = c(
        "Negative",
        "Cognitive",
        "Positive"
      )
    )
  )


# ============================================================
# Figure 1: participant-level heterogeneity summary
#
# Each participant's correlation is averaged over the three
# symptom domains, then summarized across participants. This
# produces the values reported for Model 1-3 heterogeneity.
# ============================================================

PRIMARY_MODEL_IDS <- c(
  "Model1_Frozen_Combined_no_retrain_single",
  "Model2_LoRA_Combined_no_retrain_single",
  "Model3_reg_quantile_no_retrain_single"
)

participant_primary_performance <- raw_results %>%
  filter(
    #model %in% PRIMARY_MODEL_IDS,
    is.finite(corr)
  ) %>%
  group_by(
    participant_id,
    model,
    model_label
  ) %>%
  summarise(
    mean_held_out_correlation = mean(corr),
    symptom_domains = n_distinct(symptom),
    .groups = "drop"
  )

participant_heterogeneity_summary <- participant_primary_performance %>%
  group_by(model_label) %>%
  summarise(
    participants = n(),
    median_correlation = median(mean_held_out_correlation),
    min_correlation = min(mean_held_out_correlation),
    max_correlation = max(mean_held_out_correlation),
    .groups = "drop"
  ) %>%
  arrange(desc(median_correlation))

best_primary_model_by_participant <- participant_primary_performance %>%
  group_by(participant_id) %>%
  slice_max(
    mean_held_out_correlation,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup()

best_primary_model_counts <- best_primary_model_by_participant %>%
  count(model_label, name = "participants") %>%
  mutate(
    proportion = participants / sum(participants)
  ) %>%
  arrange(desc(participants), model_label)

# Overall participant-level performance averages each participant's held-out
# correlations across the three symptom domains and three primary models.
participant_overall_primary_performance <- participant_primary_performance %>%
  group_by(participant_id) %>%
  summarise(
    mean_correlation = mean(mean_held_out_correlation),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_correlation))

participant_overall_primary_summary <- participant_overall_primary_performance %>%
  summarise(
    participants = n(),
    median_correlation = median(mean_correlation),
    iqr_low = quantile(mean_correlation, 0.25),
    iqr_high = quantile(mean_correlation, 0.75),
    min_correlation = min(mean_correlation),
    max_correlation = max(mean_correlation)
  )

# Count participant-symptom results with lower MAE than each baseline. These
# values support the MAE-improvement comparison in the Results section.
baseline_mae <- raw_results %>%
  filter(model %in% c(TRAIN_MEAN_MODEL, LAST_MODEL)) %>%
  select(participant_id, symptom, model, mae) %>%
  pivot_wider(names_from = model, values_from = mae)

primary_model_mae_improvement <- raw_results %>%
  filter(model %in% PRIMARY_MODEL_IDS) %>%
  left_join(
    baseline_mae,
    by = c("participant_id", "symptom"),
    suffix = c("", "_baseline")
  ) %>%
  group_by(model_label) %>%
  summarise(
    participant_symptom_results = n(),
    lower_mae_than_train_mean = sum(mae < .data[[TRAIN_MEAN_MODEL]]),
    lower_mae_than_last_observation = sum(mae < .data[[LAST_MODEL]]),
    proportion_lower_than_train_mean =
      lower_mae_than_train_mean / participant_symptom_results,
    proportion_lower_than_last_observation =
      lower_mae_than_last_observation / participant_symptom_results,
    .groups = "drop"
  ) %>%
  arrange(desc(proportion_lower_than_train_mean), model_label)

cat("\nFigure 1 participant-level heterogeneity summary\n")
print(participant_heterogeneity_summary)
cat("\nBest primary model by participant\n")
print(best_primary_model_counts)
cat("\nOverall participant-level performance across primary models\n")
print(participant_overall_primary_performance)
print(participant_overall_primary_summary)
cat("\nPrimary-model MAE improvement relative to baselines\n")
print(primary_model_mae_improvement)


# ============================================================
# Assign participant colors
# ============================================================

participant_ids <- sort(
  unique(raw_results$participant_id)
)

participant_colors <- setNames(
  participant_colors[
    seq_along(participant_ids)
  ],
  participant_ids
)

# Preserve Figure 1's participant-to-color mapping, using Figure 2's padded IDs.
figure1_participant_colors <- setNames(
  participant_colors,
  sprintf("%05d", as.integer(names(participant_colors)))
)


# ============================================================
# Baseline MAE lookup
# ============================================================

train_mean_lookup <- raw_results %>%
  filter(
    model == TRAIN_MEAN_MODEL
  ) %>%
  select(
    participant_id,
    target,
    train_mean_mae = mae
  )


last_lookup <- raw_results %>%
  filter(
    model == LAST_MODEL
  ) %>%
  select(
    participant_id,
    target,
    last_mae = mae
  )


# ============================================================
# Raw MAE improvement
#
# Positive = lower MAE than baseline
# Negative = higher MAE than baseline
# ============================================================

comparison_wide <- raw_results %>%
  left_join(
    train_mean_lookup,
    by = c(
      "participant_id",
      "target"
    )
  ) %>%
  left_join(
    last_lookup,
    by = c(
      "participant_id",
      "target"
    )
  ) %>%
  mutate(
    
    mae_gain_train =
      train_mean_mae - mae,
    
    mae_gain_last =
      last_mae - mae
  )


# ============================================================
# Long format
# ============================================================

comparison_results <- comparison_wide %>%
  pivot_longer(
    cols = c(
      corr,
      mae_gain_train,
      mae_gain_last
    ),
    names_to = "metric_raw",
    values_to = "value"
  ) %>%
  mutate(
    metric = factor(
      metric_raw,
      levels = c(
        "corr",
        "mae_gain_train",
        "mae_gain_last"
      ),
      labels = c(
        "Pearson correlation (r)",
        "MAE improvement over Train Mean",
        "MAE improvement over Last Observation"
      )
    )
  )


# ============================================================
# Model ordering
#
# Based ONLY on median Pearson correlation.
# Same order used for all three rows.
# ============================================================

median_corr_order <- comparison_results %>%
  filter(
    metric_raw == "corr",
    is.finite(value)
  ) %>%
  group_by(
    symptom,
    model_label
  ) %>%
  summarise(
    median_corr = median(
      value,
      na.rm = TRUE
    ),
    .groups = "drop"
  )


model_order <- comparison_results %>%
  distinct(
    symptom,
    model_label
  ) %>%
  left_join(
    median_corr_order,
    by = c(
      "symptom",
      "model_label"
    )
  ) %>%
  arrange(
    symptom,
    is.na(median_corr),
    desc(median_corr)
  ) %>%
  mutate(
    model_key = paste(
      model_label,
      symptom,
      sep = "___"
    )
  )


comparison_results_plot <- comparison_results %>%
  mutate(
    model_key = paste(
      model_label,
      symptom,
      sep = "___"
    ),
    
    model_plot = factor(
      model_key,
      levels = model_order$model_key
    )
  )


# ============================================================
# Data actually shown as dots
#
# Leave baseline-vs-itself comparisons blank
# ============================================================

plot_points <- comparison_results_plot %>%
  filter(
    is.finite(value),
    
    !(
      metric_raw == "mae_gain_train" &
        model == TRAIN_MEAN_MODEL
    ),
    
    !(
      metric_raw == "mae_gain_last" &
        model == LAST_MODEL
    )
  )


# ============================================================
# Median summary markers
# ============================================================

summary_results <- plot_points %>%
  group_by(
    metric,
    symptom,
    model_label,
    model_plot
  ) %>%
  summarise(
    median_value = median(
      value,
      na.rm = TRUE
    ),
    .groups = "drop"
  )


# ============================================================
# Zero reference lines
# ============================================================

zero_lines <- data.frame(
  metric = factor(
    c(
      "Pearson correlation (r)",
      "MAE improvement over Train Mean",
      "MAE improvement over Last Observation"
    ),
    levels = levels(
      comparison_results_plot$metric
    )
  )
)


# ============================================================
# Plot
# ============================================================

figure1 <- ggplot() +
  
  # Zero reference:
  # correlation = no association
  # MAE improvement = no difference from baseline
  geom_hline(
    data = zero_lines,
    aes(
      yintercept = 0
    ),
    linetype = "dashed",
    linewidth = 0.55,
    color = "grey50"
  ) +
  
  # Participant-level dots
  geom_point(
    data = plot_points,
    aes(
      x = model_plot,
      y = value,
      color = factor(participant_id)
    ),
    position = position_jitter(
      width = 0.01,
      height = 0,
      seed = 123
    ),
    size = 2.8,
    alpha = 0.9
  ) +
  
  # Median across participants
  geom_point(
    data = summary_results,
    aes(
      x = model_plot,
      y = median_value
    ),
    shape = 95,
    size = 7,
    color = "black"
  ) +
  
  # Rows = metric
  # Columns = symptom
  # Independent y scales across panels
  ggh4x::facet_grid2(
    rows = vars(metric),
    cols = vars(symptom),
    scales = "free",
    independent = "y",
    axes = "y",
    switch = "y"
  ) +
  
  # Remove symptom suffix from displayed model names
  scale_x_discrete(
    labels = function(x) {
      sub(
        "___.*$",
        "",
        x
      )
    },
    drop = TRUE
  ) +
  
  scale_color_manual(
    values = participant_colors,
    drop = FALSE
  ) +
  
  scale_y_continuous(
    expand = expansion(
      mult = c(
        0.08,
        0.13
      )
    )
  ) +
  
  labs(
 #   title = "Participant-level held-out model performance",
    x = NULL,
    y = NULL,
    color = "Participant"
  ) +
  
  theme_ipsum(
    base_size = 12,
    grid = "Y"
  ) +
  
  theme(
    legend.position = "right",
    
    plot.title = element_text(
      face = "bold",
      size = 18
    ),
    
    strip.text.x = element_text(
      face = "bold",
      size = 11
    ),
    
    strip.text.y = element_text(
      face = "bold",
      size = 10
    ),
    
    axis.text.x = element_text(
      angle = 38,
      hjust = 1,
      vjust = 1
    ),
    
    panel.grid.minor = element_blank()
  ) +
  theme(
    strip.placement = "outside",
    strip.background.y = element_blank(),
    strip.text.y.left = element_text(
      face = "bold",
      size = 10
    )
  )


figure1
ggsave("/Users/f007qrc/Library/CloudStorage/GoogleDrive-anna.m.langener@dartmouth.edu/My Drive/Darmouth Drive/6_PersonalizedSensingSchizophrenia/Results1.png", 
       figure1, width = 10.5, height = 10.2,
       units = "in", dpi = 300)







# =============================================================================
# Figure 2: four selected participant trajectories
# =============================================================================

has_complete_predictions <- function(participant_id) {
  all(vapply(unname(PLOT_MODEL_DIRS), function(model_dir) {
    all(file.exists(file.path(
      RESULTS_ROOT,
      model_dir,
      paste0("results_", participant_id, "_", OUTCOMES, ".pth")
    )))
  }, logical(1)))
}

py_run_string("
import numpy as np
import torch

def load_test_predictions(path):
    try:
        folds = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        folds = torch.load(path, map_location='cpu')
    rows = []
    for fold_index, fold in enumerate(folds):
        y_true = np.asarray(fold.get('y_test', [])).reshape(-1)
        y_pred = np.asarray(fold.get('y_pred', [])).reshape(-1)
        if len(y_true) != len(y_pred):
            continue
        outer_i = int(fold.get('outer_i', fold_index))
        for point_i, (truth, prediction) in enumerate(zip(y_true, y_pred)):
            rows.append({
                'outer_i': outer_i,
                'point_i': int(point_i),
                'y_true': float(truth),
                'y_pred': float(prediction)
            })
    return rows
")

load_predictions <- function(participant_id, outcome, model_label, model_dir) {
  path <- file.path(
    RESULTS_ROOT,
    model_dir,
    paste0("results_", participant_id, "_", outcome, ".pth")
  )
  rows <- py_to_r(py$load_test_predictions(path))
  if (!length(rows)) stop("No held-out predictions in: ", path)

  bind_rows(rows) %>%
    arrange(outer_i, point_i) %>%
    mutate(
      participant_id = participant_id,
      outcome = outcome,
      model = model_label,
      observation = row_number()
    )
}

participant_scores <- read.csv(SUMMARY_PATH, stringsAsFactors = FALSE) %>%
  filter(model == SELECTION_MODEL, is.finite(corr)) %>%
  mutate(
    participant_id = sprintf("%05d", as.integer(participant_id)),
    target = tolower(target)
  ) %>%
  filter(target %in% OUTCOMES) %>%
  group_by(participant_id) %>%
  summarise(
    symptoms_available = n_distinct(target),
    mean_held_out_correlation = mean(corr),
    .groups = "drop"
  ) %>%
  filter(
    symptoms_available == length(OUTCOMES),
    !participant_id %in% EXCLUDED_SELECTION_PARTICIPANTS
  ) %>%
  rowwise() %>%
  filter(has_complete_predictions(participant_id)) %>%
  ungroup() %>%
  arrange(mean_held_out_correlation, participant_id)

if (nrow(participant_scores) < 4) {
  stop("At least four participants with complete predictions for all three models are required.")
}

selected_positions <- unique(round(seq(1, nrow(participant_scores), length.out = 4)))
selected_participants <- participant_scores %>%
  slice(selected_positions) %>%
  mutate(selection_band = c("Lowest", "Lower-middle", "Upper-middle", "Highest")) %>%
  arrange(desc(mean_held_out_correlation)) %>%
  mutate(
    selection = factor(
      paste0(
        selection_band,
        " mean held-out r\nParticipant ", participant_id,
        " (mean r = ", sprintf("%.2f", mean_held_out_correlation), ")"
      ),
      levels = paste0(
        selection_band,
        " mean held-out r\nParticipant ", participant_id,
        " (mean r = ", sprintf("%.2f", mean_held_out_correlation), ")"
      )
    )
  )

prediction_data <- bind_rows(lapply(seq_len(nrow(selected_participants)), function(i) {
  participant_id <- selected_participants$participant_id[i]
  selection <- selected_participants$selection[i]
  bind_rows(lapply(OUTCOMES, function(outcome) {
    bind_rows(lapply(names(PLOT_MODEL_DIRS), function(model_label) {
      load_predictions(participant_id, outcome, model_label, PLOT_MODEL_DIRS[[model_label]])
    }))
  })) %>% mutate(selection = selection)
}))

alignment_check <- prediction_data %>%
  group_by(selection, participant_id, outcome, outer_i, point_i, observation) %>%
  summarise(
    model_count = n_distinct(model),
    truth_range = max(y_true) - min(y_true),
    .groups = "drop"
  )
if (any(alignment_check$model_count != length(PLOT_MODEL_DIRS) | alignment_check$truth_range > 1e-8)) {
  stop("The three model outputs do not have identical held-out observations.")
}

panel_correlations <- prediction_data %>%
  group_by(selection, outcome, model) %>%
  summarise(panel_r = safe_cor(y_true, y_pred), .groups = "drop") %>%
  mutate(model = factor(model, levels = names(PLOT_MODEL_DIRS))) %>%
  arrange(selection, outcome, model) %>%
  group_by(selection, outcome) %>%
  summarise(
    annotation = paste0(as.character(model), ": r = ", sprintf("%.2f", panel_r), collapse = "\n"),
    .groups = "drop"
  ) %>%
  mutate(outcome_label = factor(outcome_label(outcome), levels = c("Negative", "Cognitive", "Positive")))

true_data <- prediction_data %>%
  distinct(selection, participant_id, outcome, outer_i, point_i, observation, y_true) %>%
  transmute(selection, participant_id, outcome, observation, series = "True", score = y_true)
prediction_lines <- prediction_data %>%
  transmute(selection, participant_id, outcome, observation, series = model, score = y_pred)
plot_data <- bind_rows(true_data, prediction_lines) %>%
  mutate(
    series = factor(series, levels = c("True", names(PLOT_MODEL_DIRS))),
    outcome_label = factor(outcome_label(outcome), levels = c("Negative", "Cognitive", "Positive")),
    line_key = paste(participant_id, series, sep = " | ")
  )

selected_ids <- unique(as.character(selected_participants$participant_id))
if (any(!selected_ids %in% names(figure1_participant_colors))) {
  stop("Every Figure 2 participant must have a Figure 1 color assignment.")
}

lighten_hex <- function(hex, amount = 0.45) {
  rgb <- grDevices::col2rgb(hex)
  grDevices::rgb(
    rgb[1, 1] + (255 - rgb[1, 1]) * amount,
    rgb[2, 1] + (255 - rgb[2, 1]) * amount,
    rgb[3, 1] + (255 - rgb[3, 1]) * amount,
    maxColorValue = 255
  )
}

# True scores use a lighter shade; predictions use the Figure 1 color.
trajectory_keys <- expand.grid(
  participant_id = selected_ids,
  series = c("True", names(PLOT_MODEL_DIRS)),
  stringsAsFactors = FALSE
) %>%
  mutate(
    base_color = unname(figure1_participant_colors[participant_id]),
    color = if_else(series == "True", vapply(base_color, lighten_hex, character(1)), base_color),
    line_key = paste(participant_id, series, sep = " | ")
  )
trajectory_colors <- setNames(trajectory_keys$color, trajectory_keys$line_key)

figure2 <- ggplot(plot_data, aes(
  x = observation,
  y = score,
  color = line_key,
  group = line_key
)) +
  geom_line(data = filter(plot_data, series == "True"), linewidth = 0.85, alpha = 0.95) +
  geom_line(data = filter(plot_data, series != "True"), linewidth = 0.85, alpha = 0.95) +
  geom_text(
    data = panel_correlations,
    aes(x = Inf, y = Inf, label = annotation),
    inherit.aes = FALSE,
    hjust = 1.1,
    vjust = 1.4,
    color = "black",
    fontface = "bold",
    size = 3.0
  ) +
  facet_grid(selection ~ outcome_label, scales = "free") +
  scale_color_manual(values = trajectory_colors, guide = "none") +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
   # title = "Held-out true versus predicted symptom scores",
    subtitle = "Darker lines = predicted values; lighter lines = observed values.",
    x = "Test observation",
    y = "Symptom score",
  ) +
  theme_ipsum(base_size = 12, base_family = "sans", grid = "Y") +
  theme(
    legend.position = "top",
    legend.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 10),
    strip.text.x = element_text(face = "bold", size = 12),
    strip.text.y = element_text(face = "bold", size = 9, angle = 0),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(hjust = 0)
  )


figure2


ggsave("/Users/f007qrc/Library/CloudStorage/GoogleDrive-anna.m.langener@dartmouth.edu/My Drive/Darmouth Drive/6_PersonalizedSensingSchizophrenia/Results2.png", 
       figure2, width = 10, height = 8,
       units = "in", dpi = 300)
