# --- Packages ---
suppressPackageStartupMessages({
  library(fs)
  library(lme4)
  library(lmerTest)
  library(parallel)
  library(robustlmm)
  library(readr)
  library(dplyr)
  library(rstudioapi)
})

DEFAULT_PROJECT <- "/Users/mojackhu/Research/pd"
DEFAULT_INPUT_CSV <- path(
  DEFAULT_PROJECT,
  "summary",
  "table",
  "cycle",
  "preprocessed",
  "cycle_trace_long.csv"
)
DEFAULT_OUTPUT_DIR <- path(
  DEFAULT_PROJECT,
  "summary",
  "table",
  "cycle",
  "interval"
)
DEFAULT_JOBS <- max(1L, detectCores(logical = TRUE))
ALPHA_LEVEL <- 0.05
LOCAL_METRICS <- c("aperiodic", "periodic", "raw_power")
CONNECTIVITY_METRICS <- c(
  "coherence",
  "wpli",
  "ciplv",
  "pli",
  "plv",
  "psi",
  "trgc"
)

FORMULA_LOCAL <- Value ~ Region + (1 | Subject)
FORMULA_CONNECTIVITY <- Value ~ 1 + (1 | Subject)

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_flag <- "--file="
  file_arg <- args[startsWith(args, file_flag)]
  if (length(file_arg) > 0) {
    return(path_abs(path_dir(sub(file_flag, "", file_arg[[1]]))))
  }

  if (rstudioapi::isAvailable()) {
    editor_path <- rstudioapi::getSourceEditorContext()$path
    if (nzchar(editor_path)) {
      return(path_abs(path_dir(editor_path)))
    }
  }

  stop("Cannot determine script directory for run_cycle_interval_fit.R")
}

parse_cli_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- list(
    input_csv = DEFAULT_INPUT_CSV,
    output_dir = DEFAULT_OUTPUT_DIR,
    jobs = DEFAULT_JOBS
  )

  if (length(args) == 0) {
    return(opts)
  }

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--input-csv", "--output-dir", "--jobs")) {
      if (i == length(args)) {
        stop("Missing value for ", key)
      }
      value <- args[[i + 1L]]
      if (key == "--input-csv") {
        opts$input_csv <- value
      } else if (key == "--jobs") {
        opts$jobs <- as.integer(value)
      } else {
        opts$output_dir <- value
      }
      i <- i + 2L
      next
    }

    if (key %in% c("-h", "--help")) {
      cat(
        paste(
          "Usage: Rscript run_cycle_interval_fit.R [--input-csv PATH] [--output-dir PATH] [--jobs N]",
          "",
          "Defaults:",
          paste("  --input-csv", DEFAULT_INPUT_CSV),
          paste("  --output-dir", DEFAULT_OUTPUT_DIR),
          paste("  --jobs", DEFAULT_JOBS),
          sep = "\n"
        )
      )
      quit(save = "no", status = 0)
    }

    stop("Unknown argument: ", key)
  }

  opts
}

write_output_table <- function(frame, csv_path) {
  dir_create(path_dir(csv_path), recurse = TRUE)
  write_csv(frame, csv_path)
}

safe_lmer_fit <- function(formula, data) {
  status <- "ok"
  model <- tryCatch(
    suppressWarnings(lmer(formula, data = data, REML = TRUE)),
    error = function(err) {
      status <<- paste0("error: ", conditionMessage(err))
      NULL
    }
  )
  list(model = model, status = status)
}

safe_rlmer_fit <- function(formula, data) {
  status <- "ok"
  model <- tryCatch(
    suppressWarnings(rlmer(formula, data = data, method = "DAStau")),
    error = function(err) {
      status <<- paste0("error: ", conditionMessage(err))
      NULL
    }
  )
  list(model = model, status = status)
}

direction_from_estimate <- function(x) {
  if (is.na(x) || x == 0) {
    return("none")
  }
  if (x > 0) {
    return("above_0")
  }
  "below_0"
}

extract_lmer_coef_row <- function(model, term) {
  coef_tab <- coef(summary(model))
  if (!(term %in% rownames(coef_tab))) {
    return(
      list(
        estimate = NA_real_,
        p_value = NA_real_
      )
    )
  }

  row <- coef_tab[term, , drop = FALSE]
  list(
    estimate = as.numeric(row[, "Estimate"]),
    p_value = if ("Pr(>|t|)" %in% colnames(row)) {
      as.numeric(row[, "Pr(>|t|)"])
    } else {
      NA_real_
    }
  )
}

extract_rlmer_coef_row <- function(model, term) {
  coef_tab <- coef(summary(model))
  if (!(term %in% rownames(coef_tab))) {
    return(
      list(
        estimate = NA_real_,
        p_value = NA_real_
      )
    )
  }

  row <- coef_tab[term, , drop = FALSE]
  t_value <- as.numeric(row[, "t value"])
  list(
    estimate = as.numeric(row[, "Estimate"]),
    p_value = 2 * pnorm(abs(t_value), lower.tail = FALSE)
  )
}

build_region_contrast <- function(model, region) {
  coef_names <- names(fixef(model))
  region_levels <- levels(model.frame(model)$Region)
  newdata <- data.frame(Region = factor(region, levels = region_levels))
  mm <- model.matrix(~ Region, data = newdata)
  contrast <- setNames(rep(0, length(coef_names)), coef_names)
  common_terms <- intersect(colnames(mm), coef_names)
  contrast[common_terms] <- as.numeric(mm[1, common_terms, drop = TRUE])
  contrast
}

extract_lmer_region_contrast <- function(model, region) {
  contrast <- build_region_contrast(model, region)
  test_row <- tryCatch(
    lmerTest::contest1D(model, contrast, rhs = 0),
    error = function(err) NULL
  )
  if (is.null(test_row)) {
    return(list(estimate = NA_real_, p_value = NA_real_))
  }

  test_df <- as.data.frame(test_row)
  list(
    estimate = as.numeric(test_df[["Estimate"]][[1]]),
    p_value = as.numeric(test_df[["Pr(>|t|)"]][[1]])
  )
}

extract_rlmer_region_contrast <- function(model, region) {
  contrast <- build_region_contrast(model, region)
  beta <- tryCatch(fixef(model), error = function(err) NULL)
  vcov_mat <- tryCatch(as.matrix(vcov(model)), error = function(err) NULL)
  if (is.null(beta) || is.null(vcov_mat)) {
    return(list(estimate = NA_real_, p_value = NA_real_))
  }

  common_terms <- intersect(names(beta), names(contrast))
  if (length(common_terms) == 0L) {
    return(list(estimate = NA_real_, p_value = NA_real_))
  }

  beta <- beta[common_terms]
  contrast <- contrast[common_terms]
  vcov_mat <- vcov_mat[common_terms, common_terms, drop = FALSE]

  estimate <- as.numeric(sum(contrast * beta))
  std_error <- tryCatch(
    sqrt(as.numeric(t(contrast) %*% vcov_mat %*% contrast)),
    error = function(err) NA_real_
  )
  if (is.na(std_error) || std_error <= 0) {
    return(list(estimate = estimate, p_value = NA_real_))
  }

  t_value <- estimate / std_error
  list(
    estimate = estimate,
    p_value = 2 * pnorm(abs(t_value), lower.tail = FALSE)
  )
}

build_result_row <- function(
  metric,
  band,
  region,
  t_pct,
  lmer_estimate,
  lmer_p_value,
  lmer_status,
  rlmer_estimate,
  rlmer_p_value,
  rlmer_status
) {
  lmer_direction <- direction_from_estimate(lmer_estimate)
  rlmer_direction <- direction_from_estimate(rlmer_estimate)
  lmer_sig <- !is.na(lmer_p_value) && (lmer_p_value < ALPHA_LEVEL)
  rlmer_sig <- !is.na(rlmer_p_value) && (rlmer_p_value < ALPHA_LEVEL)

  tibble::tibble(
    Metric = metric,
    Band = band,
    Region = region,
    t_pct = t_pct,
    lmer_estimate = lmer_estimate,
    lmer_p_value = lmer_p_value,
    lmer_status = lmer_status,
    rlmer_estimate = rlmer_estimate,
    rlmer_p_value = rlmer_p_value,
    rlmer_status = rlmer_status,
    lmer_direction = lmer_direction,
    rlmer_direction = rlmer_direction,
    lmer_sig = lmer_sig,
    rlmer_sig = rlmer_sig,
    both_sig_same_direction = lmer_sig &&
      rlmer_sig &&
      lmer_direction != "none" &&
      identical(lmer_direction, rlmer_direction)
  )
}

run_group_jobs <- function(groups, fit_fun, label, jobs) {
  if (length(groups) == 0L) {
    return(list())
  }

  jobs <- max(1L, as.integer(jobs))
  message(
    "Running ", label, " pointwise models with ",
    jobs, " worker(s) across ", length(groups), " grouped slices"
  )

  if (.Platform$OS.type == "windows" || jobs == 1L) {
    return(lapply(groups, fit_fun))
  }

  mclapply(
    groups,
    fit_fun,
    mc.cores = jobs,
    mc.preschedule = TRUE
  )
}

fit_local_timepoint <- function(data) {
  region_levels <- intersect(c("SNr", "STN"), unique(as.character(data$Region)))
  if (length(region_levels) == 0) {
    region_levels <- sort(unique(as.character(data$Region)))
  }

  data <- data %>%
    mutate(Region = factor(as.character(Region), levels = region_levels))

  lmer_fit <- safe_lmer_fit(FORMULA_LOCAL, data)
  rlmer_fit <- safe_rlmer_fit(FORMULA_LOCAL, data)

  metric <- data$Metric[[1]]
  band <- data$Band[[1]]
  t_pct <- data$t_pct[[1]]

  bind_rows(lapply(region_levels, function(region) {
    lmer_row <- if (!is.null(lmer_fit$model)) {
      extract_lmer_region_contrast(lmer_fit$model, region)
    } else {
      list(estimate = NA_real_, p_value = NA_real_)
    }
    rlmer_row <- if (!is.null(rlmer_fit$model)) {
      extract_rlmer_region_contrast(rlmer_fit$model, region)
    } else {
      list(estimate = NA_real_, p_value = NA_real_)
    }

    build_result_row(
      metric = metric,
      band = band,
      region = region,
      t_pct = t_pct,
      lmer_estimate = lmer_row$estimate,
      lmer_p_value = lmer_row$p_value,
      lmer_status = lmer_fit$status,
      rlmer_estimate = rlmer_row$estimate,
      rlmer_p_value = rlmer_row$p_value,
      rlmer_status = rlmer_fit$status
    )
  }))
}

fit_connectivity_timepoint <- function(data) {
  regions <- unique(as.character(data$Region))
  if (length(regions) != 1L) {
    stop(
      "Connectivity timepoint must contain exactly one Region, got: ",
      paste(regions, collapse = ", ")
    )
  }

  lmer_fit <- safe_lmer_fit(FORMULA_CONNECTIVITY, data)
  rlmer_fit <- safe_rlmer_fit(FORMULA_CONNECTIVITY, data)

  lmer_row <- if (!is.null(lmer_fit$model)) {
    extract_lmer_coef_row(lmer_fit$model, "(Intercept)")
  } else {
    list(estimate = NA_real_, p_value = NA_real_)
  }
  rlmer_row <- if (!is.null(rlmer_fit$model)) {
    extract_rlmer_coef_row(rlmer_fit$model, "(Intercept)")
  } else {
    list(estimate = NA_real_, p_value = NA_real_)
  }

  build_result_row(
    metric = data$Metric[[1]],
    band = data$Band[[1]],
    region = regions[[1]],
    t_pct = data$t_pct[[1]],
    lmer_estimate = lmer_row$estimate,
    lmer_p_value = lmer_row$p_value,
    lmer_status = lmer_fit$status,
    rlmer_estimate = rlmer_row$estimate,
    rlmer_p_value = rlmer_row$p_value,
    rlmer_status = rlmer_fit$status
  )
}

main <- function() {
  options <- parse_cli_args()
  input_csv <- path_abs(options$input_csv)
  output_dir <- path_abs(options$output_dir)
  jobs <- max(1L, as.integer(options$jobs))

  if (!file_exists(input_csv)) {
    stop("Input CSV not found: ", input_csv)
  }
  dir_create(output_dir, recurse = TRUE)

  long_df <- read_csv(
    input_csv,
    show_col_types = FALSE,
    col_types = cols(
      Subject = col_character(),
      Channel = col_character(),
      Metric = col_character(),
      Band = col_character(),
      Region = col_character(),
      t_pct = col_double(),
      Value = col_double()
    )
  ) %>%
    filter(!is.na(Value)) %>%
    arrange(Metric, Band, Region, t_pct, Subject, Channel)

  required_cols <- c(
    "Subject",
    "Channel",
    "Metric",
    "Band",
    "Region",
    "t_pct",
    "Value"
  )
  missing_cols <- setdiff(required_cols, colnames(long_df))
  if (length(missing_cols) > 0) {
    stop("Missing required input columns: ", paste(missing_cols, collapse = ", "))
  }

  known_metrics <- c(LOCAL_METRICS, CONNECTIVITY_METRICS)
  unknown_metrics <- setdiff(sort(unique(long_df$Metric)), known_metrics)
  if (length(unknown_metrics) > 0) {
    stop("Unknown cycle metrics in input CSV: ", paste(unknown_metrics, collapse = ", "))
  }

  result_rows <- vector("list", length = 0L)

  local_groups <- long_df %>%
    filter(Metric %in% LOCAL_METRICS) %>%
    group_by(Metric, Band, t_pct) %>%
    group_split(.keep = TRUE)

  local_results <- run_group_jobs(local_groups, fit_local_timepoint, "local", jobs)
  if (length(local_results) > 0L) {
    result_rows <- c(result_rows, local_results)
  }

  connectivity_groups <- long_df %>%
    filter(Metric %in% CONNECTIVITY_METRICS) %>%
    group_by(Metric, Band, Region, t_pct) %>%
    group_split(.keep = TRUE)

  connectivity_results <- run_group_jobs(
    connectivity_groups,
    fit_connectivity_timepoint,
    "connectivity",
    jobs
  )
  if (length(connectivity_results) > 0L) {
    result_rows <- c(result_rows, connectivity_results)
  }

  result_df <- if (length(result_rows) == 0L) {
    tibble::tibble(
      Metric = character(),
      Band = character(),
      Region = character(),
      t_pct = numeric(),
      lmer_estimate = numeric(),
      lmer_p_value = numeric(),
      lmer_status = character(),
      rlmer_estimate = numeric(),
      rlmer_p_value = numeric(),
      rlmer_status = character(),
      lmer_direction = character(),
      rlmer_direction = character(),
      lmer_sig = logical(),
      rlmer_sig = logical(),
      both_sig_same_direction = logical()
    )
  } else {
    bind_rows(result_rows) %>%
      arrange(Metric, Band, Region, t_pct)
  }

  write_output_table(
    result_df,
    path(output_dir, "cycle_timepoint_deviation.csv")
  )
}

SCRIPT_DIR <- get_script_dir()
main()
