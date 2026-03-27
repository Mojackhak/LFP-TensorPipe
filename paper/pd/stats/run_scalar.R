# --- Packages ---
suppressPackageStartupMessages({
  library(fs)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(parallel)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(broom.mixed)
  library(readr)
  library(stringr)
  library(effects)
  library(sjPlot)
  library(ggeffects)
  library(ggpubr)
  library(ggplot2)
  library(ggbreak)
  library(rlang)
  library(robustlmm)
  library(rstudioapi)
  library(Cairo)
  library(performance)
  library(DHARMa)
  library(influence.ME)
  library(splines)
})

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

  stop("Cannot determine script directory for run_scalar.R")
}

SCRIPT_DIR <- get_script_dir()
source(path(SCRIPT_DIR, "emm.R"))

emm_options(pbkrtest.limit = 1e6, lmerTest.limit = 1e6)
emm_options(lmer.df = "kenward-roger")

detect_default_jobs <- function() {
  jobs <- parallel::detectCores(logical = TRUE)
  if (is.na(jobs) || jobs < 1L) {
    return(1L)
  }
  as.integer(jobs)
}

parse_cli_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- list(jobs = detect_default_jobs())

  if (length(args) == 0L) {
    return(opts)
  }

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--jobs") {
      if (i == length(args)) {
        stop("Missing value for --jobs")
      }
      opts$jobs <- args[[i + 1L]]
      i <- i + 2L
      next
    }

    if (key %in% c("-h", "--help")) {
      default_jobs <- detect_default_jobs()
      cat(
        paste(
          "Usage: Rscript run_scalar.R [--jobs N]",
          "",
          "Defaults:",
          paste("  --jobs", default_jobs),
          sep = "\n"
        )
      )
      quit(save = "no", status = 0)
    }

    stop("Unknown argument: ", key)
  }

  opts
}

normalize_jobs <- function(jobs) {
  jobs_int <- suppressWarnings(as.integer(jobs))
  if (is.na(jobs_int) || jobs_int < 1L) {
    stop("--jobs must be an integer greater than or equal to 1")
  }
  jobs_int
}

project <- "/Users/mojackhu/Research/pd"
stats_dir <- path(project, "summary", "table")

SINGLE_METRICS <- c("raw_power", "periodic", "aperiodic", "burst")
DOUBLE_METRICS <- c("coherence", "ciplv", "pli", "plv", "psi", "trgc", "wpli")

MODEL_SPECS <- list(
  lmer_model = list(
    fn = lmer,
    args = list(REML = TRUE),
    diagnostic_plots = 1L
  ),
  rlmer_model = list(
    fn = rlmer,
    args = list(method = "DAStau"),
    diagnostic_plots = 3L
  )
)

ANALYSIS_SPECS <- list(
  cycle_single = list(
    section = "cycle",
    metric_dirs = SINGLE_METRICS,
    formula = "Value ~ Phase*Lat*Region + (1|Subject)",
    emm = list(
      kind = "triple",
      base_name = "Phase-Lat-Region",
      x_var = "Phase",
      panel_var = "Lat",
      facet_var = "Region"
    )
  ),
  cycle_double = list(
    section = "cycle",
    metric_dirs = DOUBLE_METRICS,
    formula = "Value ~ Phase*Lat + (1|Subject)",
    emm = list(
      kind = "double",
      base_name = "Phase-Lat",
      x_var = "Phase",
      panel_var = "Lat"
    )
  ),
  med_single = list(
    section = "med",
    metric_dirs = SINGLE_METRICS,
    formula = "Value ~ Phase*Region + (1|Subject)",
    emm = list(
      kind = "double",
      base_name = "Phase-Region",
      x_var = "Phase",
      panel_var = "Region"
    )
  ),
  med_double = list(
    section = "med",
    metric_dirs = DOUBLE_METRICS,
    formula = "Value ~ Phase + (1|Subject)",
    emm = list(
      kind = "single",
      base_name = "Phase",
      x_var = "Phase"
    )
  ),
  motor_single = list(
    section = "motor",
    metric_dirs = SINGLE_METRICS,
    formula = "Value ~ Phase*Region + (1|Subject)",
    emm = list(
      kind = "double",
      base_name = "Phase-Region",
      x_var = "Phase",
      panel_var = "Region"
    )
  ),
  motor_double = list(
    section = "motor",
    metric_dirs = DOUBLE_METRICS,
    formula = "Value ~ Phase + (1|Subject)",
    emm = list(
      kind = "single",
      base_name = "Phase",
      x_var = "Phase"
    )
  ),
  turn_single = list(
    section = "turn",
    metric_dirs = SINGLE_METRICS,
    formula = "Value ~ Phase*Region + (1|Subject)",
    emm = list(
      kind = "double",
      base_name = "Phase-Region",
      x_var = "Phase",
      panel_var = "Region"
    )
  ),
  turn_double = list(
    section = "turn",
    metric_dirs = DOUBLE_METRICS,
    formula = "Value ~ Phase + (1|Subject)",
    emm = list(
      kind = "single",
      base_name = "Phase",
      x_var = "Phase"
    )
  )
)

SELECTED_ANALYSES <- names(ANALYSIS_SPECS)

collect_workbook_paths <- function(section, metric_dirs) {
  section_dir <- path(stats_dir, section)
  wk_dirs <- path(section_dir, metric_dirs)
  wk_dirs <- wk_dirs[file_exists(wk_dirs)]
  if (length(wk_dirs) == 0) {
    return(character())
  }
  sort(dir_ls(
    wk_dirs,
    recurse = FALSE,
    type = "file",
    glob = "*scalar_summary_trans.xlsx"
  ))
}

load_scalar_table <- function(df_path) {
  df <- readxl::read_xlsx(df_path)
  value_raw <- df$Value
  df <- df %>%
    mutate(Value = readr::parse_double(Value))

  if (any(is.na(df$Value) & !is.na(value_raw))) {
    stop(paste("Failed to parse numeric Value from", df_path))
  }

  df
}

export_model_diagnostics <- function(model, save_dir, n_plots) {
  tab <- joint_tests(model)
  write_csv(tab, path(save_dir, "model_omnibus.csv"))

  cairo_pdf(path(save_dir, "model_evaluation.pdf"), width = 16, height = 16)
  print(performance::check_model(model))
  dev.off()

  base_name <- "model_evaluation"
  for (i in seq_len(n_plots)) {
    cairo_pdf(
      path(save_dir, paste0(base_name, as.character(i), ".pdf")),
      width = 8,
      height = 6
    )
    print(plot(model, which = i))
    dev.off()
  }
}

run_emm_export <- function(model, data, emm_spec, save_dir) {
  out <- switch(
    emm_spec$kind,
    single = EMM_single_effect(
      m = model,
      data = data,
      x_var = emm_spec$x_var
    ),
    double = EMM_double_interaction(
      m = model,
      data = data,
      x_var = emm_spec$x_var,
      panel_var = emm_spec$panel_var
    ),
    triple = EMM_triple_interaction(
      m = model,
      data = data,
      x_var = emm_spec$x_var,
      panel_var = emm_spec$panel_var,
      facet_var = emm_spec$facet_var
    ),
    stop("Unknown EMM kind: ", emm_spec$kind)
  )

  write_csv(out$emm, path(save_dir, paste0(emm_spec$base_name, "_emm.csv")))
  write_csv(out$tuk, path(save_dir, paste0(emm_spec$base_name, "_tukey.csv")))
}

build_workbook_tasks <- function(selected_analyses = SELECTED_ANALYSES) {
  tasks <- list()
  idx <- 1L

  for (spec_name in selected_analyses) {
    spec <- ANALYSIS_SPECS[[spec_name]]
    df_paths <- collect_workbook_paths(spec$section, spec$metric_dirs)
    if (length(df_paths) == 0L) {
      next
    }

    for (df_path in df_paths) {
      tasks[[idx]] <- list(
        spec_name = spec_name,
        spec = spec,
        df_path = as.character(path_abs(df_path))
      )
      idx <- idx + 1L
    }
  }

  tasks
}

run_workbook_task <- function(task) {
  spec_name <- task$spec_name
  spec <- task$spec
  df_path <- task$df_path
  message("Running analysis spec ", spec_name, " on workbook ", path_file(df_path))

  wk_dir <- path_dir(df_path)
  df_name <- path_ext_remove(path_file(df_path))
  df_dir <- path(wk_dir, df_name)
  dir_create(df_dir, recurse = TRUE)

  df <- load_scalar_table(df_path)
  bands <- sort(unique(df$Band[!is.na(df$Band)]))

  for (bd in bands) {
    save_dir_bd <- path(df_dir, bd)
    dir_create(save_dir_bd, recurse = TRUE)

    d <- df %>% filter(Band == bd) %>% droplevels()
    openxlsx::write.xlsx(d, path(save_dir_bd, "input_band.xlsx"), overwrite = TRUE)

    if (nrow(d) <= 3) {
      next
    }

    for (nm in names(MODEL_SPECS)) {
      model_spec <- MODEL_SPECS[[nm]]
      save_dir_nm <- path(save_dir_bd, nm)
      dir_create(save_dir_nm, recurse = TRUE)

      model <- do.call(
        model_spec$fn,
        c(
          list(
            formula = stats::as.formula(spec$formula),
            data = d
          ),
          model_spec$args
        )
      )

      export_model_diagnostics(
        model = model,
        save_dir = save_dir_nm,
        n_plots = model_spec$diagnostic_plots
      )

      run_emm_export(
        model = model,
        data = d,
        emm_spec = spec$emm,
        save_dir = save_dir_nm
      )
    }
  }

  list(
    ok = TRUE,
    spec_name = spec_name,
    df_path = df_path
  )
}

run_workbook_task_safe <- function(task) {
  tryCatch(
    run_workbook_task(task),
    error = function(err) {
      list(
        ok = FALSE,
        spec_name = task$spec_name,
        df_path = task$df_path,
        error = paste(class(err)[[1]], conditionMessage(err), sep = ": ")
      )
    }
  )
}

run_workbook_tasks <- function(tasks, jobs) {
  if (length(tasks) == 0L) {
    return(list())
  }

  jobs <- min(normalize_jobs(jobs), length(tasks))
  message(
    "Running ", length(tasks), " scalar workbook task(s) with ",
    jobs, " worker(s)"
  )

  if (.Platform$OS.type == "windows" || jobs == 1L) {
    return(lapply(tasks, run_workbook_task_safe))
  }

  parallel::mclapply(
    tasks,
    run_workbook_task_safe,
    mc.cores = jobs,
    mc.preschedule = TRUE
  )
}

report_failures <- function(results) {
  failures <- Filter(function(item) !isTRUE(item$ok), results)
  if (length(failures) == 0L) {
    return(invisible(results))
  }

  failure_lines <- vapply(
    failures,
    function(item) {
      paste0(
        "[", item$spec_name, "] ",
        item$df_path,
        ": ",
        item$error
      )
    },
    character(1)
  )
  stop(
    paste(
      c("Scalar workbook tasks failed:", failure_lines),
      collapse = "\n"
    )
  )
}

main <- function() {
  opts <- parse_cli_args()
  tasks <- build_workbook_tasks()
  results <- run_workbook_tasks(tasks, opts$jobs)
  report_failures(results)
  invisible(results)
}

main()
