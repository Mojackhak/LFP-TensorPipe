# emm.R
#
# Purpose:
#   Helper functions to compute estimated marginal means (EMMs) and stratified post-hoc
#   pairwise comparisons (e.g., Tukey-adjusted contrasts) from models supported by the
#   {emmeans} package.
#
#   The functions return tidy data frames for EMMs and pairwise contrasts, optionally
#   including an additional p-value adjustment across strata (useful when interpreting
#   many stratified comparisons together).


# ============================
# Dependencies (checked lazily)
# ============================

.check_pkgs <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Missing required packages: ", paste(missing, collapse = ", "),
      ". Please install them before running.",
      call. = FALSE
    )
  }
}


# ============================
# Small helpers
# ============================

.is_in_model <- function(model, var_name) {
  var_name %in% all.vars(stats::formula(model))
}

.as_factor_with_levels <- function(x, levels_vec) {
  factor(as.character(x), levels = levels_vec)
}

.validate_col <- function(data, col_name) {
  if (!col_name %in% names(data)) {
    stop("Column not found in `data`: ", col_name, call. = FALSE)
  }
}

.validate_numeric_col <- function(data, col_name) {
  .validate_col(data, col_name)
  if (!is.numeric(data[[col_name]])) {
    stop("Column must be numeric in `data`: ", col_name, call. = FALSE)
  }
}

# Build a mapping between a display factor (e.g., phase levels) and numeric values
# (e.g., phase_c) used for slicing in emmeans.
.build_slice_map <- function(
    data,
    display_var,
    numeric_var,
    levels_vec,
    user_vals = NULL,
    stat_choice = c("median", "mean")
) {
  stat_choice <- match.arg(stat_choice)
  .validate_col(data, display_var)
  .validate_numeric_col(data, numeric_var)

  disp <- factor(data[[display_var]], levels = levels_vec)

  if (is.null(user_vals)) {
    fun <- if (stat_choice == "median") stats::median else mean
    vals <- vapply(
      levels_vec,
      function(lv) {
        x <- data[[numeric_var]][disp == lv]
        fun(x, na.rm = TRUE)
      },
      numeric(1)
    )
    names(vals) <- levels_vec
  } else {
    if (!is.null(names(user_vals)) && all(levels_vec %in% names(user_vals))) {
      vals <- as.numeric(user_vals[levels_vec])
      names(vals) <- levels_vec
    } else {
      if (length(user_vals) != length(levels_vec)) {
        stop(
          "Length of slice values must match number of levels, or provide a named vector ",
          "whose names cover all levels.",
          call. = FALSE
        )
      }
      vals <- as.numeric(user_vals)
      names(vals) <- levels_vec
    }
  }

  out <- data.frame(
    numeric_val = as.numeric(vals),
    display_val = factor(names(vals), levels = levels_vec)
  )
  names(out) <- c(numeric_var, display_var)
  out
}

# Safe join for numeric slice variables by rounding keys (avoids floating-point mismatches).
.join_slice_map <- function(df, map_df, numeric_var, display_var, digits = 8L) {
  if (is.null(map_df)) return(df)
  if (!numeric_var %in% names(df)) return(df)

  df[[numeric_var]] <- as.numeric(df[[numeric_var]])
  map_df[[numeric_var]] <- as.numeric(map_df[[numeric_var]])

  df$.k <- round(df[[numeric_var]], digits)
  map_df$.k <- round(map_df[[numeric_var]], digits)

  joined <- dplyr::left_join(
    df,
    map_df[, c(".k", display_var), drop = FALSE],
    by = ".k"
  )
  joined$.k <- NULL
  joined
}

# Normalize CI columns across model types.
.normalize_ci_cols <- function(df) {
  if ("asymp.LCL" %in% names(df)) {
    df <- dplyr::rename(df, lower.CL = asymp.LCL, upper.CL = asymp.UCL)
  } else if ("lower" %in% names(df)) {
    df <- dplyr::rename(df, lower.CL = lower, upper.CL = upper)
  }
  df
}

# Normalize emmean column across summary types.
.normalize_emmean_col <- function(df) {
  if ("response" %in% names(df) && !"emmean" %in% names(df)) {
    names(df)[names(df) == "response"] <- "emmean"
  }
  df
}

# Extract the last number from a string (used to parse contrast labels for numeric slices).
.extract_last_number <- function(s) {
  vapply(
    regmatches(s, gregexpr("[-+]?\\d*\\.?\\d+", s, perl = TRUE)),
    function(x) if (length(x)) as.numeric(tail(x, 1)) else NA_real_,
    numeric(1)
  )
}

# Split a contrast label like "A - B" into lhs/rhs (base R; avoids tidyr dependency).
.split_contrast <- function(df, contrast_col = "contrast", sep = " - ") {
  if (!contrast_col %in% names(df)) {
    stop("Contrast column not found: ", contrast_col, call. = FALSE)
  }
  parts <- strsplit(as.character(df[[contrast_col]]), sep, fixed = TRUE)
  df$lhs <- vapply(parts, function(x) if (length(x) >= 1) x[[1]] else NA_character_, character(1))
  df$rhs <- vapply(parts, function(x) if (length(x) >= 2) x[[2]] else NA_character_, character(1))
  df
}

# Add across-strata p-value adjustment (optional) as a new column (base R implementation).
.add_p_adjustment <- function(df, p_col, method = "none", by_cols = NULL, new_col = "p_across") {
  if (identical(method, "none")) {
    df[[new_col]] <- NA_real_
    return(df)
  }

  if (!p_col %in% names(df)) {
    stop("p-value column not found: ", p_col, call. = FALSE)
  }

  if (!method %in% p.adjust.methods) {
    stop(
      "Unknown p.adjust method: ", method,
      ". Valid methods: ", paste(p.adjust.methods, collapse = ", "),
      call. = FALSE
    )
  }

  p <- df[[p_col]]
  out <- rep(NA_real_, length(p))

  if (is.null(by_cols) || length(by_cols) == 0) {
    out <- stats::p.adjust(p, method = method)
    df[[new_col]] <- out
    return(df)
  }

  missing_cols <- setdiff(by_cols, names(df))
  if (length(missing_cols) > 0) {
    stop("p_adjust_by columns not in table: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  grp <- interaction(df[, by_cols, drop = FALSE], drop = TRUE, lex.order = TRUE)
  for (g in levels(grp)) {
    idx <- which(grp == g)
    out[idx] <- stats::p.adjust(p[idx], method = method)
  }
  df[[new_col]] <- out
  df
}


# ============================
# Core: EMMs + stratified pairwise contrasts
# ============================

#' Compute EMMs and Tukey pairwise contrasts for a single effect.
#'
#' If x_var is not in the model, you must provide x_slice_on (a numeric covariate in the model)
#' and optionally x_slice_values (numeric values per x level).
EMM_single_effect <- function(
    m,
    data,
    x_var = "phase",
    x_levels = NULL,
    at_vars = list(),
    weights = c("equal", "proportional"),
    conf_level = 0.95,
    x_slice_on = NULL,
    x_slice_values = NULL,
    x_slice_stat = c("median", "mean"),
    # across-strata adjustment (single effect => across all contrasts unless grouped)
    p_across_method = "none",
    p_across_on = c("p_raw", "p_tukey"),
    p_across_by = NULL,
    join_digits = 8L
) {
  .check_pkgs(c("emmeans", "dplyr"))

  weights <- match.arg(weights)
  x_slice_stat <- match.arg(x_slice_stat)
  p_across_on <- match.arg(p_across_on)

  if (is.null(x_levels)) x_levels <- levels(factor(data[[x_var]]))
  data[[x_var]] <- .as_factor_with_levels(data[[x_var]], x_levels)

  x_in_model <- .is_in_model(m, x_var)
  if (!x_in_model) {
    if (is.null(x_slice_on)) {
      stop(
        "`x_var` ('", x_var, "') is not in the model. Please provide `x_slice_on` ",
        "(the numeric covariate used in the model for slicing).",
        call. = FALSE
      )
    }
    .validate_numeric_col(data, x_slice_on)
    if (!.is_in_model(m, x_slice_on)) {
      stop("`x_slice_on` is not in the model: ", x_slice_on, call. = FALSE)
    }
  }

  x_map <- NULL
  if (!x_in_model) {
    x_map <- .build_slice_map(
      data = data,
      display_var = x_var,
      numeric_var = x_slice_on,
      levels_vec = x_levels,
      user_vals = x_slice_values,
      stat_choice = x_slice_stat
    )
    at_vars <- utils::modifyList(at_vars, setNames(list(unique(x_map[[x_slice_on]])), x_slice_on))
  }

  x_spec <- if (x_in_model) x_var else x_slice_on
  form <- stats::as.formula(paste("~", x_spec))

  emm_obj <- emmeans::emmeans(m, specs = form, at = at_vars, weights = weights, level = conf_level)
  emm_df <- summary(emm_obj, type = "response", level = conf_level) |>
    as.data.frame() |>
    .normalize_ci_cols() |>
    .normalize_emmean_col()

  if (!x_in_model) {
    emm_df <- .join_slice_map(emm_df, x_map, numeric_var = x_slice_on, display_var = x_var, digits = join_digits)
  }
  if (x_var %in% names(emm_df)) {
    emm_df[[x_var]] <- .as_factor_with_levels(emm_df[[x_var]], x_levels)
  }

  # Pairwise contrasts (raw + Tukey)
  pairs_obj <- pairs(emm_obj, adjust = "none")
  raw_df <- summary(pairs_obj, type = "response", infer = c(TRUE, TRUE), adjust = "none") |>
    as.data.frame()
  tuk_df <- summary(pairs_obj, type = "response", infer = c(TRUE, TRUE), adjust = "tukey") |>
    as.data.frame()

  if (!"p.value" %in% names(raw_df) || !"p.value" %in% names(tuk_df)) {
    stop("Expected 'p.value' in pairs() summaries.", call. = FALSE)
  }

  tuk_df <- dplyr::rename(tuk_df, p_tukey = p.value)
  raw_df <- dplyr::rename(raw_df, p_raw = p.value)

  tuk_df <- dplyr::left_join(
    tuk_df,
    raw_df[, c("contrast", "p_raw"), drop = FALSE],
    by = "contrast"
  )

  tuk_df <- tuk_df |>
    dplyr::mutate(
      contrast = gsub("/", "-", .data$contrast)
    )
  tuk_df <- .split_contrast(tuk_df, contrast_col = "contrast", sep = " - ")

  # group1/group2 mapping
  if (!x_in_model) {
    tuk_df$lhs_num <- .extract_last_number(tuk_df$lhs)
    tuk_df$rhs_num <- .extract_last_number(tuk_df$rhs)

    map_l <- x_map
    names(map_l)[names(map_l) == x_slice_on] <- "lhs_num"
    names(map_l)[names(map_l) == x_var] <- "group1"

    map_r <- x_map
    names(map_r)[names(map_r) == x_slice_on] <- "rhs_num"
    names(map_r)[names(map_r) == x_var] <- "group2"

    map_l$lhs_num <- round(map_l$lhs_num, join_digits)
    map_r$rhs_num <- round(map_r$rhs_num, join_digits)
    tuk_df$lhs_num <- round(tuk_df$lhs_num, join_digits)
    tuk_df$rhs_num <- round(tuk_df$rhs_num, join_digits)

    tuk_df <- dplyr::left_join(tuk_df, map_l[, c("lhs_num", "group1"), drop = FALSE], by = "lhs_num")
    tuk_df <- dplyr::left_join(tuk_df, map_r[, c("rhs_num", "group2"), drop = FALSE], by = "rhs_num")
  } else {
    tuk_df <- dplyr::mutate(tuk_df, group1 = .data$lhs, group2 = .data$rhs)
  }

  if ("group1" %in% names(tuk_df)) tuk_df$group1 <- .as_factor_with_levels(tuk_df$group1, x_levels)
  if ("group2" %in% names(tuk_df)) tuk_df$group2 <- .as_factor_with_levels(tuk_df$group2, x_levels)

  # Across-strata adjustment (here: across all contrasts unless grouped)
  p_base <- if (p_across_on == "p_raw") "p_raw" else "p_tukey"
  tuk_df <- .add_p_adjustment(
    df = tuk_df,
    p_col = p_base,
    method = p_across_method,
    by_cols = p_across_by,
    new_col = "p_across"
  )

  list(emm = emm_df, tuk = tuk_df)
}


#' Compute EMMs and Tukey contrasts for a 2-way layout (x within panels).
#'
#' If x_var or panel_var are not in the model, you must explicitly provide
#' x_slice_on / panel_slice_on (numeric covariates in the model).
EMM_double_interaction <- function(
    m,
    data,
    x_var = "phase",
    panel_var = "region",
    x_levels = NULL,
    panel_levels = NULL,
    at_vars = list(),
    weights = c("equal", "proportional"),
    conf_level = 0.95,
    x_slice_on = NULL,
    x_slice_values = NULL,
    x_slice_stat = c("median", "mean"),
    panel_slice_on = NULL,
    panel_slice_values = NULL,
    panel_slice_stat = c("median", "mean"),
    # across-strata adjustment (across all rows unless grouped)
    p_across_method = "none",
    p_across_on = c("p_raw", "p_tukey"),
    p_across_by = NULL,
    join_digits = 8L
) {
  .check_pkgs(c("emmeans", "dplyr"))

  weights <- match.arg(weights)
  x_slice_stat <- match.arg(x_slice_stat)
  panel_slice_stat <- match.arg(panel_slice_stat)
  p_across_on <- match.arg(p_across_on)

  if (is.null(x_levels)) x_levels <- levels(factor(data[[x_var]]))
  if (is.null(panel_levels)) panel_levels <- levels(factor(data[[panel_var]]))

  data[[x_var]] <- .as_factor_with_levels(data[[x_var]], x_levels)
  data[[panel_var]] <- .as_factor_with_levels(data[[panel_var]], panel_levels)

  x_in_model <- .is_in_model(m, x_var)
  panel_in_model <- .is_in_model(m, panel_var)

  if (!x_in_model) {
    if (is.null(x_slice_on)) {
      stop("`x_var` ('", x_var, "') is not in the model. Please provide `x_slice_on`.", call. = FALSE)
    }
    .validate_numeric_col(data, x_slice_on)
    if (!.is_in_model(m, x_slice_on)) stop("`x_slice_on` is not in the model: ", x_slice_on, call. = FALSE)
  }

  if (!panel_in_model) {
    if (is.null(panel_slice_on)) {
      stop("`panel_var` ('", panel_var, "') is not in the model. Please provide `panel_slice_on`.", call. = FALSE)
    }
    .validate_numeric_col(data, panel_slice_on)
    if (!.is_in_model(m, panel_slice_on)) stop("`panel_slice_on` is not in the model: ", panel_slice_on, call. = FALSE)
  }

  if (!x_in_model && !panel_in_model && identical(x_slice_on, panel_slice_on)) {
    stop(
      "x_var and panel_var map to the same numeric covariate ('", x_slice_on, "'). ",
      "Please provide distinct slice variables.",
      call. = FALSE
    )
  }

  x_map <- panel_map <- NULL
  if (!x_in_model) {
    x_map <- .build_slice_map(
      data = data,
      display_var = x_var,
      numeric_var = x_slice_on,
      levels_vec = x_levels,
      user_vals = x_slice_values,
      stat_choice = x_slice_stat
    )
    at_vars <- utils::modifyList(at_vars, setNames(list(unique(x_map[[x_slice_on]])), x_slice_on))
  }

  if (!panel_in_model) {
    panel_map <- .build_slice_map(
      data = data,
      display_var = panel_var,
      numeric_var = panel_slice_on,
      levels_vec = panel_levels,
      user_vals = panel_slice_values,
      stat_choice = panel_slice_stat
    )
    at_vars <- utils::modifyList(at_vars, setNames(list(unique(panel_map[[panel_slice_on]])), panel_slice_on))
  }

  x_spec <- if (x_in_model) x_var else x_slice_on
  panel_spec <- if (panel_in_model) panel_var else panel_slice_on
  form <- stats::as.formula(paste("~", x_spec, "|", panel_spec))

  emm_obj <- emmeans::emmeans(m, specs = form, at = at_vars, weights = weights, level = conf_level)
  emm_df <- summary(emm_obj, type = "response", level = conf_level) |>
    as.data.frame() |>
    .normalize_ci_cols() |>
    .normalize_emmean_col()

  if (!x_in_model) {
    emm_df <- .join_slice_map(emm_df, x_map, numeric_var = x_slice_on, display_var = x_var, digits = join_digits)
  }
  if (!panel_in_model) {
    emm_df <- .join_slice_map(emm_df, panel_map, numeric_var = panel_slice_on, display_var = panel_var, digits = join_digits)
  }

  if (x_var %in% names(emm_df)) emm_df[[x_var]] <- .as_factor_with_levels(emm_df[[x_var]], x_levels)
  if (panel_var %in% names(emm_df)) emm_df[[panel_var]] <- .as_factor_with_levels(emm_df[[panel_var]], panel_levels)

  # Pairwise contrasts within each panel
  pairs_obj <- pairs(emm_obj, adjust = "none")
  raw_df <- summary(pairs_obj, type = "response", infer = c(TRUE, TRUE), adjust = "none") |>
    as.data.frame()
  tuk_df <- summary(pairs_obj, type = "response", infer = c(TRUE, TRUE), adjust = "tukey") |>
    as.data.frame()

  if (!"p.value" %in% names(raw_df) || !"p.value" %in% names(tuk_df)) {
    stop("Expected 'p.value' in pairs() summaries.", call. = FALSE)
  }

  tuk_df <- dplyr::rename(tuk_df, p_tukey = p.value)
  raw_df <- dplyr::rename(raw_df, p_raw = p.value)

  key_cols <- c(panel_spec, "contrast")
  key_cols <- key_cols[key_cols %in% names(tuk_df)]

  tuk_df <- dplyr::left_join(
    tuk_df,
    raw_df[, unique(c(key_cols, "p_raw")), drop = FALSE],
    by = key_cols
  )

  tuk_df <- tuk_df |>
    dplyr::mutate(
      contrast = gsub("/", "-", .data$contrast)
    )
  tuk_df <- .split_contrast(tuk_df, contrast_col = "contrast", sep = " - ")

  # group1/group2 mapping
  if (!x_in_model) {
    tuk_df$lhs_num <- .extract_last_number(tuk_df$lhs)
    tuk_df$rhs_num <- .extract_last_number(tuk_df$rhs)

    map_l <- x_map
    names(map_l)[names(map_l) == x_slice_on] <- "lhs_num"
    names(map_l)[names(map_l) == x_var] <- "group1"

    map_r <- x_map
    names(map_r)[names(map_r) == x_slice_on] <- "rhs_num"
    names(map_r)[names(map_r) == x_var] <- "group2"

    map_l$lhs_num <- round(map_l$lhs_num, join_digits)
    map_r$rhs_num <- round(map_r$rhs_num, join_digits)
    tuk_df$lhs_num <- round(tuk_df$lhs_num, join_digits)
    tuk_df$rhs_num <- round(tuk_df$rhs_num, join_digits)

    tuk_df <- dplyr::left_join(tuk_df, map_l[, c("lhs_num", "group1"), drop = FALSE], by = "lhs_num")
    tuk_df <- dplyr::left_join(tuk_df, map_r[, c("rhs_num", "group2"), drop = FALSE], by = "rhs_num")
  } else {
    tuk_df <- dplyr::mutate(tuk_df, group1 = .data$lhs, group2 = .data$rhs)
  }

  # attach panel display labels if panel is sliced
  if (!panel_in_model) {
    tuk_df <- .join_slice_map(tuk_df, panel_map, numeric_var = panel_slice_on, display_var = panel_var, digits = join_digits)
  }

  if ("group1" %in% names(tuk_df)) tuk_df$group1 <- .as_factor_with_levels(tuk_df$group1, x_levels)
  if ("group2" %in% names(tuk_df)) tuk_df$group2 <- .as_factor_with_levels(tuk_df$group2, x_levels)
  if (panel_var %in% names(tuk_df)) tuk_df[[panel_var]] <- .as_factor_with_levels(tuk_df[[panel_var]], panel_levels)

  # Across-strata adjustment (across all panels unless grouped)
  p_base <- if (p_across_on == "p_raw") "p_raw" else "p_tukey"
  tuk_df <- .add_p_adjustment(
    df = tuk_df,
    p_col = p_base,
    method = p_across_method,
    by_cols = p_across_by,
    new_col = "p_across"
  )

  list(emm = emm_df, tuk = tuk_df)
}


#' Compute EMMs and Tukey contrasts for a 3-way layout (x within panel × facet strata).
#'
#' If any display var is not in the model, you must explicitly provide the corresponding
#' *_slice_on numeric covariate.
EMM_triple_interaction <- function(
    m,
    data,
    x_var = "phase",
    panel_var = "region",
    facet_var = "lat",
    x_levels = NULL,
    panel_levels = NULL,
    facet_levels = NULL,
    at_vars = list(),
    weights = c("equal", "proportional"),
    conf_level = 0.95,
    x_slice_on = NULL,
    x_slice_values = NULL,
    x_slice_stat = c("median", "mean"),
    panel_slice_on = NULL,
    panel_slice_values = NULL,
    panel_slice_stat = c("median", "mean"),
    facet_slice_on = NULL,
    facet_slice_values = NULL,
    facet_slice_stat = c("median", "mean"),
    # across-strata adjustment (across all panel×facet strata unless grouped)
    p_across_method = "none",
    p_across_on = c("p_raw", "p_tukey"),
    p_across_by = NULL,
    join_digits = 8L
) {
  .check_pkgs(c("emmeans", "dplyr"))

  weights <- match.arg(weights)
  x_slice_stat <- match.arg(x_slice_stat)
  panel_slice_stat <- match.arg(panel_slice_stat)
  facet_slice_stat <- match.arg(facet_slice_stat)
  p_across_on <- match.arg(p_across_on)

  if (is.null(x_levels)) x_levels <- levels(factor(data[[x_var]]))
  if (is.null(panel_levels)) panel_levels <- levels(factor(data[[panel_var]]))
  if (is.null(facet_levels)) facet_levels <- levels(factor(data[[facet_var]]))

  data[[x_var]] <- .as_factor_with_levels(data[[x_var]], x_levels)
  data[[panel_var]] <- .as_factor_with_levels(data[[panel_var]], panel_levels)
  data[[facet_var]] <- .as_factor_with_levels(data[[facet_var]], facet_levels)

  x_in_model <- .is_in_model(m, x_var)
  panel_in_model <- .is_in_model(m, panel_var)
  facet_in_model <- .is_in_model(m, facet_var)

  if (!x_in_model) {
    if (is.null(x_slice_on)) stop("`x_var` ('", x_var, "') is not in the model. Provide `x_slice_on`.", call. = FALSE)
    .validate_numeric_col(data, x_slice_on)
    if (!.is_in_model(m, x_slice_on)) stop("`x_slice_on` is not in the model: ", x_slice_on, call. = FALSE)
  }
  if (!panel_in_model) {
    if (is.null(panel_slice_on)) stop("`panel_var` ('", panel_var, "') is not in the model. Provide `panel_slice_on`.", call. = FALSE)
    .validate_numeric_col(data, panel_slice_on)
    if (!.is_in_model(m, panel_slice_on)) stop("`panel_slice_on` is not in the model: ", panel_slice_on, call. = FALSE)
  }
  if (!facet_in_model) {
    if (is.null(facet_slice_on)) stop("`facet_var` ('", facet_var, "') is not in the model. Provide `facet_slice_on`.", call. = FALSE)
    .validate_numeric_col(data, facet_slice_on)
    if (!.is_in_model(m, facet_slice_on)) stop("`facet_slice_on` is not in the model: ", facet_slice_on, call. = FALSE)
  }

  # Guard against ambiguous mappings
  slice_vars <- c(
    if (!x_in_model) x_slice_on else NA_character_,
    if (!panel_in_model) panel_slice_on else NA_character_,
    if (!facet_in_model) facet_slice_on else NA_character_
  )
  slice_vars <- slice_vars[!is.na(slice_vars)]
  if (length(slice_vars) != length(unique(slice_vars))) {
    stop(
      "At least two display axes map to the same numeric covariate. ",
      "Please provide distinct *_slice_on variables.",
      call. = FALSE
    )
  }

  x_map <- panel_map <- facet_map <- NULL

  if (!x_in_model) {
    x_map <- .build_slice_map(
      data = data,
      display_var = x_var,
      numeric_var = x_slice_on,
      levels_vec = x_levels,
      user_vals = x_slice_values,
      stat_choice = x_slice_stat
    )
    at_vars <- utils::modifyList(at_vars, setNames(list(unique(x_map[[x_slice_on]])), x_slice_on))
  }

  if (!panel_in_model) {
    panel_map <- .build_slice_map(
      data = data,
      display_var = panel_var,
      numeric_var = panel_slice_on,
      levels_vec = panel_levels,
      user_vals = panel_slice_values,
      stat_choice = panel_slice_stat
    )
    at_vars <- utils::modifyList(at_vars, setNames(list(unique(panel_map[[panel_slice_on]])), panel_slice_on))
  }

  if (!facet_in_model) {
    facet_map <- .build_slice_map(
      data = data,
      display_var = facet_var,
      numeric_var = facet_slice_on,
      levels_vec = facet_levels,
      user_vals = facet_slice_values,
      stat_choice = facet_slice_stat
    )
    at_vars <- utils::modifyList(at_vars, setNames(list(unique(facet_map[[facet_slice_on]])), facet_slice_on))
  }

  x_spec <- if (x_in_model) x_var else x_slice_on
  panel_spec <- if (panel_in_model) panel_var else panel_slice_on
  facet_spec <- if (facet_in_model) facet_var else facet_slice_on
  form <- stats::as.formula(paste("~", x_spec, "|", panel_spec, "*", facet_spec))

  emm_obj <- emmeans::emmeans(m, specs = form, at = at_vars, weights = weights, level = conf_level)
  emm_df <- summary(emm_obj, type = "response", level = conf_level) |>
    as.data.frame() |>
    .normalize_ci_cols() |>
    .normalize_emmean_col()

  if (!x_in_model) {
    emm_df <- .join_slice_map(emm_df, x_map, numeric_var = x_slice_on, display_var = x_var, digits = join_digits)
  }
  if (!panel_in_model) {
    emm_df <- .join_slice_map(emm_df, panel_map, numeric_var = panel_slice_on, display_var = panel_var, digits = join_digits)
  }
  if (!facet_in_model) {
    emm_df <- .join_slice_map(emm_df, facet_map, numeric_var = facet_slice_on, display_var = facet_var, digits = join_digits)
  }

  if (x_var %in% names(emm_df)) emm_df[[x_var]] <- .as_factor_with_levels(emm_df[[x_var]], x_levels)
  if (panel_var %in% names(emm_df)) emm_df[[panel_var]] <- .as_factor_with_levels(emm_df[[panel_var]], panel_levels)
  if (facet_var %in% names(emm_df)) emm_df[[facet_var]] <- .as_factor_with_levels(emm_df[[facet_var]], facet_levels)

  # Pairwise contrasts within each panel×facet stratum
  pairs_obj <- pairs(emm_obj, adjust = "none")
  raw_df <- summary(pairs_obj, type = "response", infer = c(TRUE, TRUE), adjust = "none") |>
    as.data.frame()
  tuk_df <- summary(pairs_obj, type = "response", infer = c(TRUE, TRUE), adjust = "tukey") |>
    as.data.frame()

  if (!"p.value" %in% names(raw_df) || !"p.value" %in% names(tuk_df)) {
    stop("Expected 'p.value' in pairs() summaries.", call. = FALSE)
  }

  tuk_df <- dplyr::rename(tuk_df, p_tukey = p.value)
  raw_df <- dplyr::rename(raw_df, p_raw = p.value)

  by_cols <- c(panel_spec, facet_spec)
  by_cols <- by_cols[by_cols %in% names(tuk_df)]

  tuk_df <- dplyr::left_join(
    tuk_df,
    raw_df[, unique(c(by_cols, "contrast", "p_raw")), drop = FALSE],
    by = c(by_cols, "contrast")
  )

  tuk_df <- tuk_df |>
    dplyr::mutate(
      contrast = gsub("/", "-", .data$contrast)
    )
  tuk_df <- .split_contrast(tuk_df, contrast_col = "contrast", sep = " - ")

  # group1/group2 mapping
  if (!x_in_model) {
    tuk_df$lhs_num <- .extract_last_number(tuk_df$lhs)
    tuk_df$rhs_num <- .extract_last_number(tuk_df$rhs)

    map_l <- x_map
    names(map_l)[names(map_l) == x_slice_on] <- "lhs_num"
    names(map_l)[names(map_l) == x_var] <- "group1"

    map_r <- x_map
    names(map_r)[names(map_r) == x_slice_on] <- "rhs_num"
    names(map_r)[names(map_r) == x_var] <- "group2"

    map_l$lhs_num <- round(map_l$lhs_num, join_digits)
    map_r$rhs_num <- round(map_r$rhs_num, join_digits)
    tuk_df$lhs_num <- round(tuk_df$lhs_num, join_digits)
    tuk_df$rhs_num <- round(tuk_df$rhs_num, join_digits)

    tuk_df <- dplyr::left_join(tuk_df, map_l[, c("lhs_num", "group1"), drop = FALSE], by = "lhs_num")
    tuk_df <- dplyr::left_join(tuk_df, map_r[, c("rhs_num", "group2"), drop = FALSE], by = "rhs_num")
  } else {
    tuk_df <- dplyr::mutate(tuk_df, group1 = .data$lhs, group2 = .data$rhs)
  }

  # attach panel/facet display labels if sliced
  if (!panel_in_model) {
    tuk_df <- .join_slice_map(tuk_df, panel_map, numeric_var = panel_slice_on, display_var = panel_var, digits = join_digits)
  }
  if (!facet_in_model) {
    tuk_df <- .join_slice_map(tuk_df, facet_map, numeric_var = facet_slice_on, display_var = facet_var, digits = join_digits)
  }

  if ("group1" %in% names(tuk_df)) tuk_df$group1 <- .as_factor_with_levels(tuk_df$group1, x_levels)
  if ("group2" %in% names(tuk_df)) tuk_df$group2 <- .as_factor_with_levels(tuk_df$group2, x_levels)
  if (panel_var %in% names(tuk_df)) tuk_df[[panel_var]] <- .as_factor_with_levels(tuk_df[[panel_var]], panel_levels)
  if (facet_var %in% names(tuk_df)) tuk_df[[facet_var]] <- .as_factor_with_levels(tuk_df[[facet_var]], facet_levels)

  # Across-strata adjustment (across all panel×facet strata unless grouped)
  p_base <- if (p_across_on == "p_raw") "p_raw" else "p_tukey"
  tuk_df <- .add_p_adjustment(
    df = tuk_df,
    p_col = p_base,
    method = p_across_method,
    by_cols = p_across_by,
    new_col = "p_across"
  )

  list(emm = emm_df, tuk = tuk_df)
}


# ============================
# Additional: numerical predictor effects (kept; minimal refactor)
# ============================

test_numerical_var_effects <- function(
    mod,
    data,
    by = c("phase", "region"),
    test_var = "logI_c",
    slope_at = 0,
    grid_probs = seq(from=0, to=1, by=0.1),
    grid_values = NULL,
    weights = c("equal", "proportional"),
    p_adjust = "none",
    p_adjust_by = NULL
) {
  .check_pkgs(c("emmeans", "dplyr"))

  weights <- match.arg(weights)
  by_is_null <- is.null(by) || length(by) == 0

  std_emm_cols <- function(df) {
    if ("p.value" %in% names(df)) df <- dplyr::rename(df, p = p.value)
    if ("t.ratio" %in% names(df)) df <- dplyr::rename(df, stat = t.ratio)
    if ("z.ratio" %in% names(df)) df <- dplyr::rename(df, stat = z.ratio)
    if ("lower.CL" %in% names(df)) df <- dplyr::rename(df, lower = lower.CL)
    if ("upper.CL" %in% names(df)) df <- dplyr::rename(df, upper = upper.CL)
    if ("asymp.LCL" %in% names(df)) df <- dplyr::rename(df, lower = asymp.LCL)
    if ("asymp.UCL" %in% names(df)) df <- dplyr::rename(df, upper = asymp.UCL)
    df
  }

  std_joint_tests <- function(df) {
    if ("model term" %in% names(df)) df <- dplyr::rename(df, term = `model term`)
    if ("p.value" %in% names(df)) df <- dplyr::rename(df, p = p.value)

    if ("Chisq" %in% names(df)) {
      df <- dplyr::rename(df, stat = Chisq); df$test <- "Chisq"
    } else if ("LR Chisq" %in% names(df)) {
      df <- dplyr::rename(df, stat = `LR Chisq`); df$test <- "LR Chisq"
    } else if ("F.ratio" %in% names(df)) {
      df <- dplyr::rename(df, stat = F.ratio); df$test <- "F"
    } else if ("t.ratio" %in% names(df)) {
      df <- dplyr::rename(df, stat = t.ratio); df$test <- "t"
    } else if ("z.ratio" %in% names(df)) {
      df <- dplyr::rename(df, stat = z.ratio); df$test <- "z"
    } else {
      df$test <- NA_character_
    }

    if (!"df" %in% names(df) && "df1" %in% names(df)) {
      df <- dplyr::rename(df, df = df1)
    }
    df
  }

  if (!test_var %in% names(data)) stop("`test_var` not in data: ", test_var, call. = FALSE)
  model_vars <- all.vars(stats::formula(mod))
  if (!test_var %in% model_vars) {
    stop("`test_var` not in model: ", test_var,
         " (model vars: ", paste(unique(model_vars), collapse = ", "), ")", call. = FALSE)
  }
  if (!by_is_null) {
    miss_by <- setdiff(by, names(data))
    if (length(miss_by) > 0) stop("`by` vars not in data: ", paste(miss_by, collapse = ", "), call. = FALSE)
  }

  group_adjust <- function(tab, p_col = "p") {
    if (!p_col %in% names(tab)) return(tab)
    if (is.null(p_adjust_by) || length(p_adjust_by) == 0) {
      tab |>
        dplyr::mutate(q = stats::p.adjust(.data[[p_col]], method = p_adjust))
    } else {
      miss <- setdiff(p_adjust_by, names(tab))
      if (length(miss) > 0) stop("`p_adjust_by` not in table: ", paste(miss, collapse = ", "), call. = FALSE)
      tab |>
        dplyr::group_by(dplyr::across(dplyr::all_of(p_adjust_by))) |>
        dplyr::mutate(q = stats::p.adjust(.data[[p_col]], method = p_adjust)) |>
        dplyr::ungroup()
    }
  }

  by_formula <- if (by_is_null) stats::as.formula("~ 1")
  else stats::as.formula(paste("~", paste(by, collapse = " * ")))

  out <- list()

  jt_raw <- if (by_is_null) {
    as.data.frame(emmeans::joint_tests(mod))
  } else {
    as.data.frame(emmeans::joint_tests(mod, by = by))
  }
  jt <- std_joint_tests(jt_raw)

  patt <- paste0("\\b", test_var, "\\b")
  if (!"term" %in% names(jt)) stop("Could not find 'term' column in joint_tests() output.", call. = FALSE)

  omni <- jt |>
    dplyr::filter(grepl(patt, .data$term))

  if (!"Chisq" %in% names(omni)) {
    omni$Chisq <- ifelse(!is.na(omni$test) & grepl("Chisq", omni$test), omni$stat, NA_real_)
  }
  if (!"F" %in% names(omni)) {
    omni$F <- ifelse(!is.na(omni$test) & omni$test == "F", omni$stat, NA_real_)
  }

  out$omnibus <- group_adjust(omni, p_col = "p")

  # Local slope at the chosen slice
  sl <- emmeans::emtrends(mod, specs = by_formula, var = test_var, at = setNames(list(slope_at), test_var))
  sl_df <- summary(sl, infer = c(TRUE, TRUE)) |>
    as.data.frame() |>
    std_emm_cols()

  trend_col <- paste0(test_var, ".trend")
  if (!trend_col %in% names(sl_df)) {
    if ("trend" %in% names(sl_df)) trend_col <- "trend"
    else stop("Trend column not found.", call. = FALSE)
  }

  sl_sum <- sl_df
  sl_sum$slope <- sl_sum[[trend_col]]
  sl_sum$eval_at <- slope_at
  out$slope_at <- group_adjust(sl_sum, p_col = "p")

  keep_cols <- c(if (!by_is_null) by, "slope", "lower", "upper", "p", "q", "eval_at")
  out$slope_table <- out$slope_at |>
    dplyr::select(dplyr::any_of(keep_cols))

  # EMMs on a grid (curve with CI)
  grid_points <- if (is.null(grid_values)) {
    as.numeric(stats::quantile(data[[test_var]], probs = grid_probs, na.rm = TRUE))
  } else {
    if (!is.numeric(grid_values) || length(grid_values) < 2) stop("`grid_values` must be numeric length >= 2.", call. = FALSE)
    sort(unique(as.numeric(grid_values)))
  }
  grid_points <- sort(unique(grid_points))

  form_grid <- if (by_is_null) {
    stats::as.formula(paste("~", test_var))
  } else {
    stats::as.formula(paste("~", test_var, "|", paste(by, collapse = " * ")))
  }

  em_grid <- emmeans::emmeans(mod, specs = form_grid, at = setNames(list(grid_points), test_var), weights = weights)
  df_curve <- as.data.frame(em_grid) |>
    std_emm_cols()

  out$curve_df <- if (by_is_null) {
    df_curve |>
      dplyr::select(dplyr::any_of(c(test_var, "emmean", "lower", "upper"))) |>
      dplyr::arrange(.data[[test_var]])
  } else {
    df_curve |>
      dplyr::select(dplyr::any_of(c(by, test_var, "emmean", "lower", "upper"))) |>
      dplyr::arrange(dplyr::across(dplyr::all_of(by)), .data[[test_var]])
  }

  out$grid_points <- grid_points

  # Endpoint delta (p90 - p10)
  k <- length(grid_points)
  contr <- numeric(k); contr[1] <- -1; contr[k] <- 1
  
  q_lo <- min(grid_probs)
  q_hi <- max(grid_probs)
  lab <- sprintf("p%g - p%g", 100*q_hi, 100*q_lo)   # e.g., p100 - p0
  
  end_con <- emmeans::contrast(
    em_grid,
    method = setNames(list(contr), lab),
    by = if (by_is_null) NULL else by
  )
  
  end_sum <- summary(end_con, infer = c(TRUE, TRUE)) |>
    as.data.frame() |>
    std_emm_cols()
  end_sum$delta <- end_sum$estimate
  out$endpoint_delta <- group_adjust(end_sum, p_col = "p")

  # Polynomial components across the grid
  poly_con <- emmeans::contrast(em_grid, "poly", by = if (by_is_null) NULL else by)
  poly_comp <- summary(poly_con, infer = c(TRUE, TRUE)) |>
    as.data.frame() |>
    std_emm_cols()
  poly_comp$component <- poly_comp$contrast
  poly_comp$est <- poly_comp$estimate
  out$grid_poly_components <- group_adjust(poly_comp, p_col = "p")

  out
}
