from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from lfptensorpipe.viz import visualdf as vdf
import matplotlib.pyplot as plt
import nibabel as nib
from cmcrameri import cm
from typing import Any, Final, Literal, List


#%% configuration
value_col = "Value"
palette = ['#F2000E', '#0E6AAF', '#0CA228', '#E87000', '#884392']
label_top_bg_color = "#E3DCCF"
label_right_bg_color = "#D7E3E0"
raw_mean_line_color = '#404040'
hline_color = '#404040'
label_text_color = "black"
label_fontweight = "bold"
panel_gap = (3.0, 3.0)
dpi = 600
line_width = 1
font_size = 7
tick_label_size = 6
grid = False
show_top_right_axes = False
transparent = True

ribbon_alpha = 0.30
legend_loc = 'outside_right'

band_shadows = {
    (1, 4): "#E5E5E540",    # delta
    (4, 8): "#CCCCCC40",    # theta
    (8, 13): "#B3B3B340",   # alpha
    (13, 20): "#9A9A9A40",  # beta_low
    (20, 35): "#81818140",  # beta_high
    (35, 80): "#68686840"  # gamma
}

band_shadows2 = {
    (4, 8): "#CCCCCC40",    # theta
    (8, 13): "#B3B3B340",   # alpha
    (13, 20): "#9A9A9A40",  # beta_low
    (20, 35): "#81818140",  # beta_high
    (35, 80): "#68686840"  # gamma
}

boxsize_base = (6, 25)
boxsize = (30, 25)
strip_top_height_mm = 4.5   
strip_right_width_mm = 4.5  
strip_pad_mm = 2          
x_label_offset_mm = 5      
y_label_offset_mm = 9      
colorbar_width_mm = 3
colorbar_pad_mm = 2
cbar_label_offset_mm = 9

def save_fig(fig, save_path: str | Path):
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

def type_n_cfg(config_updated: dict, type_n: int) -> dict:
    
    if type_n <= 2: # only top strip
        config_updated.pop('strip_right_width_mm', None)
        config_updated.pop('label_right_bg_color', None)
    
    if type_n <= 1: # no strips and multiple panels
        config_updated.pop('strip_top_height_mm', None)
        config_updated.pop('strip_pad_mm', None)
        config_updated.pop('label_fontsize', None)
        config_updated.pop('label_top_bg_color', None)
        config_updated.pop('label_text_color', None)
        config_updated.pop('label_fontweight', None)
        config_updated.pop('panel_gap', None)
    
    return config_updated

#%% label dict
RAW_YLABEL_TPL: Final[dict[str, str]] = {
    "aperiod-mean": "{prefix}{key}",
    "period-mean": "{prefix}Periodic {key}power (dB)",
    "raw-mean": "{prefix}Total {key}power (dB)",
    "burst-rate": r"{prefix}$\operatorname{{arcsinh}}$ {key}burst rate (Hz)",
    "burst-duration": r"{prefix}$\log_{{10}}$ {key}burst duration (s)",
    # "burst-duration": r"{prefix}$\operatorname{{ln}}$ {key}burst duration (s)",
    "burst-occupation": "{prefix}{key}burst occupation",
    "burst-mean": r"{prefix}$\log_{{10}}$ {key}burst amplitude (V)",
    # "burst-mean": r"{prefix}$\operatorname{{ln}}$ {key}burst amplitude (V)",
    "coh-mean": r"{prefix}Fisher z of $\sqrt{{{key}\mathrm{{coherence}}}}$",
    "ciplv-mean": r"{prefix}$\operatorname{{logit}}({key}\mathrm{{ciPLV}})$",
    "wpli-mean": r"{prefix}$\operatorname{{logit}}({key}\mathrm{{wPLI}})$",
    "plv-mean": r"{prefix}$\operatorname{{logit}}({key}\mathrm{{PLV}})$",
    "pli-mean": r"{prefix}$\operatorname{{logit}}({key}\mathrm{{PLI}})$",    
    "delta_net_gc-mean": "{prefix}{key}TRGC",
    "psi-mean": "{prefix}{key}PSI",
  
}

#%% Band dict
BAND_TPL = {'Delta': 'δ ', 'Theta': 'θ ', 'Alpha': 'α ', 'Beta': 'β ', 
            'Beta_low': 'β-low ', 'Beta_high': 'β-high ', 'Gamma': 'γ ',
            'Exponent': 'Exponent ', 'Offset': 'Offset ', 
            'Error_mae': 'Error mae ', 'Gof_rsquared': 'Gof rsquared ',}

#%% Prefix dict
PREFIX_TPL = {'raw': '', 'norm': 'Δ', 'percent': 'Δ%'}

#%% label format function
def format_label(param_type: str, *, prefix: str = "", key: str = "") -> str:
    try:
        tpl = RAW_YLABEL_TPL[param_type]
    except KeyError as e:
        raise KeyError(f"Unknown param_type: {param_type!r}") from e
    return tpl.format(prefix=prefix, key=key)

#%%  scalar plot config

scalar_cfg = {
    # --- data ---
    
    # --- roles ---
    'value_col': value_col,  

    # --- palettes ---
    'jitter_palette': palette,
    'fill_palette': palette,
    'outline_palette': palette,

    # --- visual style ---
    'jitter_width': 0.20,
    'jitter_alpha': 1,
    'jitter_size': 2,
    'grid': grid,
    'dpi': dpi,
    'seed': 42,
    'show_top_right_axes': show_top_right_axes,
    
    # --- boxplot controls ---
    'show_box': True,
    'box_width': 0.55,
    'fill_alpha': 0.30,
    'whiskers': 'tukey',
    
    # global fallback outline styles
    'box_edge_width': line_width,
    'median_linewidth': line_width,
    'whisker_linewidth': line_width,
    'cap_linewidth': line_width,

    # outliers
    'outlier_marker': "o",
    'outlier_markersize': 0,

    # EMM line
    'emm_line_width': line_width,
    'emm_line_color': "black",
    'emm_line_style': "-",

    # Raw mean line
    'raw_mean_line_width': line_width,
    'raw_mean_line_color': raw_mean_line_color,
    'raw_mean_line_style': "--",

    # Error bar styling
    'error_bar_linewidth': line_width,
    'error_bar_cap': 2.0,
    'error_bar_color': "black",
    
    # --- vertical / horizontal references ---
    'horizontal_lines': [0,], 
    'hline_color': 'black',   
    'hline_style': ':',
    'hline_width': line_width,
    'hline_alpha': 0.3,

    # --- strips (top per column; right per row) — sizes RELATIVE TO PANEL ---
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    
    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,      
    
    # --- text sizes ---
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # --- BOXSIZE-BASED GEOMETRY (deterministic) ---
    'include_global_label_margins': True,

    # absolute gaps between panels (inner-box to inner-box)
    'panel_gap': panel_gap,

    # --- Tukey bracket controls ---
    'show_brackets': True,
    'hide_ns': True,
    'y_start': 0.50,
    'y_end': 0.90,
    'y_step': 0.08,
    'bracket_height_frac': 0.0,
    'bracket_color': "black",
    'bracket_linewidth': line_width,
    'bracket_text_size': font_size,

    # --- transparency ---
    'transparent': transparent,
}


#%% wrapper functions

def update_scalar_cfg(x_var, n, type_n):
    
    config_updated = scalar_cfg.copy()
    boxsize = (boxsize_base[0]*n, boxsize_base[1])
    config_updated['boxsize'] = boxsize
        
    config_updated['jitter_var'] = x_var
    config_updated['fill_var'] = x_var
    config_updated['outline_var'] = x_var
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated
    
def plot_scalar_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    bd: str,
    x_var: str,
    x_levels: list[str] | None = None,
    panel_var: str | None = None,
    facet_var: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    cfg_update: dict | None = None,
    fig_format: str = '.pdf',
) -> plt.Figure:
    
    n = len(df[x_var].unique())
    type_n = 1

    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{x_var}-{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{x_var}-{panel_var}'
        else:
            filebase = f'{x_var}'
    
    save_dir = Path(save_dir)
    emm = pd.read_csv(save_dir / f'{filebase}_emm.csv')
    tuk = pd.read_csv(save_dir / f'{filebase}_tukey.csv')   
    
    config_updated = update_scalar_cfg(x_var, n, type_n)
    if x_label is not None:
        config_updated['x_label'] = x_label
    else:
        config_updated['x_label'] = f'{x_var}'
    
    if y_label is not None:
        config_updated['y_label'] = y_label
    else:
        key = BAND_TPL[bd]
        prefix = PREFIX_TPL[df_type]
        y_label = format_label(param_type, prefix=prefix, key=key)
        config_updated['y_label'] = y_label
    
    
    if cfg_update is not None:
        config_updated.update(cfg_update)
    
    if type_n == 1:
        fig = vdf.plot_single_effect_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            x_levels=x_levels,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            x_levels=x_levels,
            panel_var=panel_var,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            x_levels=x_levels,
            panel_var=panel_var,
            facet_var=facet_var,
            **config_updated
        )           
    save_path = save_dir / f'{filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig

#%%  series plot config

series_cfg = {
    
    'value_col': value_col,  
            
    # figure / appearance
    'line_palette': palette,
    'line_width': line_width,
    'ribbon_alpha': ribbon_alpha,
    'grid': grid,
    'dpi': dpi,

    # horizontal references (apply to every panel)
    'vline_color': "#000000",   
    'hline_color': hline_color,
    'hline_style': "--",
    'hline_width': line_width,
    'hline_alpha': 0.6,
    
    # --- strips (top per column; right per row) — sizes RELATIVE TO PANEL ---
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,    
    
    # Legend controls
    'legend_loc': legend_loc,
    'legend_fontsize': font_size,

    # text sizes
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # legend
    'legend_fontsize': font_size,

    # absolute gaps between panels (inner-box to inner-box)
    'boxsize': boxsize,
    'panel_gap': panel_gap,    # (gap_x, gap_y)

    # transparency
    'transparent': transparent,
}

series_spectral_cfg = {
    'x_log': True,
    'x_limits': [1, 75],
    'x_label': 'Frequency (Hz)',
    'vertical_shadows': band_shadows,  
    **series_cfg,
}

series_spectral_cfg2 = {
    'x_log': True,
    'x_limits': [4, 75],
    'x_label': 'Frequency (Hz)',
    'vertical_shadows': band_shadows2,  
    **series_cfg,
}

series_section_spectral_cfg = {
    'med': series_spectral_cfg,
    'default': series_spectral_cfg2,
}

series_trace_cfg = {
    'x_log': False,
    **series_cfg,
}

series_cycle_trace_cfg = {
    'x_log': False,
    'x_limits': [0, 100],
    'x_label': 'Gait cycle (%)',
    'vertical_shadows': {
        (0, 5): "#B3B3B340",    
        (13, 23): "#B3B3B340",  
        (45, 55): "#B3B3B340",    
        (63, 73): "#B3B3B340",    
        (95, 100): "#B3B3B340",    
    },
    'vertical_lines': [18, 50, 68],
    **series_cfg,
}

series_turn_trace_cfg = {
    'x_log': False,
    'x_limits': [0, 2],
    'x_label': 'Time (s)',
    'vertical_shadows': {
        (0, 0.5): "#E5E5E540",    # BL
        (0.5, 1.0): "#CCCCCC40",  # ONS
        (1.0, 1.5): "#B3B3B340",     # SUS
        (1.5, 2.0): "#9A9A9A40",    # OFS
    },
    'vertical_lines': [0.5, 1, 1.5], 
    **series_cfg,
}

series_turn_stack_trace_cfg = {
    'x_log': False,
    'x_limits': [0, 100],
    'x_label': 'Turning time (%)',
    **series_cfg,
}

series_section_trace_cfg = {
    'cycle': series_cycle_trace_cfg,
    'turn': series_turn_trace_cfg,
    'turn_stack': series_turn_stack_trace_cfg,
    'default': series_trace_cfg,
}

#%% smooth trace function

def smooth_trace(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    def apply_lowess(series, frac=0.05):
        x = np.arange(len(series))
        y = series.values
        smoothed = lowess(y, x, frac=frac, return_sorted=False)
        return pd.Series(smoothed, index=series.index)
    
    frac = 0.05
    df['Value'] = df["Value"].map(lambda s: apply_lowess(s, frac))
    return df

#%% update config functions

def update_series_spectral_cfg(type_n, section):
    
    config_updated = series_section_spectral_cfg.get(
        section, series_section_spectral_cfg['default']).copy()
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated

def update_series_trace_cfg(type_n, section):
    
    config_updated = series_section_trace_cfg.get(
        section, series_section_trace_cfg['default']).copy()
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated

DERIVE_TYPE = Literal["spectral", "trace"]

def plot_series_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    x_var: str,
    x_levels: list[str] | None = None,
    bd: str | None = None,
    panel_var: str | None = None,
    facet_var: str | None = None,
    derive_type: DERIVE_TYPE = 'spectral',
    section: str | None = None,
    fig_format: str = '.pdf',
) -> plt.Figure:
    
    type_n = 1
    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{x_var}-{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{x_var}-{panel_var}'
        else:
            filebase = f'{x_var}'
    
    save_dir = Path(save_dir)
    
    if derive_type == 'spectral':
        config_updated = update_series_spectral_cfg(type_n, section)
    else:  # derive_type == 'trace'
        config_updated = update_series_trace_cfg(type_n, section)
        
    prefix = PREFIX_TPL[df_type]
    
    if bd is None:
        key = ''
    else:
        key = BAND_TPL[bd]
        
    y_label = format_label(param_type, prefix=prefix, key=key)
    config_updated['y_label'] = y_label
    
    if df_type == 'norm':
        config_updated['horizontal_lines'] = [0,]
            
    if type_n == 1:
        fig = vdf.plot_single_effect_series(
            df=df,
            x_var=x_var,
            x_levels=x_levels,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_series(
            df=df,
            x_var=x_var,
            x_levels=x_levels,
            panel_var=panel_var,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_series(
            df=df,
            x_var=x_var,
            x_levels=x_levels,
            panel_var=panel_var,
            facet_var=facet_var,
            **config_updated
        )           
    
    save_path = save_dir / f'{filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig

#%% correlaion plot config

#%% raw plot config

raw_cfg = {
    
    'value_col': value_col,  
    'mode': 'mean',

    # axis labels (figure-level when single_*_label=True)
    'y_label': 'Frequency (Hz)',

    # axes limits/scales
    'y_limits': [4, 75],
    'x_log': False,
    'y_log': True,
    
    # figure / appearance
    'cmap': cm.vik,
    'vmode': 'sym',
    'grid': grid,
    'dpi': dpi,
    
    # vertical references (apply to every panel)
    'vline_color': "#000000",
    'vline_style': ":",
    'vline_width': line_width,
    'vline_alpha': 0.3,

    # horizontal references (apply to every panel)
    'horizontal_lines': [4, 8, 13, 20, 35],
    'hline_color': "#000000",
    'hline_style': ":",
    'hline_width': line_width,
    'hline_alpha': 0.3,
    
    # --- strips (top per column; right per row) — sizes RELATIVE TO PANEL ---
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,    
    'colorbar_width_mm': colorbar_width_mm,
    'colorbar_pad_mm': colorbar_pad_mm,
    'cbar_label_offset_mm': cbar_label_offset_mm,
    
    # text sizes
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # absolute gaps between panels (inner-box to inner-box)
    'boxsize': boxsize,
    'panel_gap': panel_gap,    # (gap_x, gap_y)

    # transparency
    'transparent': transparent,
}


raw_turn_cfg = {
    'x_limits': [0, 2],
    'x_label': 'Time (s)',
    'vertical_lines': [0.5, 1, 1.5],
    **raw_cfg,
}

raw_turn_stack_cfg = {
    'x_limits': [0, 100],
    'x_label': 'Turning time (%)',
    'vertical_lines': [],
    **raw_cfg,
}

raw_cycle_cfg = {
    'x_limits': [0, 100],
    'x_label': 'Gait cycle (%)',
    'vertical_lines': [18, 50, 68],
    **raw_cfg,
}

raw_med_burst_example_cfg = {
    **raw_cfg,
    'x_limits': [0, 60],
    'x_label': 'Time (s)',
    'y_label': 'Band',
    'y_limits': None,
    'y_log': False,
    'horizontal_lines': None,
    'vertical_lines': None,
    'cmap': 'viridis',
    'vmode': 'auto',
}

raw_section_cfg = {
    'cycle': raw_cycle_cfg,
    'turn': raw_turn_cfg,
    'turn_stack': raw_turn_stack_cfg,
    'med_burst_example': raw_med_burst_example_cfg,
    'default': raw_cfg,
}

def update_raw_cfg(type_n, section):
    
    config_updated = raw_section_cfg.get(
        section, raw_section_cfg['default']).copy()
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated


def plot_raw_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    panel_var: str | None = None,
    panel_levels: list[str] | None = None,
    facet_var: str | None = None,
    section: str | None = None,
    filebase_override: str | None = None,
    fig_format: str = '.pdf',
) -> plt.Figure:
    
    type_n = 1
    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{panel_var}'
        else:
            filebase = 'All'

    if filebase_override is not None:
        filebase = filebase_override
    
    save_dir = Path(save_dir)

    config_updated = update_raw_cfg(type_n, section)
            
    prefix = PREFIX_TPL[df_type]
        
    colorbar_label = format_label(param_type, prefix=prefix, key='')
    config_updated['colorbar_label'] = colorbar_label
    
            
    if type_n == 1:
        fig = vdf.plot_single_effect_df(
            df=df,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_df(
            df=df,
            panel_var=panel_var,
            panel_levels=panel_levels,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_df(
            df=df,
            panel_var=panel_var,
            panel_levels=panel_levels,
            facet_var=facet_var,
            **config_updated
        )           
    
    save_path = save_dir / f'{filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig
