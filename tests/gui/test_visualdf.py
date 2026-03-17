import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import warnings
from matplotlib.backends.backend_agg import FigureCanvasAgg

from lfptensorpipe.viz.visualdf import _add_global_colorbar, _to_numeric_index


def test_global_colorbar_scientific_offset_text_matches_tick_label_size() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    FigureCanvasAgg(fig)

    mappable = ax.pcolormesh(
        np.array([[1.0e-6, 2.0e-6], [3.0e-6, 4.0e-6]], dtype=float),
        shading="auto",
    )
    cbar = _add_global_colorbar(
        fig,
        {
            "rect": (0.1, 0.1, 0.8, 0.9),
            "fig_w_in": 4.0,
            "cb_pad_in": 0.2,
            "cb_w_in": 0.3,
            "cbar_off_in": 0.0,
        },
        mappable=mappable,
        right_edge=0.82,
        colorbar_label=None,
        tick_label_fontsize=11,
        axis_label_fontsize=14,
    )

    assert cbar is not None

    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    fig.canvas.draw()

    tick_fontsizes = {
        tick.get_fontsize() for tick in cbar.ax.get_yticklabels() if tick.get_text()
    }
    offset_text = cbar.ax.yaxis.get_offset_text()

    assert offset_text.get_text() != ""
    assert tick_fontsizes == {11}
    assert offset_text.get_fontsize() == pytest.approx(11)

    plt.close(fig)


def test_to_numeric_index_numeric_strings_return_float_values() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _to_numeric_index(np.asarray(["1", "2.5", "3"], dtype=object))

    assert out.tolist() == pytest.approx([1.0, 2.5, 3.0])
    assert caught == []


def test_to_numeric_index_categorical_strings_fall_back_to_positions_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _to_numeric_index(np.asarray(["delta", "theta", "gamma"], dtype=object))

    assert out.tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert caught == []


def test_to_numeric_index_mixed_datetime_strings_preserve_datetime_support() -> None:
    values = np.asarray(["2024-01-01", "2024/01/02", "2024-01-03 12:00:00"], dtype=object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _to_numeric_index(values)

    expected = (
        pd.to_datetime(values, errors="raise", format="mixed")
        .astype("int64")
        .to_numpy(dtype=float)
        / 1e9
    )

    assert out.tolist() == pytest.approx(expected.tolist())
    assert caught == []
