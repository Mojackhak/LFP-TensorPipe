"""Stable GUI import surface.

The public GUI entry is `MainWindow`. Implementation details live under
`gui.shell`, `gui.dialogs`, `gui.stages`, and `gui.state`.
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]
