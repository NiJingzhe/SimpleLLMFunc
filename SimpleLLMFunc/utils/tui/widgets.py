"""Small reusable Textual widgets for the TUI."""

from __future__ import annotations

from textual.widgets import Static


class DotPulse(Static):
    """Compact animated pulse rendered as '.', '..', '...'."""

    _FRAMES = (".", "..", "...")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self._FRAMES[0], *args, **kwargs)
        self._frame_index = 0

    def on_mount(self) -> None:
        self.set_interval(0.35, self._advance_frame)

    def _advance_frame(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(self._FRAMES)
        self.update(self._FRAMES[self._frame_index], layout=False)


__all__ = ["DotPulse"]
