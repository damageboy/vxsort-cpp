"""Minimal Textual UI scaffold for wave progress."""

from __future__ import annotations

import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from memory_monitor import collect_memory_snapshot
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import DescendantFocus, MouseDown, MouseMove, MouseUp, Resize
from textual.widgets import Log, Static

try:
    from .instruction_stats import InstructionAttemptStats
except ImportError:
    # type: ignore[no-redef]
    from instruction_stats import InstructionAttemptStats


@dataclass(slots=True)
class StageSelectionModel:
    """Track the selected stage and clamp navigation within valid bounds."""

    stage_count: int
    selected_stage: int = 0

    def __post_init__(self) -> None:
        if self.stage_count < 1:
            msg = "stage_count must be at least 1"
            raise ValueError(msg)
        self.selected_stage = max(0, min(self.selected_stage, self.stage_count - 1))

    def move(self, delta: int) -> int:
        """Move the selected stage while staying within valid bounds."""
        self.selected_stage = max(
            0, min(self.selected_stage + delta, self.stage_count - 1)
        )
        return self.selected_stage


@dataclass(frozen=True, slots=True)
class InstructionStatsRow:
    """A render-ready instruction stats row."""

    key: str
    attempts_total: int
    attempts_no_valid: int
    attempts_valid: int
    attempts_unique: int


@dataclass(frozen=True, slots=True)
class StageProgressSnapshot:
    """A render-ready snapshot of one wave stage."""

    stage_idx: int
    attempts: int
    total: int
    valid: int
    unique: int


def _instruction_stats_sort_key(
    item: tuple[str, InstructionAttemptStats],
) -> tuple[float, int, int, str]:
    key, stats = item
    unique_ratio = (
        stats.attempts_unique / stats.attempts_total if stats.attempts_total else 0.0
    )
    return (-unique_ratio, -stats.attempts_unique, -stats.attempts_total, key)


def select_stage_rows(
    snapshot_by_stage: dict[int, dict[str, InstructionAttemptStats]],
    selected_stage: int,
    top_k: int,
) -> list[InstructionStatsRow]:
    """Return the top instruction rows for the selected stage only."""
    if top_k <= 0:
        return []

    stage_stats = snapshot_by_stage.get(selected_stage, {})
    rows = sorted(stage_stats.items(), key=_instruction_stats_sort_key)
    return [
        InstructionStatsRow(
            key=key,
            attempts_total=stats.attempts_total,
            attempts_no_valid=stats.attempts_no_valid,
            attempts_valid=stats.attempts_valid,
            attempts_unique=stats.attempts_unique,
        )
        for key, stats in rows[:top_k]
    ]


@dataclass(slots=True)
class FocusModel:
    """Track which top pane currently owns keyboard navigation."""

    VALID_PANES = frozenset({"stage", "stats"})
    current: str = "stage"

    def __setattr__(self, name: str, value: str) -> None:
        if name == "current" and value not in self.VALID_PANES:
            msg = "current must be 'stage' or 'stats'"
            raise ValueError(msg)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        if self.current not in self.VALID_PANES:
            msg = "current must be 'stage' or 'stats'"
            raise ValueError(msg)

    def toggle(self) -> str:
        """Swap keyboard focus between the stage and stats panes."""
        self.current = "stats" if self.current == "stage" else "stage"
        return self.current


class FocusablePane(Static, can_focus=True):
    """Simple focusable pane used for keyboard navigation."""


class VerticalResizeHandle(Static):
    """A one-line drag handle that resizes the top and bottom panes."""

    DEFAULT_CSS = """
    VerticalResizeHandle {
        height: 1;
        content-align: center middle;
        background: $boost;
        color: $text;
    }

    VerticalResizeHandle:hover {
        background: $accent 20%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(" drag to resize ", **kwargs)
        self._dragging = False

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.button != 1:
            return
        self._dragging = True
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: MouseMove) -> None:
        if not self._dragging or event.delta_y == 0:
            return
        self.app.adjust_top_panel(event.delta_y)
        event.stop()

    def on_mouse_up(self, event: MouseUp) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self.release_mouse()
        event.stop()


class WaveProgressApp(App):
    """Textual application scaffold for wave progress."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #top-region {
        min-height: 8;
        height: 16;
    }

    #stage-pane {
        width: 60%;
        border: round $accent;
        padding: 0 0;
    }

    #instruction-stats-pane {
        width: 40%;
        border: round $primary;
        padding: 0 0;
    }

    #log-pane {
        border: round $surface;
    }

    #status-line {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("tab", "toggle_focus", "Toggle Focus", priority=True),
        ("up", "previous_stage", "Prev Stage"),
        ("down", "next_stage", "Next Stage"),
        ("q", "quit", "Quit"),
    ]

    MIN_TOP_PANEL_HEIGHT = 8
    MIN_LOG_PANEL_HEIGHT = 5
    DEFAULT_TOP_PANEL_HEIGHT = 18
    STAGE_TABLE_OVERHEAD_ROWS = 8
    RESERVED_ROWS = 0
    INSTRUCTION_WINDOW_ROWS = 3
    DEFAULT_INSTRUCTION_TOP_K = 8
    RATE_EMA_ALPHA = 0.35

    def __init__(
        self,
        stage_count: int = 1,
        selected_stage: int = 0,
        stage_metrics: list[StageProgressSnapshot] | None = None,
        snapshot_by_stage: dict[int, dict[str, InstructionAttemptStats]] | None = None,
        instruction_top_k: int | None = None,
        memory_text: str | None = None,
        status_text: str | None = None,
        log_lines: list[str] | None = None,
        ready_event: threading.Event | None = None,
        on_quit_requested: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.stage_selection = StageSelectionModel(
            stage_count=stage_count,
            selected_stage=selected_stage,
        )
        self.focus_model = FocusModel()
        self._top_panel_height = max(
            self.DEFAULT_TOP_PANEL_HEIGHT,
            stage_count + self.STAGE_TABLE_OVERHEAD_ROWS,
        )
        self._stage_metrics = list(
            stage_metrics
            or [
                StageProgressSnapshot(
                    stage_idx=stage_idx,
                    attempts=0,
                    total=0,
                    valid=0,
                    unique=0,
                )
                for stage_idx in range(stage_count)
            ]
        )
        self._snapshot_by_stage = {
            stage_idx: dict(stage_stats)
            for stage_idx, stage_stats in (snapshot_by_stage or {}).items()
        }
        self._instruction_top_k = instruction_top_k or self.DEFAULT_INSTRUCTION_TOP_K
        self._instruction_scroll_offset = 0
        self._memory_text = memory_text
        self._status_text = status_text
        self._log_lines = list(log_lines or [])
        self._ready_event = ready_event
        self._on_quit_requested = on_quit_requested
        self._mounted = False
        self._last_attempts_by_stage: dict[int, int] = {}
        self._last_attempt_time_by_stage: dict[int, float] = {}
        self._rate_by_stage: dict[int, float] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="top-region"):
                yield FocusablePane(id="stage-pane")
                yield FocusablePane(id="instruction-stats-pane")
            yield VerticalResizeHandle(id="top-bottom-separator")
            yield Log(id="log-pane")
            yield Static(id="status-line")

    def on_mount(self) -> None:
        self._mounted = True
        self.adjust_top_panel(0)
        log_widget = self.query_one(Log)
        if self._log_lines:
            for line in self._log_lines:
                log_widget.write_line(line)
        self._refresh_stage_pane()
        self._refresh_instruction_stats_pane()
        self._refresh_status_line()
        self._sync_widget_focus()
        if self._ready_event is not None:
            self._ready_event.set()

    def on_unmount(self) -> None:
        self._mounted = False

    def on_resize(self, event: Resize) -> None:
        self.adjust_top_panel(0, viewport_height=event.size.height)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        self._sync_focus_model_from_widget(event.widget)
        self._refresh_status_line()

    def action_previous_stage(self) -> None:
        if self._active_pane() == "stage":
            self.stage_selection.move(-1)
            self._refresh_stage_pane()
            self._instruction_scroll_offset = 0
            self._refresh_instruction_stats_pane()
            self._refresh_status_line()
            return
        self._scroll_instruction_rows(-1)

    def action_next_stage(self) -> None:
        if self._active_pane() == "stage":
            self.stage_selection.move(1)
            self._refresh_stage_pane()
            self._instruction_scroll_offset = 0
            self._refresh_instruction_stats_pane()
            self._refresh_status_line()
            return
        self._scroll_instruction_rows(1)

    def action_toggle_focus(self) -> None:
        self.focus_model.toggle()
        self._sync_widget_focus()
        self._refresh_status_line()

    def action_quit(self) -> None:
        if self._on_quit_requested is not None:
            self._on_quit_requested()
        self.exit()

    def adjust_top_panel(self, delta: int, viewport_height: int | None = None) -> None:
        """Resize the top pane while preserving a usable log pane."""
        height = viewport_height or self.size.height or self.DEFAULT_TOP_PANEL_HEIGHT
        max_top_height = max(
            self.MIN_TOP_PANEL_HEIGHT,
            height - self.MIN_LOG_PANEL_HEIGHT - self.RESERVED_ROWS,
        )
        self._top_panel_height = max(
            self.MIN_TOP_PANEL_HEIGHT,
            min(self._top_panel_height + delta, max_top_height),
        )
        self.query_one("#top-region").styles.height = self._top_panel_height

    def _refresh_stage_pane(self) -> None:
        table = Table(
            title="Stage Progress",
            show_header=True,
            header_style="bold cyan",
            box=None,
            pad_edge=False,
        )
        table.add_column("Stage", no_wrap=True)
        table.add_column("Progress", no_wrap=True)
        table.add_column("Attempts", justify="right")
        table.add_column("Valid", justify="right")
        table.add_column("Unique", justify="right")
        table.add_column("Rate", justify="right")

        for snapshot in self._stage_metrics:
            selected = snapshot.stage_idx == self.stage_selection.selected_stage
            marker = ">" if selected else " "
            stage_label = f"{marker}{snapshot.stage_idx + 1}"
            stage_cell = Text(stage_label, style="bold" if selected else "")
            table.add_row(
                stage_cell,
                self._render_stage_bar(snapshot),
                str(snapshot.attempts),
                str(snapshot.valid),
                str(snapshot.unique),
                f"{self._rate_by_stage.get(snapshot.stage_idx, 0.0):7.2f}/s",
            )

        try:
            stage_pane = self.query_one("#stage-pane", Static)
            stage_pane.update(Group(table))
        except NoMatches:
            return

    def _render_stage_bar(self, snapshot: StageProgressSnapshot) -> Text:
        """Render the original dual-layer bar logic (valid over attempted)."""
        total_width = 60
        total = snapshot.total if snapshot.total > 0 else max(snapshot.attempts, 1)
        attempted_fraction = min(max(snapshot.attempts / total, 0.0), 1.0)
        valid_fraction = min(max(snapshot.valid / total, 0.0), attempted_fraction)

        valid_v = valid_fraction * total_width
        attempt_v = attempted_fraction * total_width
        blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
        attempt_style = "yellow"
        success_style = "green"
        success_on_attempt_style = "green on yellow"

        pieces = Text("[")
        for i in range(total_width):
            s_level = max(0.0, min(1.0, valid_v - i))
            a_level = max(0.0, min(1.0, attempt_v - i))

            if s_level >= 1.0:
                pieces.append("█", style=success_style)
                continue

            s_idx = int(round(s_level * 8))
            if s_idx > 0:
                style = success_on_attempt_style if a_level > s_level else success_style
                pieces.append(blocks[s_idx], style=style)
                continue

            if a_level >= 1.0:
                pieces.append("█", style=attempt_style)
                continue

            a_idx = int(round(a_level * 8))
            if a_idx > 0:
                pieces.append(blocks[a_idx], style=attempt_style)
                continue

            pieces.append(" ")

        pieces.append("]")
        return pieces

    def _refresh_instruction_stats_pane(self) -> None:
        rows = select_stage_rows(
            self._snapshot_by_stage,
            selected_stage=self.stage_selection.selected_stage,
            top_k=self._instruction_top_k,
        )
        max_offset = max(0, len(rows) - self.INSTRUCTION_WINDOW_ROWS)
        self._instruction_scroll_offset = min(
            self._instruction_scroll_offset, max_offset
        )
        start = self._instruction_scroll_offset
        visible_rows = rows[start : start + self.INSTRUCTION_WINDOW_ROWS]
        table = Table(
            title="Instruction Stats",
            show_header=True,
            header_style="bold cyan",
            box=None,
            pad_edge=False,
        )
        table.add_column("Instr", no_wrap=True)
        table.add_column("Attempts", justify="right")
        table.add_column("NoValid", justify="right")
        table.add_column("Valid", justify="right")
        table.add_column("Unique", justify="right")

        if visible_rows:
            for row in visible_rows:
                table.add_row(
                    row.key,
                    str(row.attempts_total),
                    str(row.attempts_no_valid),
                    str(row.attempts_valid),
                    str(row.attempts_unique),
                )
            try:
                self.query_one("#instruction-stats-pane", Static).update(Group(table))
            except NoMatches:
                return
            return

        try:
            self.query_one("#instruction-stats-pane", Static).update(
                Group(table, Text("No instruction stats for selected stage."))
            )
        except NoMatches:
            return

    def _refresh_status_line(self) -> None:
        try:
            self.query_one("#status-line", Static).update(self._status_line_text())
        except NoMatches:
            return

    def _status_line_text(self) -> str:
        current_stage = self.stage_selection.selected_stage + 1
        stage_count = self.stage_selection.stage_count
        if self._memory_text is not None:
            memory_text = self._memory_text
        else:
            snapshot = collect_memory_snapshot()
            memory_text = (
                snapshot.format() if snapshot is not None else "Memory: unavailable"
            )
        focus_text = (
            "Stage focus" if self.focus_model.current == "stage" else "Stats focus"
        )
        status_parts = [memory_text]
        if self._status_text:
            status_parts.append(self._status_text)
        status_parts.extend(
            [
                f"Stage {current_stage}/{stage_count}",
                focus_text,
                "Tab toggle focus",
                "Up/Down navigate focused pane",
                "Drag separator",
                "q stop+quit",
            ]
        )
        return " | ".join(status_parts)

    def _scroll_instruction_rows(self, delta: int) -> None:
        row_count = len(
            select_stage_rows(
                self._snapshot_by_stage,
                selected_stage=self.stage_selection.selected_stage,
                top_k=self._instruction_top_k,
            )
        )
        max_offset = max(0, row_count - self.INSTRUCTION_WINDOW_ROWS)
        self._instruction_scroll_offset = max(
            0, min(self._instruction_scroll_offset + delta, max_offset)
        )
        self._refresh_instruction_stats_pane()
        self._refresh_status_line()

    def _sync_widget_focus(self) -> None:
        target_id = (
            "#stage-pane"
            if self.focus_model.current == "stage"
            else "#instruction-stats-pane"
        )
        self.set_focus(self.query_one(target_id, FocusablePane), scroll_visible=False)

    def _active_pane(self) -> str:
        focused = self.focused
        self._sync_focus_model_from_widget(focused)
        return self.focus_model.current

    def _sync_focus_model_from_widget(self, widget: object | None) -> None:
        if getattr(widget, "id", None) == "stage-pane":
            self.focus_model.current = "stage"
        elif getattr(widget, "id", None) == "instruction-stats-pane":
            self.focus_model.current = "stats"

    def set_stage_metrics(self, stage_metrics: list[StageProgressSnapshot]) -> None:
        """Update all stage metrics without affecting stage selection."""
        now = time.monotonic()
        for snapshot in stage_metrics:
            stage_idx = snapshot.stage_idx
            prev_attempts = self._last_attempts_by_stage.get(stage_idx)
            if prev_attempts is None:
                self._last_attempts_by_stage[stage_idx] = snapshot.attempts
                self._last_attempt_time_by_stage[stage_idx] = now
                self._rate_by_stage.setdefault(stage_idx, 0.0)
                continue

            delta_attempts = snapshot.attempts - prev_attempts
            if delta_attempts > 0:
                prev_time = self._last_attempt_time_by_stage.get(stage_idx, now)
                elapsed = max(now - prev_time, 1e-6)
                instant_rate = delta_attempts / elapsed
                prev_rate = self._rate_by_stage.get(stage_idx, 0.0)
                if prev_rate <= 0.0:
                    self._rate_by_stage[stage_idx] = instant_rate
                else:
                    alpha = self.RATE_EMA_ALPHA
                    self._rate_by_stage[stage_idx] = (
                        alpha * instant_rate + (1.0 - alpha) * prev_rate
                    )
                self._last_attempt_time_by_stage[stage_idx] = now
            elif delta_attempts < 0:
                # Counter reset (e.g. resumed/restarted run): reset local rate state.
                self._rate_by_stage[stage_idx] = 0.0
                self._last_attempt_time_by_stage[stage_idx] = now

            self._last_attempts_by_stage[stage_idx] = snapshot.attempts
            self._rate_by_stage.setdefault(snapshot.stage_idx, 0.0)
        self._stage_metrics = list(stage_metrics)
        if self._mounted:
            self._refresh_stage_pane()
            self._refresh_status_line()

    def set_instruction_stats_snapshot(
        self,
        snapshot_by_stage: dict[int, dict[str, InstructionAttemptStats]],
    ) -> None:
        """Replace instruction stats data without affecting selected stage."""
        self._snapshot_by_stage = {
            stage_idx: dict(stage_stats)
            for stage_idx, stage_stats in snapshot_by_stage.items()
        }
        if self._mounted:
            self._refresh_instruction_stats_pane()
            self._refresh_status_line()

    def append_log_line(self, line: str) -> None:
        """Append one runtime log line."""
        self._log_lines.append(line)
        if self._mounted:
            self.query_one(Log).write_line(line)

    def set_memory_text(self, memory_text: str | None) -> None:
        """Set the status-line memory text."""
        self._memory_text = memory_text
        if self._mounted:
            self._refresh_status_line()

    def set_status_text(self, status_text: str | None) -> None:
        """Set additional runtime status text shown in the footer."""
        self._status_text = status_text
        if self._mounted:
            self._refresh_status_line()


class NullWaveRuntimeSession:
    """No-op runtime session used for tests and non-interactive runs."""

    def __init__(self, stage_count: int, selected_stage: int = 0) -> None:
        self.stage_count = stage_count
        self.selected_stage = selected_stage

    def __enter__(self) -> NullWaveRuntimeSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update_stage_metrics(self, stage_metrics: list[StageProgressSnapshot]) -> None:
        del stage_metrics

    def update_memory_line(self, memory_text: str) -> None:
        del memory_text

    def update_instruction_stats(
        self,
        snapshot_by_stage: dict[int, dict[str, InstructionAttemptStats]],
    ) -> None:
        del snapshot_by_stage

    def log(self, line: str) -> None:
        del line

    def set_status(self, text: str | None) -> None:
        del text


class TextualWaveRuntimeSession:
    """Run the Textual app in a background thread and forward runtime updates."""

    READY_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        stage_count: int,
        instruction_top_k: int | None = None,
        selected_stage: int = 0,
        app_factory=None,
        headless: bool = False,
        inline: bool = False,
        warning_callback: Callable[[str], None] | None = None,
        on_quit_requested: Callable[[], None] | None = None,
    ) -> None:
        self._ready_event = threading.Event()
        app_factory = app_factory or WaveProgressApp
        self._app = app_factory(
            stage_count=stage_count,
            selected_stage=selected_stage,
            instruction_top_k=instruction_top_k,
            ready_event=self._ready_event,
            on_quit_requested=on_quit_requested,
        )
        self._headless = headless
        self._inline = inline
        self._thread: threading.Thread | None = None
        self._warning_callback = warning_callback or self._default_warning_callback
        self._closing = False

    @property
    def selected_stage(self) -> int:
        return self._app.stage_selection.selected_stage

    @staticmethod
    def _default_warning_callback(message: str) -> None:
        print(message, file=sys.stderr)

    def _warn(self, message: str) -> None:
        self._warning_callback(f"Warning: {message}")

    def _run_app_in_background_thread(self) -> None:
        """Run Textual in a non-main thread with safe signal registration."""
        original_signal = signal.signal

        def _safe_signal(sig_num, handler):
            try:
                return original_signal(sig_num, handler)
            except ValueError as exc:
                if threading.current_thread() is threading.main_thread():
                    raise
                del sig_num, handler, exc
                return signal.SIG_DFL

        signal.signal = _safe_signal
        try:
            self._app.run(headless=self._headless, inline=self._inline)
        finally:
            signal.signal = original_signal

    def __enter__(self) -> TextualWaveRuntimeSession:
        self._thread = threading.Thread(
            target=self._run_app_in_background_thread,
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=self.READY_TIMEOUT_SECONDS):
            raise RuntimeError("Timed out starting textual progress UI")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        self._closing = True
        if self._thread is not None and self._thread.is_alive():
            try:
                self._app.call_from_thread(self._app.exit)
            except RuntimeError as exc:
                if "App is not running" not in str(exc):
                    self._warn(f"textual runtime shutdown failed: {exc}")
            self._thread.join(timeout=self.READY_TIMEOUT_SECONDS)
        return False

    def _dispatch(self, callback, *args) -> None:
        if self._closing:
            return
        if self._thread is None or threading.current_thread() is self._thread:
            callback(*args)
            return
        if not self._thread.is_alive():
            return
        try:
            self._app.call_from_thread(callback, *args)
        except Exception as exc:
            message = str(exc)
            if "App is not running" in message or "No nodes match" in message:
                return
            self._warn(f"textual runtime dispatch failed: {exc}")

    def update_stage_metrics(self, stage_metrics: list[StageProgressSnapshot]) -> None:
        self._dispatch(self._app.set_stage_metrics, stage_metrics)

    def update_memory_line(self, memory_text: str) -> None:
        self._dispatch(self._app.set_memory_text, memory_text)

    def update_instruction_stats(
        self,
        snapshot_by_stage: dict[int, dict[str, InstructionAttemptStats]],
    ) -> None:
        self._dispatch(self._app.set_instruction_stats_snapshot, snapshot_by_stage)

    def log(self, line: str) -> None:
        self._dispatch(self._app.append_log_line, line)

    def set_status(self, text: str | None) -> None:
        self._dispatch(self._app.set_status_text, text)
