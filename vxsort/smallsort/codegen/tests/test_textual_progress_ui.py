"""Tests for the textual progress UI scaffold."""

import asyncio
import io
import types

import pytest
from rich.console import Console

from instruction_stats import InstructionAttemptStats


def _stats(
    *, total: int, no_valid: int = 0, valid: int = 0, unique: int = 0
) -> InstructionAttemptStats:
    return InstructionAttemptStats(
        attempts_total=total,
        attempts_no_valid=no_valid,
        attempts_valid=valid,
        attempts_unique=unique,
    )


def _render_content(content) -> str:
    console = Console(record=True, width=120, file=io.StringIO())
    console.print(content)
    return console.export_text()


def test_textual_ui_module_importable():
    import textual_progress_ui  # noqa: F401


def test_stage_selection_model_initializes_with_stage_zero():
    from textual_progress_ui import StageSelectionModel

    model = StageSelectionModel(stage_count=4)

    assert model.selected_stage == 0


def test_stage_selection_model_clamps_at_zero_when_moving_left():
    from textual_progress_ui import StageSelectionModel

    model = StageSelectionModel(stage_count=4)

    model.move(-1)

    assert model.selected_stage == 0


def test_stage_selection_model_clamps_at_last_stage_when_moving_right():
    from textual_progress_ui import StageSelectionModel

    model = StageSelectionModel(stage_count=4)

    model.move(10)

    assert model.selected_stage == 3


def test_stage_selection_model_moves_by_one():
    from textual_progress_ui import StageSelectionModel

    model = StageSelectionModel(stage_count=4, selected_stage=1)

    model.move(1)

    assert model.selected_stage == 2


def test_focus_model_toggles_between_stage_and_stats():
    from textual_progress_ui import FocusModel

    model = FocusModel()

    assert model.current == "stage"

    assert model.toggle() == "stats"
    assert model.current == "stats"

    assert model.toggle() == "stage"
    assert model.current == "stage"


def test_focus_model_rejects_invalid_assignment():
    from textual_progress_ui import FocusModel

    model = FocusModel()

    with pytest.raises(ValueError, match="current must be 'stage' or 'stats'"):
        model.current = "log"


def test_wave_progress_app_uses_up_down_stage_bindings():
    from textual_progress_ui import WaveProgressApp

    def has_binding(key: str, action: str) -> bool:
        return any(
            getattr(binding, "key", None) == key
            and getattr(binding, "action", None) == action
            and getattr(binding, "priority", False) is True
            for binding in WaveProgressApp.BINDINGS
        )

    assert has_binding("tab", "toggle_focus")
    assert has_binding("up", "previous_stage")
    assert has_binding("down", "next_stage")
    assert has_binding("pageup", "page_up")
    assert has_binding("pagedown", "page_down")
    assert has_binding("home", "jump_top")
    assert has_binding("end", "jump_bottom")
    assert has_binding("s", "focus_stats")
    assert has_binding("g", "focus_stage")
    assert not has_binding("left", "previous_stage")
    assert not has_binding("right", "next_stage")


def test_wave_progress_app_layout_exposes_expected_regions():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        app = WaveProgressApp(stage_count=4)
        async with app.run_test():
            assert app.query_one("#top-region") is not None
            assert app.query_one("#stage-pane") is not None
            assert app.query_one("#instruction-stats-pane") is not None
            assert app.query_one("#top-bottom-separator") is not None
            assert app.query_one("#log-pane") is not None
            assert app.query_one("#status-line") is not None

    asyncio.run(run())


def test_wave_progress_app_q_triggers_quit_callback():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        calls: list[str] = []
        app = WaveProgressApp(
            stage_count=2, on_quit_requested=lambda: calls.append("q")
        )
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause()
        assert calls == ["q"]

    asyncio.run(run())


def test_wave_progress_app_tab_toggles_focus_and_routes_up_down_by_pane():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        stats_rows = {
            f"inst_{idx:02d}": _stats(total=4, unique=max(0, 20 - idx))
            for idx in range(20)
        }
        app = WaveProgressApp(
            stage_count=4,
            snapshot_by_stage={0: stats_rows},
        )
        async with app.run_test() as pilot:
            stats_pane = app.query_one("#instruction-stats-pane")

            assert app.focus_model.current == "stage"
            assert app.focused is app.query_one("#stage-pane")

            await pilot.press("down")
            assert app.stage_selection.selected_stage == 1

            await pilot.press("up")
            assert app.stage_selection.selected_stage == 0

            stats_before = _render_content(stats_pane.content)

            await pilot.press("s")
            assert app.focus_model.current == "stats"
            assert app.focused is stats_pane

            await pilot.press("down")
            assert app.stage_selection.selected_stage == 0

            stats_after_down = _render_content(stats_pane.content)
            assert stats_after_down != stats_before

            await pilot.press("up")
            assert _render_content(stats_pane.content) == stats_before

            await pilot.press("g")
            assert app.focus_model.current == "stage"
            assert app.focused is app.query_one("#stage-pane")

    asyncio.run(run())


def test_wave_progress_app_external_focus_change_syncs_model_and_routing():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        app = WaveProgressApp(
            stage_count=4,
            snapshot_by_stage={
                0: {
                    "alpha": _stats(total=4, unique=3),
                    "bravo": _stats(total=4, unique=2),
                    "charlie": _stats(total=4, unique=1),
                    "delta": _stats(total=4, unique=0, valid=4),
                }
            },
        )
        async with app.run_test() as pilot:
            stats_pane = app.query_one("#instruction-stats-pane")

            assert app.focus_model.current == "stage"
            assert app.focused is app.query_one("#stage-pane")

            stats_before = str(stats_pane.content)
            app.set_focus(stats_pane, scroll_visible=False)
            await pilot.pause()

            assert app.focused is stats_pane
            assert app.focus_model.current == "stats"

            await pilot.press("down")

            assert app.stage_selection.selected_stage == 0
            assert str(stats_pane.content) != stats_before

    asyncio.run(run())


def test_wave_progress_app_stats_page_navigation_scrolls_and_home_resets():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        stats_rows = {
            f"inst_{idx:03d}": _stats(total=7, valid=3, unique=max(0, 60 - idx))
            for idx in range(60)
        }
        app = WaveProgressApp(stage_count=2, snapshot_by_stage={0: stats_rows})
        async with app.run_test() as pilot:
            stats_pane = app.query_one("#instruction-stats-pane")
            top_text = _render_content(stats_pane.content)

            await pilot.press("s")
            await pilot.press("pagedown")
            page_text = _render_content(stats_pane.content)

            assert app.stage_selection.selected_stage == 0
            assert page_text != top_text

            await pilot.press("home")
            assert _render_content(stats_pane.content) == top_text

    asyncio.run(run())


def test_wave_progress_app_log_pane_is_not_focusable_and_cannot_steal_keys():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        stats_rows = {
            f"inst_{idx:03d}": _stats(total=3, valid=1, unique=max(0, 40 - idx))
            for idx in range(40)
        }
        app = WaveProgressApp(stage_count=2, snapshot_by_stage={0: stats_rows})
        async with app.run_test() as pilot:
            stats_pane = app.query_one("#instruction-stats-pane")
            log_pane = app.query_one("#log-pane")

            await pilot.press("s")
            before = _render_content(stats_pane.content)

            app.set_focus(log_pane, scroll_visible=False)
            await pilot.pause()
            assert app.focused is not log_pane

            await pilot.press("down")
            after = _render_content(stats_pane.content)
            assert after != before

    asyncio.run(run())


def test_wave_progress_app_instruction_rows_include_full_stage_list():
    from textual_progress_ui import WaveProgressApp

    app = WaveProgressApp(
        stage_count=1,
        instruction_top_k=2,
        snapshot_by_stage={
            0: {
                "alpha": _stats(total=0),
                "bravo": _stats(total=0),
                "charlie": _stats(total=0),
                "delta": _stats(total=0),
            }
        },
    )

    rows = app._instruction_rows_for_selected_stage()
    assert [row.key for row in rows] == ["alpha", "bravo", "charlie", "delta"]


def test_select_stage_rows_ignores_top_k_and_returns_full_stage():
    from textual_progress_ui import select_stage_rows

    rows = select_stage_rows(
        snapshot_by_stage={
            0: {
                "alpha": _stats(total=0),
                "bravo": _stats(total=0),
                "charlie": _stats(total=0),
                "delta": _stats(total=0),
            }
        },
        selected_stage=0,
        top_k=1,
    )

    assert [row.key for row in rows] == ["alpha", "bravo", "charlie", "delta"]


def test_select_stage_rows_includes_arch_latency_values_in_cli_order():
    from textual_progress_ui import select_stage_rows

    rows = select_stage_rows(
        snapshot_by_stage={
            0: {
                "alpha": _stats(total=2, valid=1),
                "bravo": _stats(total=1, unique=1),
            }
        },
        selected_stage=0,
        top_k=10,
        latency_arches=["ZEN4", "ADL-P"],
        latency_by_key={
            "alpha": {"ZEN4": "3", "ADL-P": "4"},
            "bravo": {"ZEN4": "-", "ADL-P": "2"},
        },
    )

    assert [row.key for row in rows] == ["bravo", "alpha"]
    assert rows[0].arch_latencies == ("-", "2")
    assert rows[1].arch_latencies == ("3", "4")


def test_wave_progress_app_instruction_window_rows_uses_top_panel_height():
    from textual_progress_ui import WaveProgressApp

    app = WaveProgressApp(stage_count=1)
    app._top_panel_height = 18

    assert app._instruction_window_rows() == 14


def test_wave_progress_app_instruction_table_renders_arch_latency_columns():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        app = WaveProgressApp(
            stage_count=1,
            snapshot_by_stage={
                0: {
                    "alpha": _stats(total=2, valid=1),
                }
            },
            instruction_latency_arches=["ZEN4", "ADL-P"],
            instruction_latency_by_key={
                "alpha": {"ZEN4": "3", "ADL-P": "4"},
            },
        )
        async with app.run_test():
            pane_text = _render_content(
                app.query_one("#instruction-stats-pane").content
            )
            assert "ZEN4" in pane_text
            assert "ADL-P" in pane_text
            assert "alpha" in pane_text
            assert "3c" in pane_text
            assert "4c" in pane_text

    asyncio.run(run())


def test_wave_progress_app_formats_uncovered_input_percentage():
    from textual_progress_ui import StageProgressSnapshot, WaveProgressApp

    app = WaveProgressApp(stage_count=2)
    snapshot = StageProgressSnapshot(
        stage_idx=1,
        attempts=10,
        total=10,
        valid=3,
        unique=2,
        covered_inputs=3,
        total_inputs=12,
        uncovered_input_pct=75.0,
    )

    assert app._format_uncovered_input(snapshot) == "75%"


def test_wave_progress_app_formats_uncovered_input_percentage_two_decimals_max():
    from textual_progress_ui import StageProgressSnapshot, WaveProgressApp

    app = WaveProgressApp(stage_count=2)
    snapshot = StageProgressSnapshot(
        stage_idx=1,
        attempts=10,
        total=10,
        valid=3,
        unique=2,
        covered_inputs=1,
        total_inputs=3,
        uncovered_input_pct=66.6666666667,
    )

    assert app._format_uncovered_input(snapshot) == "66.67%"


def test_wave_progress_app_stage_navigation_refreshes_selected_stage_stats():
    from textual_progress_ui import WaveProgressApp

    async def run() -> None:
        app = WaveProgressApp(
            stage_count=2,
            snapshot_by_stage={
                0: {"alpha": _stats(total=3, unique=2)},
                1: {"beta": _stats(total=5, valid=5)},
            },
        )
        async with app.run_test() as pilot:
            stats_pane = app.query_one("#instruction-stats-pane")

            pane_text = _render_content(stats_pane.content)
            assert "alpha" in pane_text
            assert "beta" not in pane_text

            await pilot.press("down")

            assert app.stage_selection.selected_stage == 1
            pane_text = _render_content(stats_pane.content)
            assert "beta" in pane_text
            assert "alpha" not in pane_text

    asyncio.run(run())


def test_wave_progress_app_background_updates_preserve_selected_stage():
    from textual_progress_ui import StageProgressSnapshot, WaveProgressApp

    async def run() -> None:
        app = WaveProgressApp(stage_count=3)
        async with app.run_test() as pilot:
            await pilot.press("down", "down")

            app.set_stage_metrics(
                [
                    StageProgressSnapshot(
                        stage_idx=0,
                        attempts=4,
                        total=5,
                        valid=1,
                        unique=1,
                    ),
                    StageProgressSnapshot(
                        stage_idx=1,
                        attempts=7,
                        total=8,
                        valid=2,
                        unique=1,
                    ),
                    StageProgressSnapshot(
                        stage_idx=2,
                        attempts=9,
                        total=9,
                        valid=3,
                        unique=2,
                    ),
                ]
            )
            app.set_instruction_stats_snapshot(
                {
                    2: {
                        "gamma": _stats(total=5, unique=2),
                    }
                }
            )
            await pilot.pause()

            assert app.stage_selection.selected_stage == 2
            pane_text = _render_content(
                app.query_one("#instruction-stats-pane").content
            )
            assert "gamma" in pane_text

    asyncio.run(run())


def test_wave_progress_app_status_text_mentions_up_down_navigation():
    from textual_progress_ui import WaveProgressApp

    app = WaveProgressApp(stage_count=4)

    status_text = app._status_line_text()

    assert "Tab toggle focus" in status_text
    assert "s stats | g stage" in status_text
    assert "Up/Down + PgUp/PgDn + Home/End" in status_text
    assert "Left/Right select stage" not in status_text


def test_wave_progress_app_status_text_uses_memory_snapshot(monkeypatch):
    import textual_progress_ui

    class StubSnapshot:
        def format(self) -> str:
            return "Memory: Main 10 MB | Workers: none | Total 10 MB"

    monkeypatch.setattr(
        textual_progress_ui, "collect_memory_snapshot", lambda: StubSnapshot()
    )
    app = textual_progress_ui.WaveProgressApp(stage_count=4)

    status_text = app._status_line_text()

    assert "Memory: Main 10 MB | Workers: none | Total 10 MB" in status_text


def test_wave_progress_app_status_text_handles_missing_memory_snapshot(monkeypatch):
    import textual_progress_ui

    monkeypatch.setattr(textual_progress_ui, "collect_memory_snapshot", lambda: None)
    app = textual_progress_ui.WaveProgressApp(stage_count=4)

    status_text = app._status_line_text()

    assert "Memory: unavailable" in status_text


def test_wave_progress_app_rate_uses_per_stage_attempt_deltas(monkeypatch):
    import textual_progress_ui

    times = iter([100.0, 101.0, 101.5])
    monkeypatch.setattr(textual_progress_ui.time, "monotonic", lambda: next(times))

    app = textual_progress_ui.WaveProgressApp(stage_count=1)
    app.set_stage_metrics(
        [
            textual_progress_ui.StageProgressSnapshot(
                stage_idx=0,
                attempts=0,
                total=0,
                valid=0,
                unique=0,
            )
        ]
    )
    app.set_stage_metrics(
        [
            textual_progress_ui.StageProgressSnapshot(
                stage_idx=0,
                attempts=10,
                total=10,
                valid=4,
                unique=2,
            )
        ]
    )
    app.set_stage_metrics(
        [
            textual_progress_ui.StageProgressSnapshot(
                stage_idx=0,
                attempts=10,
                total=10,
                valid=4,
                unique=2,
            )
        ]
    )

    assert app._rate_by_stage[0] == pytest.approx(10.0)


def test_vertical_resize_handle_ignores_non_primary_mouse_down():
    from textual_progress_ui import VerticalResizeHandle

    class FakeEvent:
        button = 3
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    handle = VerticalResizeHandle()
    captured = []
    handle.capture_mouse = lambda capture=True: captured.append(capture)

    event = FakeEvent()
    handle.on_mouse_down(event)

    assert handle._dragging is False
    assert captured == []
    assert event.stopped is False


def test_textual_runtime_session_dispatch_runtime_error_emits_warning():
    from textual_progress_ui import TextualWaveRuntimeSession

    warnings: list[str] = []

    class FakeApp:
        def __init__(self, **_kwargs) -> None:
            self.stage_selection = types.SimpleNamespace(selected_stage=0)

        def call_from_thread(self, callback, *args, **kwargs) -> None:
            del callback, args, kwargs
            raise RuntimeError("app thread unavailable")

        def append_log_line(self, line: str) -> None:
            del line

    class FakeThread:
        def is_alive(self) -> bool:
            return True

    session = TextualWaveRuntimeSession(
        stage_count=1,
        app_factory=FakeApp,
        warning_callback=warnings.append,
    )
    session._thread = FakeThread()

    session.log("hello")

    assert warnings == [
        "Warning: textual runtime dispatch failed: app thread unavailable"
    ]


def test_textual_runtime_session_shutdown_runtime_error_emits_warning():
    from textual_progress_ui import TextualWaveRuntimeSession

    warnings: list[str] = []
    join_calls: list[float] = []

    class FakeApp:
        def __init__(self, **_kwargs) -> None:
            self.stage_selection = types.SimpleNamespace(selected_stage=0)
            self.exit = lambda: None

        def call_from_thread(self, callback, *args, **kwargs) -> None:
            del callback, args, kwargs
            raise RuntimeError("app already stopped")

    class FakeThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            join_calls.append(timeout if timeout is not None else -1.0)

    session = TextualWaveRuntimeSession(
        stage_count=1,
        app_factory=FakeApp,
        warning_callback=warnings.append,
    )
    session._thread = FakeThread()

    assert session.__exit__(None, None, None) is False

    assert warnings == ["Warning: textual runtime shutdown failed: app already stopped"]
    assert join_calls == [session.READY_TIMEOUT_SECONDS]
