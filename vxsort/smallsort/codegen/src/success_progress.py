from __future__ import annotations

from collections.abc import Callable, Iterable
from time import sleep

from rich.console import Console, RenderableType
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.rule import Rule
from rich.table import Column
from rich.text import Text

from memory_monitor import collect_memory_snapshot

console = Console()


class DualBarColumn(BarColumn):
    """A BarColumn that overlays a 'success' layer (green) over the attempted layer (yellow)."""

    def __init__(
        self,
        bar_width: int = 40,
        attempt_style: str = "yellow",
        success_style: str = "green",
        table_column: Column | None = None,
    ):
        super().__init__(
            bar_width=bar_width,
            complete_style=attempt_style,
            table_column=table_column,
        )
        self.attempt_style = attempt_style
        self.success_style = success_style

        # Pre-compute the composite style for partial success blocks over attempted areas
        bg_color = self.attempt_style.split(" ")[0]
        self.success_on_attempt_style = f"{self.success_style} on {bg_color}"

    def render(self, task):
        # Determine bar width (use provided or fallback)
        total_width = self.bar_width - 2

        # Indeterminate task (total not yet known)
        if task.total is None:
            return Text("[" + " " * total_width + "]", style="dim")

        # Safely get progress numbers
        total = task.total or 1  # avoid division by zero
        attempted_fraction = min(max(task.completed / total, 0.0), 1.0)

        # Get successes from fields (managed by SuccessProgress)
        successes = task.fields.get("successes", 0)
        success_fraction = min(max(successes / total, 0.0), attempted_fraction)

        # Total float widths
        success_v = success_fraction * total_width
        attempt_v = attempted_fraction * total_width

        blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]

        pieces = Text("[")

        for i in range(total_width):
            # Success level in this character (0.0 to 1.0)
            s_level = max(0.0, min(1.0, success_v - i))
            # Attempted level in this character (0.0 to 1.0)
            a_level = max(0.0, min(1.0, attempt_v - i))

            # 1. Full success
            if s_level >= 1.0:
                pieces.append("█", style=self.success_style)
                continue

            # 2. Partial success
            s_idx = int(round(s_level * 8))
            if s_idx > 0:
                # Use pre-computed style to avoid "gaps" if attempt continues
                style = (
                    self.success_on_attempt_style
                    if a_level > s_level
                    else self.success_style
                )
                pieces.append(blocks[s_idx], style=style)
                continue

            # 3. Full attempt (no success in this block)
            if a_level >= 1.0:
                pieces.append("█", style=self.attempt_style)
                continue

            # 4. Partial attempt
            a_idx = int(round(a_level * 8))
            if a_idx > 0:
                pieces.append(blocks[a_idx], style=self.attempt_style)
                continue

            # 5. Unfilled
            pieces.append(" ")

        pieces.append("]")

        assert len(pieces) == self.bar_width, (
            f"Expected width {self.bar_width}, got {len(pieces)} "
            f"for total={task.total}, completed={task.completed}, successes={
                task.fields.get('successes', 0)
            }"
        )

        return pieces


class TqdmColumn(ProgressColumn):
    """A column that displays timing and speed in tqdm style [elapsed<remaining, speed it/s]."""

    def render(self, task):
        elapsed = task.elapsed
        remaining = task.time_remaining
        speed = task.speed or 0

        elapsed_str = self.format_time(elapsed)
        remaining_str = (
            self.format_time(remaining) if remaining is not None else "--:--"
        )

        # Use fixed-width formatting to prevent dancing
        # Speed: pad to 7 chars (e.g., "  0.00", "999.99")
        speed_str = f"{speed:7.2f}"

        return Text(
            f"[{elapsed_str}<{remaining_str}, {speed_str}/s]",
            style="progress.data.speed",
        )

    @staticmethod
    def format_time(seconds: float | None) -> str:
        if seconds is None:
            return "  --:--"
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        # Pad to match HH:MM:SS width (8 chars) for consistency
        return f"  {minutes:02d}:{seconds:02d}"


class AttemptsColumn(ProgressColumn):
    """Render the attempt count (completed units)."""

    def render(self, task):
        return Text(str(int(task.completed)), style="yellow")


class RateColumn(ProgressColumn):
    """Render per-task throughput as items per second."""

    def render(self, task):
        speed = task.speed or 0.0
        return Text(f"{speed:7.2f}/s", style="progress.data.speed")


class SuccessProgress(Progress):
    """A Progress subclass that automatically tracks cumulative successes."""

    def __init__(self, *args, **kwargs):
        self._memory_monitor_enabled = False
        self._status_text: str | None = None
        self._show_table_header: bool = False
        self._extra_renderables_provider: (
            Callable[[], Iterable[RenderableType]] | None
        ) = None
        super().__init__(*args, **kwargs)

    def enable_memory_monitor(self) -> None:
        """Enable the memory usage status line below the progress bars."""
        self._memory_monitor_enabled = True

    def set_status(self, text: str | None) -> None:
        """Set a persistent status line shown below the progress bars."""
        self._status_text = text

    def set_extra_renderables_provider(
        self, provider: Callable[[], Iterable[RenderableType]] | None
    ) -> None:
        """Set a callback that yields additional live renderables."""
        self._extra_renderables_provider = provider

    def get_renderables(self) -> Iterable[RenderableType]:
        """Yield standard progress table, optionally followed by status and memory stats."""
        task_table = self.make_tasks_table(self.tasks)
        if self._show_table_header:
            task_table.show_header = True
        yield task_table
        if self._status_text is not None:
            yield Text(self._status_text, style="bold green")
        if self._extra_renderables_provider is not None:
            yield from self._extra_renderables_provider()
        if self._memory_monitor_enabled:
            snapshot = collect_memory_snapshot()
            if snapshot is not None:
                yield Rule(style="dim")
                yield Text(snapshot.format(), style="dim cyan")

    @staticmethod
    def create(
        description_column: str = "[orange1]{task.description}",
        width: int = 60,
        success_style: str = "green",
        attempt_style: str = "yellow",
        success_label: str = "Valid",
        unique_label: str = "Unique",
        compact_stage_table: bool = False,
    ) -> SuccessProgress:
        """Factory method to create a SuccessProgress with standard columns."""
        if compact_stage_table:
            progress = SuccessProgress(
                TextColumn(
                    "{task.description}",
                    table_column=Column("Stage", style="orange1"),
                ),
                DualBarColumn(
                    bar_width=width,
                    attempt_style=attempt_style,
                    success_style=success_style,
                    table_column=Column("Progress"),
                ),
                AttemptsColumn(table_column=Column("Attempts", justify="right")),
                TextColumn(
                    f"[{success_style}]{{task.fields[successes]}}[/]",
                    table_column=Column(success_label, justify="right"),
                ),
                TextColumn(
                    "[cyan]{task.fields[unique]}[/]",
                    table_column=Column(unique_label, justify="right"),
                ),
                RateColumn(table_column=Column("Rate", justify="right")),
                console=console,
                refresh_per_second=1,
            )
            progress._show_table_header = True
            return progress

        return SuccessProgress(
            TextColumn(description_column),
            TaskProgressColumn(),
            DualBarColumn(
                bar_width=width,
                attempt_style=attempt_style,
                success_style=success_style,
            ),
            MofNCompleteColumn(),
            TextColumn(
                f"[{success_style}]{success_label}: {{task.fields[successes]}}[/]"
            ),
            TextColumn(f"[cyan]{unique_label}: {{task.fields[unique]}}[/]"),
            TqdmColumn(),
            console=console,
            refresh_per_second=1,
        )

    def update(self, task_id, *args, **kwargs):
        success_inc = kwargs.pop("success", 0)
        unique_inc = kwargs.pop("unique", 0)
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.fields["successes"] = task.fields.get("successes", 0) + success_inc
                task.fields["unique"] = task.fields.get("unique", 0) + unique_inc
        super().update(task_id, *args, **kwargs)


# Demo usage
def demo():
    total = 300
    with SuccessProgress(
        TextColumn("[bold blue]{task.description}"),
        TaskProgressColumn(),
        DualBarColumn(bar_width=60, attempt_style="yellow", success_style="green"),
        MofNCompleteColumn(),
        TextColumn("[green]Good: {task.fields[successes]}[/]"),
        TqdmColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("items", total=total, successes=0)

        for i in range(total):
            sleep(0.04)
            # simulate success on 2 of 3 tries
            is_success = i % 3 != 0

            # mark attempt and pass success increment
            progress.update(task_id, advance=1, success=1 if is_success else 0)


if __name__ == "__main__":
    demo()
