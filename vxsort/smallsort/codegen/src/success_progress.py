from __future__ import annotations
from time import sleep
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    ProgressColumn,
)
from rich.text import Text

console = Console()


class DualBarColumn(BarColumn):
    """A BarColumn that overlays a 'success' layer (green) over the attempted layer (yellow)."""

    def __init__(
        self,
        bar_width: int = 40,
        attempt_style: str = "yellow",
        success_style: str = "green",
    ):
        super().__init__(bar_width=bar_width, complete_style=attempt_style)
        self.attempt_style = attempt_style
        self.success_style = success_style

        # Pre-compute the composite style for partial success blocks over attempted areas
        bg_color = self.attempt_style.split(" ")[0]
        self.success_on_attempt_style = f"{self.success_style} on {bg_color}"

    def render(self, task):
        # Determine bar width (use provided or fallback)
        total_width = self.bar_width - 2

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
            f"for total={task.total}, completed={task.completed}, successes={task.fields.get('successes', 0)}"
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


class SuccessProgress(Progress):
    """A Progress subclass that automatically tracks cumulative successes."""

    @staticmethod
    def create(
        description_column: str = "[orange1]{task.description}",
        width: int = 60,
        success_style: str = "green",
        attempt_style: str = "yellow",
        success_label: str = "Good",
    ) -> SuccessProgress:
        """Factory method to create a SuccessProgress with standard columns."""
        return SuccessProgress(
            TextColumn(description_column),
            TaskProgressColumn(),
            DualBarColumn(
                bar_width=width,
                attempt_style=attempt_style,
                success_style=success_style,
            ),
            MofNCompleteColumn(),
            # Use success_label for the "Good" count
            TextColumn(
                f"[{success_style}]{success_label}: {{task.fields[successes]}}[/]"
            ),
            TqdmColumn(),
            console=console,
        )

    def update(self, task_id, *args, **kwargs):
        success_inc = kwargs.pop("success", 0)
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.fields["successes"] = task.fields.get("successes", 0) + success_inc
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
