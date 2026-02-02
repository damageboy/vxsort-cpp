from time import sleep
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.text import Text

console = Console()


class DualBarColumn(BarColumn):
    """A BarColumn that overlays a 'success' layer (green) over the attempted layer (yellow)."""

    def __init__(
        self,
        bar_width: int | None = None,
        attempt_style: str = "yellow",
        success_style: str = "green",
    ):
        super().__init__(bar_width=bar_width, complete_style=attempt_style)
        self.attempt_style = attempt_style
        self.success_style = success_style

    def render(self, task):
        # Determine bar width (use provided or fallback)
        total_width = self.bar_width or 40

        # Safely get progress numbers
        total = task.total or 1  # avoid division by zero
        attempted_fraction = min(max(task.completed / total, 0.0), 1.0)
        success_fraction = float(task.fields.get("success_ratio", 0.0))
        success_fraction = min(
            max(success_fraction, 0.0), attempted_fraction
        )  # success <= attempted

        # Compute segment widths
        success_width = int(total_width * success_fraction)
        attempt_width = int(total_width * attempted_fraction)
        attempted_only_width = max(attempt_width - success_width, 0)
        remainder_width = max(total_width - attempt_width, 0)

        # Build Text pieces with styles so Rich renders them correctly inside Progress
        pieces = Text()
        if success_width:
            pieces.append("█" * success_width, style=self.success_style)
        if attempted_only_width:
            pieces.append("█" * attempted_only_width, style=self.attempt_style)
        if remainder_width:
            pieces.append(" " * remainder_width)  # unfilled area

        return pieces


# Demo usage
def demo():
    total = 300
    with Progress(
        TextColumn("[bold blue]Processing[/]"),
        DualBarColumn(bar_width=40, attempt_style="yellow", success_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("items", total=total, success_ratio=0.0)

        success = 0
        for i in range(total):
            sleep(0.01)
            # mark attempt
            progress.update(task_id, advance=1)

            # simulate success on 2 of 3 tries
            if i % 3 != 0:
                success += 1

            # update custom success_ratio field (relative to total)
            progress.update(task_id, success_ratio=success / total)


if __name__ == "__main__":
    demo()
