from src.success_progress import DualBarColumn
from unittest.mock import MagicMock


def test_dual_bar_column_width():
    bar_width = 50
    column = DualBarColumn(bar_width=bar_width)

    # Mock task object
    task = MagicMock()
    task.total = 100
    task.completed = 50
    task.fields = {"successes": 25}

    rendered = column.render(task)
    assert len(rendered) == bar_width
    assert str(rendered).startswith("[")
    assert str(rendered).endswith("]")


def test_dual_bar_column_edge_cases():
    bar_width = 30
    column = DualBarColumn(bar_width=bar_width)

    test_cases = [
        (100, 0, 0),  # Zero progress
        (100, 100, 100),  # Full success
        (100, 100, 0),  # Full attempt, zero success
        (100, 50, 50),  # Half progress, all success
        (100, 50, 25),  # Half progress, half success
        (1, 1, 1),  # Small total
        (1, 0, 0),  # Small total zero progress
        (100, 101, 100),  # Over progress (should be clamped)
        (100, 50, 60),  # Success > attempt (should be clamped)
    ]

    for total, completed, successes in test_cases:
        task = MagicMock()
        task.total = total
        task.completed = completed
        task.fields = {"successes": successes}

        rendered = column.render(task)
        assert (
            len(rendered) == bar_width
        ), f"Failed for total={total}, completed={completed}, successes={successes}"


def test_dual_bar_column_division_by_zero():
    bar_width = 40
    column = DualBarColumn(bar_width=bar_width)

    task = MagicMock()
    task.total = 0  # Should be handled by task.total or 1
    task.completed = 0
    task.fields = {}

    rendered = column.render(task)
    assert len(rendered) == bar_width


def test_dual_bar_column_large_numbers():
    bar_width = 60
    column = DualBarColumn(bar_width=bar_width)

    test_cases = [
        (10000, 5000, 2500),
        (50000, 40000, 30000),
        (100000, 99999, 50000),
        (1000000, 500000, 250000),
        (1694, 1, 0),
    ]

    for total, completed, successes in test_cases:
        task = MagicMock()
        task.total = total
        task.completed = completed
        task.fields = {"successes": successes}

        rendered = column.render(task)
        assert (
            len(rendered) == bar_width
        ), f"Failed for large numbers: total={total}, completed={completed}, successes={successes}"
