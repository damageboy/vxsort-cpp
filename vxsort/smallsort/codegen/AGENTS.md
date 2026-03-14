# Repository Guidelines

## Project Structure & Module Organization
This directory (`vxsort/smallsort/codegen`) contains a Python super-optimizer for bitonic sorter generation.

- `src/`: core implementation (`bitonic_super_optimizer.py`, `bitonic_compiler.py`, `z3_avx.py`, exporters, cost model).
- `tests/`: pytest suite plus JSON fixtures (for example `tests/fixture_2xAVX2_i64.json`).
- `docs/plans/`: design and implementation notes for larger features.
- Repository root artifacts: generated outputs like `bitonic_solutions_*.json` and `bitonic_solutions_*.asm`.

Do not hand-edit generated `.json`/`.asm` files; regenerate them via the compiler flow.

## Build, Test, and Development Commands
Use `uv` for all Python environment and command execution.

- `uv sync`: install/update dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run default test suite (excludes `slow` tests).
- `uv run pytest -m slow`: run long Z3 proof tests.
- `uv run ruff check .`: lint and style checks.
- `uv run vulture`: detect unused code.
- `uv run python src/bitonic_compiler.py --depth-limit=3 --gadget-depth 1 --vector-machine AVX2 --datatype i64`: run a bounded synthesis pass for local validation (keep testing limited to AVX2/i64 and gadget-depth 1 to keep things fast).

## Coding Style & Naming Conventions
- Python 3.13, 4-space indentation, PEP 8-compatible formatting.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep type hints on public functions and non-trivial internal interfaces.
- Prefer focused, testable functions over large procedural blocks.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml` with `testpaths = ["tests"]`).
- Test files follow `test_*.py`; test classes use `Test...`; test names describe behavior.
- Mark expensive solver proofs with `@pytest.mark.slow`.
- For bug fixes, add a regression test and run at least `uv run pytest` before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit-style prefixes: `feat:`, `fix:`, `refactor:`, `chore:`.

- Keep commits scoped to one logical change.
- Use imperative, concise commit summaries (for example `feat: add stage-level checkpoint resume`).
- PRs should include: problem statement, approach summary, validation commands run, and impact on generated outputs/performance.
- If synthesis output changes, include the updated generated artifacts and note why they changed.
