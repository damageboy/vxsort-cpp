# Claude Code Rules - Codegen

Guidelines for Claude Code when working in this directory.

## Python Package Management

**Always use `uv` for Python package management.**

### Package Management Commands

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

### Running Python Code

- Run a Python script: `uv run <script-name>.py`
- Run Python tools: `uv run pytest`, `uv run ruff`, etc.
- Launch a Python REPL: `uv run python`

**Never use pip, pip-tools, poetry, or conda directly for dependency management.**
