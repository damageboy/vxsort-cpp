#!/usr/bin/env bash
set -euo pipefail

# Reproducible audit for UI transition cleanup.
# 1) Collect coverage from targeted UI tests.
# 2) Append coverage from a real textual solve run.
# 3) Report module-level coverage for UI-related modules.
# 4) Run vulture over UI modules for dead-code hints.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[ui-audit] Erasing old coverage data"
uv run coverage erase

echo "[ui-audit] Running targeted UI tests with coverage"
uv run pytest \
  tests/test_textual_progress_ui.py \
  tests/test_wave_engine.py \
  tests/test_instruction_stats.py \
  --cov=textual_progress_ui \
  --cov=wave_engine \
  --cov=instruction_stats \
  --cov=success_progress \
  --cov=bitonic_compiler \
  --cov-report=term-missing \
  -q

echo "[ui-audit] Running minimal textual solve and appending coverage"
uv run coverage run --append \
  --source=bitonic_compiler,success_progress,wave_engine,textual_progress_ui,instruction_stats \
  src/bitonic_compiler.py \
  --num-vecs 2 \
  --vector-machine AVX2 \
  --datatype i64 \
  --gadget-depth 1 \
  --depth-limit 1 \
  --max-waves 1 \
  --wave-attempts 100 \
  --wave-outputs 20 \
  --runtime-ui textual

echo "[ui-audit] Combined coverage report"
uv run coverage report -m

echo "[ui-audit] Dead-code hints (vulture)"
set +e
uv run vulture src/textual_progress_ui.py src/wave_engine.py src/success_progress.py src/bitonic_compiler.py src/instruction_stats.py
VULTURE_EXIT=$?
set -e
if [[ ${VULTURE_EXIT} -ne 0 ]]; then
  echo "[ui-audit] vulture returned ${VULTURE_EXIT} (informational findings above)"
fi
