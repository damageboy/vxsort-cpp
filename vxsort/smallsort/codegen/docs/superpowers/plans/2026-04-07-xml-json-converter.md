# XML→JSON Instructions Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact, lossless XML↔JSON converter for `instructions.xml` and a strict semantic roundtrip verifier.

**Architecture:** Implement a compact tokenized JSON schema with a global string table and integer IDs. Add a converter module in `src/util/` with CLI subcommands for conversion and roundtrip verification. Add focused tests first (TDD) for schema roundtrip, semantic comparison behavior, and CLI behavior.

**Tech Stack:** Python 3.13, `xml.etree.ElementTree`, `zstandard`, `argparse`, `pytest`.

---

### Task 1: Add failing tests for converter behavior

**Files:**
- Create: `tests/test_xml_json_converter.py`

- [ ] **Step 1: Write failing test for XML→JSON→XML semantic roundtrip**
- [ ] **Step 2: Write failing test for whitespace-formatting tolerance in semantic compare**
- [ ] **Step 3: Write failing test for semantic mismatch detection**
- [ ] **Step 4: Write failing test for `.xml.zst` input support**
- [ ] **Step 5: Write failing test for CLI roundtrip verification exit code**
- [ ] **Step 6: Run `uv run pytest tests/test_xml_json_converter.py -q` and confirm failures**

### Task 2: Implement compact converter module

**Files:**
- Create: `src/util/xml_json_converter.py`

- [ ] **Step 1: Implement compact token schema encoder/decoder (string table + node arrays)**
- [ ] **Step 2: Implement XML loading for `.xml` and `.xml.zst`**
- [ ] **Step 3: Implement JSON writer/reader helpers for compact output**
- [ ] **Step 4: Implement XML reconstruction from JSON tokens**
- [ ] **Step 5: Implement strict semantic comparator (ignore attr order + formatting-only whitespace)**
- [ ] **Step 6: Implement CLI (`convert`, `to-xml`, `roundtrip-verify`)**

### Task 3: Verify tests pass and maintain quality gates

**Files:**
- Modify: `tests/test_xml_json_converter.py` (if needed for fixture polish)
- Modify: `src/util/xml_json_converter.py` (if needed for fixes)

- [ ] **Step 1: Run focused tests: `uv run pytest tests/test_xml_json_converter.py -q`**
- [ ] **Step 2: Run full tests: `uv run pytest`**
- [ ] **Step 3: Run lint: `uv run ruff check .`**
- [ ] **Step 4: Run dead-code scan: `uv run vulture`**
- [ ] **Step 5: Summarize results with concrete command outputs**
