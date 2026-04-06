# CodeTour Skill Design

## Overview

A single Claude Code skill (`codetour`) that generates `.tour` files for the
[CodeTour VS Code extension](https://marketplace.visualstudio.com/items?itemName=vsls-contrib.codetour).
Three modes, one entry point:

| Mode | Trigger | Purpose |
|------|---------|---------|
| **Onboarding** | `/codetour` (no args, no uncommitted changes) | Architectural overview of the repo |
| **Exploration** | `/codetour <path\|concept>` | Targeted tour of a specific area or flow |
| **Review** | `/codetour <commit-range\|--branch>` or post-implementation offer | Didactic walkthrough of changes grounded in the implementation plan |

## Routing Logic

```
/codetour                              → check for uncommitted changes
  ├─ non-empty `git diff HEAD`          → ask: "Review tour or onboarding tour?"
  └─ clean working tree                 → Onboarding mode

/codetour src/auth/                    → Exploration mode (path)
/codetour "authentication flow"        → Exploration mode (concept — agent finds files)
/codetour abc123..def456               → Review mode (commit range)
/codetour --branch feature/foo         → Review mode (branch diff vs main)
```

**Argument disambiguation:** When a bare argument like `/codetour foo` is given,
resolve in this order:
1. Check if it's a valid path in the working tree → Exploration mode
2. Check if it resolves as a git ref (`git rev-parse`) → Review mode
3. Treat as a concept/keyword → Exploration mode (agent searches for relevant files)

**"Meaningful changes"**: defined as non-empty output from `git diff HEAD`
(covers both staged and unstaged changes). Untracked files alone do not trigger
the review tour prompt.

Post-implementation: the skill does NOT auto-run. Other skills (e.g.,
`finishing-a-development-branch`, `executing-plans`) suggest `/codetour` to the
user when a changeset spans 3+ files or multiple logical concerns.

## Existing Tour Discovery

Before generating, the skill:

1. Globs for `.tours/**/*.tour`
2. If tours exist, lists them (title + description)
3. Presents options to the user:
   - **Create new** — add alongside existing tours
   - **Overwrite `<tour-name>`** — replace a specific tour
   - **Reference** — chain via `nextTour` to an existing tour
   - **Abort** — cancel if coverage is sufficient
4. If no tours exist, proceeds directly to generation

## Analysis & Step Generation Pipeline

All modes share this pipeline:

### 1. Gather Source Material

| Mode | Sources |
|------|---------|
| Onboarding | Repo structure, entry points, README, CLAUDE.md, directory layout |
| Exploration | Files in target area, imports/dependencies, call flows |
| Review | Git diff (scoped by context) + plan document (see below) |

**Plan document sourcing (review mode):** The skill looks for plan context in
this order: (1) plan document active in the current session context, (2) files
matching the branch name or feature in `docs/plans/` or `docs/specs/`,
(3) commit messages in the diff range. If no plan is found, the skill degrades
gracefully to a diff-only walkthrough — prose explains *what* changed rather
than *why* per the plan.

### 2. Identify Logical Concerns

Group material into coherent themes:
- **Review mode**: group changed files by *what they accomplish together*, not by directory
- **Onboarding mode**: identify major subsystems or architectural layers
- **Exploration mode**: identify components and their interactions within the target area

**Target step counts:** Onboarding: 8-15 steps. Exploration: 5-12 steps.
Review: 1 step per logical concern, typically 3-10.

### 3. Order Didactically

Arrange concerns for learning progression — dependencies before dependents,
context before detail. Not alphabetical, not by directory.

### 4. Pick Representative Location per Concern

For each logical concern, select the single file + location that best illustrates it.

- **Review tours**: prefer `pattern` (regex) over `line` for resilience to future edits
- **Onboarding/exploration tours**: `line` is acceptable since the code is stable
- Use `selection` ranges when a specific code span is more illustrative than a
  single line. **Note:** the schema documents `selection` positions as 1-based,
  but the VS Code runtime uses 0-based characters internally — emit 1-based per
  the schema and let the extension handle translation
- The skill does NOT generate `view`-type steps (pointing to VS Code panels like
  terminal or SCM). All steps point to files or directories in the codebase

### 5. Write Plan-Informed Prose

Each step's `description` uses full markdown. Content varies by mode:

**Review tour steps:**
- *What* this concern addresses (from the plan's intent)
- *Why* this change was made (from the plan's reasoning)
- *How* to evaluate it (what a reviewer should look for)

**Onboarding/exploration tour steps:**
- *What* this component/area does
- *Why* it matters in the larger architecture
- *How* it connects to adjacent components

### 6. Leverage CodeTour-Flavored Markdown

The skill should use CodeTour's extended markdown where appropriate:

| Feature | Syntax | Use case |
|---------|--------|----------|
| Step references | `[#2]` | "As we saw in [#2], the config is loaded here..." |
| Cross-tour refs | `[Tour Name#3]` | When chaining tours |
| File links | `[Open file](./path)` | Point to related files not in a step |
| Shell commands | `>> npm run test` | Suggest commands the reviewer can run |
| Code blocks | ` ```lang ``` ` | Renders "Insert Code" link for suggested fixes |

### 7. Emit `.tour` JSON

Assemble steps into a valid `.tour` file per the
[CodeTour schema](https://github.com/microsoft/codetour/blob/main/schema.json):

**Root properties:**
- `title` (required)
- `description`
- `ref` — set for review tours. Use the branch name (not commit SHA) so the tour
  remains associated with the branch during review; the branch name is more
  meaningful to reviewers than a SHA
- `isPrimary` — set for onboarding tours if no primary exists
- `nextTour` — set if user chose to chain
- `steps` (required)

**Step properties:**
- `title` — short name for the tree view
- `description` (required) — markdown prose
- `file` or `directory` — relative path from workspace root
- `line` or `pattern` — location within file. If both are present, `line` takes
  precedence and `pattern` is ignored. Prefer `pattern` alone for review tours
- `selection` — optional start/end range for highlighting

## Output

### Directory Structure

```
.tours/
  onboarding/          # Repo and area onboarding tours
  explore/             # Targeted exploration tours
  reviews/             # Review tours, pinned to a ref
```

### Naming Convention

| Mode | Pattern | Example |
|------|---------|---------|
| Onboarding | `onboarding-<repo-name>.tour` | `onboarding-vxsort.tour` |
| Exploration | `explore-<area>.tour` | `explore-codegen-pipeline.tour` |
| Review | `review-<branch-or-summary>.tour` | `review-wave-engine.tour` |

### Post-Generation

1. Show summary: tour title, step count, logical concerns covered
2. Ask user for commit approval (per existing preference: never commit without asking)
3. Suggest: "Open in VS Code with CodeTour extension to walk through it"

## Proactive Integration

The skill's description includes triggers like "Use after completing implementation
work" so that completion-phase skills discover it naturally. No edits to other
skills required.

**Guidance for skill authors:** After implementation is complete and verified,
suggest `/codetour` to the user if the changeset spans 3+ files or multiple
logical concerns.

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Path argument doesn't exist | Report error, suggest similar paths if possible |
| Commit range is invalid or empty | Report error, show recent branches/tags as alternatives |
| Concept search finds no relevant files | Report what was searched, ask user to narrow or rephrase |
| Repo has no git history | Onboarding mode only; review mode is unavailable |
| Diff is empty (no changes in range) | Report "no changes found," abort |

## .tour File Reference

Authoritative schema: https://github.com/microsoft/codetour/blob/main/schema.json

### Root Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `title` | string | yes | Display name shown in CodeTour tree view |
| `description` | string | no | Tooltip text for the tour |
| `ref` | string | no | Git ref (branch/commit/tag) the tour applies to |
| `isPrimary` | boolean | no | Primary tour for the codebase |
| `steps` | array | yes | Array of step objects |
| `stepMarker` | string | no | Marker text in code comments indicating steps |
| `nextTour` | string | no | Title of the next tour in sequence |
| `when` | string | no | JS expression for conditional visibility |

### Step Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `description` | string | yes | Markdown text explaining the step |
| `title` | string | no | Display name in tree view |
| `file` | string | no | Relative file path from workspace root |
| `directory` | string | no | Relative directory path (takes precedence over `file`) |
| `uri` | string | no | Absolute URI (mutually exclusive with `file`) |
| `view` | string | no | VS Code view ID to focus |
| `line` | number | no | 1-based line number |
| `pattern` | string | no | Regex for line matching (only when `line` is absent) |
| `selection` | object | no | `{start: {line, character}, end: {line, character}}` (1-based) |
| `commands` | array | no | VS Code command URIs to execute on navigation |

### Supported Views

`debug`, `debug:breakpoints`, `debug:callstack`, `debug:variables`, `debug:watch`,
`explorer`, `extensions`, `extensions:disabled`, `extensions:enabled`, `output`,
`problems`, `scm`, `search`, `terminal`

### Tour Storage Locations

CodeTour discovers tours in: `.tours/`, `.vscode/tours/`, `.github/tours/`,
root-level `.tour` or `main.tour`, `.vscode/main.tour`. Subdirectories within
`.tours/` are supported.
