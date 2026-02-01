# Developer Guidelines

## 1. Branch Naming Convention

**Format:** `type/short-description`

### Common Types

* **`feature/`**: Used for developing new features.
    * Example: `feature/user-login`, `feature/data-visualization`
* **`fix/`**: Used for fixing bugs.
    * Example: `fix/login-error`, `fix/typo-in-readme`
* **`docs/`**: Changes involving documentation only (e.g., README, Wiki).
    * Example: `docs/update-api-guide`, `docs/dataset-guidelines`
* **`chore/`**: Changes to the build process, configuration, or auxiliary tools; no source code changes.
    * Example: `chore/update-dependencies`, `chore/folder-structure-setup`
* **`refactor/`**: Code refactoring (neither a new feature nor a bug fix).
    * Example: `refactor/simplify-data-loader`

---
## 2. Commit Message Template

To ensure our commit history looks like the structure below (Title + Description), please follow this template.

**Structure:**
```text
<type>: <short summary>

<Body: Detailed explanation of what changed and why.>
```
---
## 3. Pull Request Convention

### Title
**Format:** `[Type]: Short description`

**Examples:**
* `[Docs]: Add dataset organization guidelines and dictionary demo`
* `[Feature]: Implement user authentication flow`
* `[Fix]: Resolve crash when loading empty CSV files`
* `[Chore]: Setup initial folder structure`

### Body/Content (Template)

Please copy and paste the following structure into your Pull Request description:

```markdown
## Description
This PR introduces ...

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Chore

## Checklist
**Pre-work Notes:**
- [ ] ...
```


## 4 Reproducible EDA Pipeline (uv + Quarto)

This repo uses **GitHub Actions + uv + Quarto** to ensure the EDA workflow and Executive Summary are reproducible from a clean environment.

## What CI Does
On pushes to `main` (and selected dev branches), CI:
1) installs deps from `uv.lock`
2) runs `EDA/ci.py`
3) renders `EDA/executive_summary.qmd` → `EDA/executive_summary.html`
4) uploads the HTML as a workflow artifact
5) (optional) deploys to GitHub Pages if enabled by the repo owner

Workflow location:
- `.github/workflows/`

## File Contracts (Do Not Break)
- **CI entry script:** `EDA/ci.py` (only script called directly by CI)
- **Report source:** `EDA/executive_summary.qmd`
- **Report output:** `EDA/executive_summary.html`

If you rename/move any of the above, you must update the workflow YAML accordingly.

## Dependencies (uv)
Files:
- `pyproject.toml`
- `uv.lock`

When changing dependencies:
1) edit `pyproject.toml`
2) run `uv sync`
3) commit **both** `pyproject.toml` and `uv.lock`

CI uses `uv sync --frozen`; outdated/missing `uv.lock` will fail CI.

## Do Not Commit
- `.venv/`
- `build/`
- `EDA/executive_summary.html`

## Safe Team Workflow
Work on a branch → push → confirm Actions is green → merge to `main`.
""""