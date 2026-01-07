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