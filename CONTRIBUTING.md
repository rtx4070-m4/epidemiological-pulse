# Contributing to Epidemiological Pulse

Thank you for your interest in contributing! This document explains the workflow and standards we follow.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Branch Strategy](#branch-strategy)
- [Commit Messages](#commit-messages)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

By participating in this project you agree to abide by the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) Code of Conduct. Please report unacceptable behaviour to the maintainers.

---

## How to Contribute

### Reporting bugs
1. Search [existing issues](../../issues) first to avoid duplicates.
2. Open a new issue using the **Bug Report** template.
3. Include: Python version, OS, full traceback, and minimal reproducible example.

### Suggesting features
1. Open an issue using the **Feature Request** template.
2. Describe the use case and expected behaviour.

### Submitting code
1. Fork the repository.
2. Create a feature branch (see [Branch Strategy](#branch-strategy)).
3. Implement your changes with tests.
4. Open a pull request.

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/your-username/epidemiological_pulse.git
cd epidemiological_pulse

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies (including dev extras)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable releases only |
| `develop` | Integration branch |
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation updates |
| `refactor/<name>` | Code restructuring |

```bash
# Always branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
```

---

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

**Types:** `feat` | `fix` | `docs` | `style` | `refactor` | `test` | `chore` | `perf`

**Examples:**
```
feat(models): add SARIMA baseline forecaster
fix(dashboard): correct UTC offset in date picker callback
docs(readme): add Docker Compose example
test(pipeline): add integration test for full run with synthetic data
```

---

## Code Style

All Python code must conform to:

| Tool | Config | Command |
|---|---|---|
| **black** | `line-length = 88` | `black src/ tests/` |
| **isort** | `profile = black` | `isort src/ tests/` |
| **flake8** | `max-line-length = 88` | `flake8 src/ tests/` |
| **mypy** | `strict = false` | `mypy src/` |

Run all checks at once:
```bash
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
```

### Additional standards
- **Type annotations** on all public functions and methods.
- **Docstrings** (Google style) on all public modules, classes, and functions.
- **Logging** via `logging.getLogger(__name__)` — never `print()` in library code.
- **No bare `except:`** — always catch specific exception types.

---

## Testing

Every code change must be accompanied by tests.

```bash
# Run the full test suite
pytest tests/ -v

# With coverage (aim for ≥ 80 %)
pytest tests/ --cov=src --cov-report=term-missing -v
```

### Test guidelines
- Tests live in `tests/` and mirror the `src/` structure.
- Use `pytest-mock` for patching external dependencies (API calls, file I/O).
- Use fixtures in `conftest.py` for shared setup.
- Each test function name must describe the scenario: `test_<function>_<scenario>_<expected>`.

---

## Pull Request Process

1. **Rebase** your branch on the latest `develop` before opening a PR.
2. Ensure all CI checks pass (lint, tests, coverage).
3. Fill in the PR template completely.
4. Request a review from at least one maintainer.
5. Address review comments promptly.
6. A maintainer will squash-merge your branch once approved.

### PR checklist
- [ ] Code follows style guidelines
- [ ] New/modified functions have docstrings and type annotations
- [ ] Tests added / updated
- [ ] `requirements.txt` updated if new dependencies added
- [ ] `CHANGELOG.md` entry added (for non-trivial changes)
- [ ] `README.md` updated if user-facing behaviour changed

---

Thank you for helping make **Epidemiological Pulse** better! 🦠📊
