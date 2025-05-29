# SMOOPs Linting and Quality Control Guide

This document provides an overview of the linting and quality control (QC) tools used in the SMOOPs project, as well as instructions for running them locally and understanding the workflow integration.

## Overview

The SMOOPs project uses a comprehensive set of linting and quality control tools to ensure code quality, maintain consistency, and prevent bugs. These tools are integrated into both local development workflows and CI/CD pipelines.

### Tools Used

#### JavaScript/TypeScript
- **ESLint**: Static code analysis tool for identifying problematic patterns
- **Jest**: Testing framework with built-in coverage reporting

#### Python
- **Ruff**: An extremely fast Python linter, written in Rust
- **Black**: Code formatter that enforces a consistent style
- **isort**: Utility to sort imports alphabetically and automatically separate them into sections
- **MyPy**: Static type checker for Python
- **Bandit**: Tool designed to find common security issues in Python code
- **pytest**: Testing framework with pytest-cov for coverage reporting

## Local Setup

### Prerequisites

1. Node.js (v18+) and npm/yarn for JavaScript/TypeScript tools
2. Python 3.10+ for Python tools

### Installation

All necessary tools are included in the project's dependencies:

1. For JavaScript/TypeScript tools:
   ```bash
   # From project root
   npm install
   ```

2. For Python tools:
   ```bash
   # From project root
   pip install -r ml/requirements.txt
   ```

## Running Linters Locally

### Using dev-tasks.sh (Recommended)

The project includes a `dev-tasks.sh` script that provides convenient commands for running linters:

```bash
# Run all linters
npm run lint
# or directly with the script
bash scripts/dev-tasks.sh lint

# Run all linters with auto-fix enabled
npm run lint:fix
# or directly with the script
bash scripts/dev-tasks.sh lint:fix
```

### Running Individual Tools Manually

If you prefer to run specific tools individually:

#### JavaScript/TypeScript
```bash
# Run ESLint
npx eslint "**/*.{js,jsx,ts,tsx}" --ignore-pattern "node_modules/" --ignore-pattern "dist/"

# Run ESLint with auto-fix
npx eslint "**/*.{js,jsx,ts,tsx}" --ignore-pattern "node_modules/" --ignore-pattern "dist/" --fix
```

#### Python
```bash
# Run Ruff (fast linter)
python -m ruff check .

# Run Ruff with auto-fix
python -m ruff check --fix .

# Run Black (formatter)
python -m black .

# Run isort (import sorter)
python -m isort .

# Run MyPy (type checker)
python -m mypy ml/src

# Run Bandit (security checker)
python -m bandit -r ml/src -ll
```

## CI/CD Integration

All linting and QC tools are integrated into our GitHub Actions CI/CD pipeline:

- **Workflow File**: `.github/workflows/ci.yml`
- **Trigger**: Runs on push to main branch and on pull requests
- **Jobs**:
  - **backend**: Runs ESLint and tests for backend code
  - **frontend**: Runs ESLint, builds the frontend, and runs tests
  - **ml**: Runs Ruff, Black, MyPy, Bandit, and pytest for Python code

The CI pipeline ensures that all code meets our quality standards before being merged.

## Configuration Files

### JavaScript/TypeScript
- ESLint configuration is located in `.eslintrc.js` files in the frontend and backend directories

### Python
- Configuration for Python tools is centralized in `pyproject.toml`:
  - **Black**: Controls code formatting style
  - **isort**: Controls import sorting
  - **MyPy**: Controls type checking strictness
  - **Ruff**: Controls linting rules
  - **pytest**: Controls test discovery and execution

## Code Quality Standards

### Code Style
- JavaScript/TypeScript: Follows Airbnb style guide with project-specific modifications
- Python: Follows PEP 8 with Black formatting (line length of 100 characters)

### Code Quality Metrics
- Test coverage: Aim for 80%+ coverage for critical components
- Complexity: Functions should not exceed 10 in cyclomatic complexity
- Maintainability: Keep functions and classes focused and small

## Troubleshooting

### Common Issues

1. **ESLint not finding configuration**:
   - Make sure you're running ESLint from the project root or a directory with an `.eslintrc.js` file

2. **Python linters raising many errors on first run**:
   - Run `npm run lint:fix` to automatically fix many common issues
   - For type errors, add appropriate type annotations gradually

3. **CI failing but local checks pass**:
   - Ensure you're using the same configuration locally as in CI
   - Pull the latest changes to get the most recent configuration files

## Adding New Rules or Tools

To add new linting rules or tools:

1. Update the appropriate configuration file (`.eslintrc.js` or `pyproject.toml`)
2. Update the `dev-tasks.sh` script if adding a new command
3. Update the CI workflow in `.github/workflows/ci.yml`
4. Update this documentation
5. Announce changes to the team

## Resources

- [ESLint Documentation](https://eslint.org/docs/user-guide/)
- [Ruff Documentation](https://beta.ruff.rs/docs/)
- [Black Documentation](https://black.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/) 