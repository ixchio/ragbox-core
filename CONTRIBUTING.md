# Contributing to RAGBox-Core

Welcome! We are thrilled that you are interested in contributing to RAGBox-Core. This document outlines the process for contributing to the project.

## Code of Conduct
Please be respectful and considerate of others when contributing. We strive to maintain a welcoming and inclusive community.

## How to Contribute

### 1. Reporting Bugs
- Check the issue tracker to see if the bug has already been reported.
- If not, open a new issue describing the bug, including steps to reproduce, expected behavior, and actual behavior.

### 2. Suggesting Enhancements
- Open a new issue detailing your proposed enhancement. Note why this enhancement would be useful to the majority of RAGBox users.

### 3. Pull Requests
1. **Fork the repository** on GitHub.
2. **Clone your fork** locally.
3. **Install dependencies** using Poetry: `poetry install`.
4. **Create a new branch** for your feature or bugfix: `git checkout -b feature/my-awesome-feature`.
5. **Write tests** for your changes.
6. **Ensure all tests pass** by running `poetry run pytest tests/`.
7. **Format your code** with Black: `poetry run black .`.
8. **Commit your changes**: `git commit -m "Add some awesome feature"`.
9. **Push to your fork**: `git push origin feature/my-awesome-feature`.
10. **Open a Pull Request** against the `main` branch of the upstream repository.

## Architecture Guidelines
When contributing, please follow the core philosophy of RAGBox: zero-configuration. 
* Avoid adding complex setup steps.
* If a new feature requires external tools, handle the fallbacks gracefully.
* Ensure type hints are used (`mypy` support is planned).

Thank you for making RAGBox better!
