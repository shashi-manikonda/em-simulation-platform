# Release Process

This document outlines the steps to release a new version of the `em-app` package.

## 1. Prepare the Release

1.  **Ensure you are on the `main` branch and have the latest changes:**
    ```bash
    git checkout main
    git pull origin main
    ```

2.  **Update the version number in `pyproject.toml`:**
    - Open `pyproject.toml` and increment the `version` number under the `[project]` section. Follow [Semantic Versioning](https://semver.org/).

3.  **Update the `CHANGELOG.md`:**
    - Add a new entry for the new version.
    - List all notable changes, including new features, bug fixes, and other improvements.
    - Make sure the release date is correct.

4.  **Commit the changes:**
    ```bash
    git add pyproject.toml CHANGELOG.md
    git commit -m "chore: Prepare release vX.Y.Z"
    ```
    (Replace `X.Y.Z` with the new version number).

## 2. Trigger the Automated Release

1.  **Tag the new version:**
    ```bash
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    ```
    (Replace `X.Y.Z` with the new version number).

2.  **Push the changes and the new tag to GitHub:**
    ```bash
    git push origin main
    git push origin vX.Y.Z
    ```

3.  **Automatic Release:**
    - Pushing a new tag that starts with `v` (e.g., `v1.2.3`) will automatically trigger the `publish.yml` GitHub Actions workflow.
    - This workflow handles the entire release process:
        - It builds the package.
        - It publishes the package to PyPI.
        - It creates a new GitHub Release, using the notes from `CHANGELOG.md` and attaching the built package files as assets.
    - You can monitor the progress of the workflow on the "Actions" tab of the GitHub repository.

This completes the release process.
