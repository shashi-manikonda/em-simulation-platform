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

## 2. Publish to PyPI and Create a GitHub Release

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

3.  **Automatic PyPI Publication:**
    - Pushing a new tag that starts with `v` (e.g., `v1.2.3`) will automatically trigger the `publish.yml` GitHub Actions workflow.
    - This workflow builds the package and publishes it to PyPI.
    - You can monitor the progress of the workflow on the "Actions" tab of the GitHub repository.

4.  **Create a GitHub Release:**
    - Once the workflow has successfully completed, navigate to the "Releases" page in the GitHub repository.
    - Click "Draft a new release."
    - Select the tag you just pushed (e.g., `vX.Y.Z`).
    - For the release title, enter `vX.Y.Z`.
    - Copy the release notes from `CHANGELOG.md` into the description.
    - Click "Publish release."

This completes the release process.