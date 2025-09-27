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

## 2. Create a Git Tag

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

## 3. Build and Publish to PyPI

1.  **Ensure you have the latest build tools:**
    ```bash
    pip install --upgrade build twine
    ```

2.  **Build the source distribution and wheel:**
    ```bash
    python -m build
    ```
    This will create a `dist` directory with the `.tar.gz` and `.whl` files.

3.  **Upload the package to PyPI:**
    - For a test release to TestPyPI:
      ```bash
      twine upload --repository testpypi dist/*
      ```
    - For the official release to PyPI:
      ```bash
      twine upload dist/*
      ```
    You will be prompted for your PyPI username and password.

## 4. Create a GitHub Release

1.  **Navigate to the "Releases" page in the GitHub repository.**
2.  **Click "Draft a new release."**
3.  **Select the tag you just pushed (e.g., `vX.Y.Z`).**
4.  **For the release title, enter `vX.Y.Z`.**
5.  **Copy the release notes from `CHANGELOG.md` into the description.**
6.  **Click "Publish release."**

This completes the release process.