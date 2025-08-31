# Setup development environment

We use [uv](https://docs.astral.sh/uv/) to manage our development environment.
Please make sure you have it installed before proceeding.

1. Clone the repository

```bash
git clone https://github.com/rendeirolab/lazyslide.git
# or
gh repo clone rendeirolab/lazyslide
```

2. Checkout to a new branch

Please replace `new-feature` with a meaningful name for your branch.

```bash
git checkout -b new-feature
```

3. Sync development environment

```bash
uv sync
```

4. Install pre-commit hooks

```bash
uv run pre-commit install
```

We use [pre-commit](https://pre-commit.com/) to run code formatting and linting checks before each commit.

5. Start a Python session

**IPython**
```bash
uv run --with ipython ipython
```

**Jupyter**
```bash
uv run --with jupyter jupyter lab
# With extensions
uv run --with jupyter --with dask-labextension jupyter lab
```

**VS Code/PyCharm**
If you run in VS Code or PyCharm, you don't need to do anything else.

## Code style

The configuration is defined in `pyproject.toml` and enforced through pre-commit hooks.

To format code

```bash
uv run task fmt
# or
ruff format docs/source src/lazyslide tests
```

## Testing

LazySlide uses pytest for testing. Tests are located in the `tests` directory.

To run all tests

```bash
uv run task test
```
To run a specific test file

```bash
uv run python -m pytest tests/test_example.py
```

When adding new tests:

1. Create a new file in the ``tests`` directory with a name starting with `test_*`.
2. Import pytest and the module you want to test.
3. Write test functions with names starting with `test_*`.
4. Use assertions to verify expected behavior.

## Documentation

Documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/) and is located in the ``docs`` directory.

To build the documentation

```bash
# Build doc with cache
uv run task doc-build
# Fresh build
uv run task doc-clean-build
```

To serve the documentation locally

```bash
uv run task doc-serve
```

This will start a local server at http://localhost:8000.

Documentation is written in reStructuredText (.rst) and Jupyter notebooks (.ipynb) using the myst-nb extension.

## Submitting changes

1. Commit your changes and push them to your branch.
2. Create a pull request on GitHub.
3. Ensure all CI checks pass.
4. Wait for a review from a maintainer.
