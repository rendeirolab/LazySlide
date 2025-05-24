Contributing
============

We welcome contributions to the LazySlide project. This document provides guidelines for contributing to the project.

Project overview
----------------

LazySlide is a modularized and scalable whole slide image analysis toolkit. The project is structured as follows:

- ``src/lazyslide``: Main package code
- ``tests``: Test files
- ``docs``: Documentation


For core contributors
---------------------

Please do not commit directly to the ``main`` branch.
Instead, create a new branch for your changes and submit a pull request.

Set up development environment
------------------------------

We use `uv <https://docs.astral.sh/uv/>`_ to manage our development environment.
Please make sure you have it installed before proceeding.

1. Clone the repository::

    git clone https://github.com/rendeirolab/lazyslide.git
    # or
    gh repo clone rendeirolab/lazyslide

2. Checkout a new branch::

    git checkout -b my-new-branch

3. We use `uv <https://docs.astral.sh/uv/>`_ to manage our development environment::

    uv lock
    uv run pre-commit install

   We use `pre-commit <https://pre-commit.com/>`_ to run code formatting and linting checks before each commit.

4. Start an IPython/Jupyter session::

   uv run --with ipython ipython
   # or
   uv run --with jupyter jupyter lab

5. Make your changes.

Testing
-------

LazySlide uses pytest for testing. Tests are located in the ``tests`` directory.

To run all tests::

    uv run task test

To run a specific test file::

    uv run python -m pytest tests/test_example.py

When adding new tests:

1. Create a new file in the ``tests`` directory with a name starting with ``test_``.
2. Import pytest and the module you want to test.
3. Write test functions with names starting with ``test_``.
4. Use assertions to verify expected behavior.

Code style and development guidelines
------------------------------------

LazySlide uses `ruff <https://github.com/astral-sh/ruff>`_ for both linting and formatting. 
The configuration is defined in ``pyproject.toml`` and enforced through pre-commit hooks.

To format code::

    uv run task fmt
    # or
    ruff format docs/source src/lazyslide tests

Documentation
------------

Documentation is built using Sphinx and is located in the ``docs`` directory.

To build the documentation::

   # Build doc with cache
   uv run task doc-build
   # Fresh build
   uv run task doc-clean-build

To serve the documentation locally::

   uv run task doc-serve

This will start a local server at http://localhost:8000.

Documentation is written in reStructuredText (.rst) and Jupyter notebooks (.ipynb) using the myst-nb extension.

Submitting changes
-----------------

1. Commit your changes and push them to your branch.
2. Create a pull request on GitHub.
3. Ensure all CI checks pass.
4. Wait for a review from a maintainer.

Reporting issues
---------------

If you encounter a bug or have a feature request, please open an issue on the 
`GitHub repository <https://github.com/rendeirolab/lazyslide/issues>`_.

When reporting a bug, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Any relevant logs or error messages
- Your environment (OS, Python version, package versions)
