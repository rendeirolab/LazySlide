# Contributing

We welcome contributions to the LazySlide project. This document provides guidelines for contributing to the project.

## Project overview

LazySlide is a modularized and scalable whole slide image analysis toolkit. The project is structured as follows:

- `src/lazyslide`: Main package code
- `tests`: Test files
- `docs`: Documentation
- [lazyslide-tutorial](https://github.com/rendeirolab/lazyslide-tutorials)

## Reporting issues

If you encounter a bug or have a feature request, please open an issue on the 
[GitHub repository](https://github.com/rendeirolab/lazyslide/issues).

When reporting a bug, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Any relevant logs or error messages
- Your environment (OS, Python version, package versions)


## For core contributors

Please do not commit directly to the ``main`` branch.
Instead, create a new branch for your changes and submit a pull request.


::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Setup development environment {octicon}`tools;1em;`
:link: setup
:link-type: doc

Setting up your dev environment and start contributing!
:::

:::{grid-item-card} New model!
:link: new_models
:link-type: doc

The procedure on how to contribute a new model
:::

:::{grid-item-card} Release cycle
:link: release_cycle
:link-type: doc

Understand the release cycle of LazySlide
:::

::::


```{toctree}
:hidden: true
:maxdepth: 1

setup
new_models
release_cycle
```