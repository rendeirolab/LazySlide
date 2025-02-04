# Contributing

We welcome contributions to this project.


## For core contributors

Please do not commit directly to the `main` branch. 
Instead, create a new branch for your changes and submit a pull request.

### How to set up your development environment

1. Clone the repository

    ```bash
    git clone https://github.com/rendeirolab/LazySlide.git
    # or
    gh repo clone rendeirolab/LazySlide
    ```
   
2. Checkout a new branch

    ```bash
    git checkout -b my-new-branch
    ```

3. We use [uv](https://docs.astral.sh/uv/) to manage our development environment.

    ```bash
    uv lock
    uv run pre-commit install
    ```
   
    We use [pre-commit](https://pre-commit.com/) to run code formatting and linting checks before each commit.

4. Start a IPython/Jupyter session

    ```bash
   uv run --with ipython ipython
   # or
   uv run --with jupyter jupyter lab
   ```

5. Make your changes

6. (If needed) Add a test case and then run the tests

    ```bash
    uv run task test
    ```

7. (If needed) Update the documentation

   To build the documentation, use:
   
   ```bash
   # Build doc with cache
   uv run task doc-build
   # Fresh build
   uv run task doc-clean-build
   ```
   
   To serve the documentation, use:
   
   ```bash
   uv run task doc-serve
   ```
   
   This will start a local server at [http://localhost:8000](http://localhost:8000).

8. Commit your changes and push them to your fork

9. Submit a pull request


## How to report bugs


## How to suggest enhancements
