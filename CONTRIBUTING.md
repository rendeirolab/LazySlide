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

3. Create a new environment and install the dependencies

    ```bash
    mamba create -n lazyslide python==3.11
    conda activate lazyslide
    mamba install -c conda-forage openslide-python
    pip install -e '.[dev,all]'
    ```
   
    We use [pre-commit](https://pre-commit.com/) to run code formatting and linting checks before each commit.

    ```bash
    pip install pre-commit
    pre-commit install
    ```

4. Make your changes

5. (If needed) Add a test case and then run the tests

    ```bash
    pytest tests
    ```

6. (If needed) Update the documentation

    ```bash
    cd docs
    make clean html
    # Launch a local server to view the documentation
    python -m http.server -d build/html
    ```
    
    Open your browser and navigate to `http://localhost:8000`
   
7. Commit your changes and push them to your fork

8. Submit a pull request


## How to contribute


## How to report bugs


## How to suggest enhancements
