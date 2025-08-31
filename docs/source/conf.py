import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from natsort import natsorted

import lazyslide

project = "LazySlide"
copyright = f"{datetime.now().year}, Rendeiro Lab"
author = "LazySlide Contributors"
release = lazyslide.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_contributors",
    "myst_nb",
    "myst_sphinx_gallery",
]
autoclass_content = "class"
autodoc_docstring_signature = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "no-undoc-members": True,
    "special-members": "__call__",
    "exclude-members": "__init__, __weakref__",
    "class-doc-from": "class",
}
autodoc_typehints = "none"
autosectionlabel_prefix_document = True
# setting autosummary
autosummary_generate = True
numpydoc_show_class_members = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo@3x.png"
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/rendeirolab/LazySlide",
    "navigation_with_keys": True,
    "show_prev_next": False,
}
# html_sidebars = {"installation": [], "cli": []}

nb_output_stderr = "remove"
nb_execution_in_temp = True
nb_execution_mode = "off"
nb_execution_excludepatterns = []
nb_merge_streams = True
myst_enable_extensions = [
    "colon_fence",
    "html_image",
]

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5," r"8}: "
copybutton_prompt_is_regexp = True

# Plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 200)]

nitpicky = True
# Alternative approach - ignore specific references
nitpick_ignore = [
    ("py:class", "lazyslide.models.tile_prediction.cv_features._CVFeatures"),
    ("py:class", "abc.ABC"),
    ("py:class", "Scorer"),
    ("py:class", "lazyslide.models.vision.hibou.Hibou"),
    ("py:class", "lazyslide.models.tile_prediction.spider.Spider"),
]

intersphinx_mapping = {
    "wsidata": ("https://wsidata.readthedocs.io/en/latest", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
}


# -- Dynamic pulling tagged or the latest version of tutorials -------------
def get_clean_version(version):
    """
    Extract the semantic version (X.Y.Z) from a version string,
    removing any dev labels, post-release info, etc.

    Example: "0.8.0.post10.dev0+9888929" -> "0.8.0"
    """
    # Match X.Y.Z pattern at the beginning of the version string
    match = re.match(r"^(\d+\.\d+\.\d+)", version)
    if match:
        return match.group(1)
    return version


def pull_tutorials(app, config):
    """
    Pull tutorials from the lazyslide-tutorials repository based on the current version.
    If a tag matching the current version exists, pull from that tag.
    Otherwise, pull from the latest commit.
    """
    # Get the clean version
    clean_version = get_clean_version(release)
    print(f"LazySlide version: {release}, clean version: {clean_version}")

    # Define paths
    tutorials_repo_url = "https://github.com/rendeirolab/lazyslide-tutorials.git"
    tutorials_dir = Path(app.srcdir) / "tutorials"

    # Create a temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Clone the repository
        print(f"Cloning {tutorials_repo_url} to {temp_dir_path}")
        subprocess.run(
            ["git", "clone", tutorials_repo_url, str(temp_dir_path)],
            check=True,
            capture_output=True,
        )

        # Check if a tag exists for the current version
        result = subprocess.run(
            ["git", "tag", "-l", f"v{clean_version}"],
            cwd=str(temp_dir_path),
            check=True,
            capture_output=True,
            text=True,
        )

        # If tag exists, checkout that tag
        if result.stdout.strip():
            tag = f"v{clean_version}"
            print(f"Tag {tag} found, checking out")
            subprocess.run(
                ["git", "checkout", tag],
                cwd=str(temp_dir_path),
                check=True,
                capture_output=True,
            )
        else:
            print(f"No tag found for version {clean_version}, using latest commit")

        # Copy tutorials to the docs directory
        src_tutorials_dir = temp_dir_path / "tutorials"

        # Create tutorials directory if it doesn't exist
        tutorials_dir.mkdir(exist_ok=True)

        # Copy all .ipynb files from the tutorials directory
        for ipynb_file in src_tutorials_dir.rglob("*.ipynb"):
            dest_file = tutorials_dir / ipynb_file.name
            print(f"Copying {ipynb_file} to {dest_file}")
            shutil.copy2(ipynb_file, dest_file)

    print(f"Tutorials pulled successfully to {tutorials_dir}")


# -- Dynamic documentation generation for models ---------------------------
def template_model_api(title, module_name, models):
    currentmodule = "lazyslide.models"
    if module_name is not None:
        currentmodule += f".{module_name}"

    content = [
        "",
        f"{title}",
        f"{'~' * len(title)}",
        "",
        f".. currentmodule:: {currentmodule}",
        "",
        ".. autosummary::",
        "    :toctree: _autogen",
        "    :nosignatures:",
        "",
    ]

    if module_name is not None:
        names = [model.__name__ for model in models]
    else:
        names = [
            model.__module__.split(".")[2] + "." + model.__name__ for model in models
        ]
    for name in natsorted(names):
        content.append(f"    {name}")

    return content


def generate_models_rst(app, config):
    """Generate models.rst file dynamically based on the model registry."""
    models_dir = Path(app.srcdir) / "api"
    models_file = models_dir / "models.rst"

    from lazyslide.models import MODEL_REGISTRY
    from lazyslide.models import base as mb
    from lazyslide.models.segmentation import SMPBase
    from lazyslide.models.tile_prediction import CV_FEATURES

    # Define model lists manually based on the current models.rst file
    model_sections = {
        "vision": ("Vision models", "vision", set()),
        "multimodal": ("Multimodal models", "multimodal", set()),
        "segmentation": (
            "Segmentation models",
            "segmentation",
            {
                SMPBase,
            },
        ),
        "tile_prediction": ("Tile prediction models", "tile_prediction", set()),
        "slide_encoder": ("Slide encoder models", None, set()),
        "cv_feature": (
            "Computer vision features",
            "tile_prediction.cv_features",
            set(),
        ),
        "base": (
            "Base model class",
            "base",
            [
                mb.ModelBase,
                mb.ImageModel,
                mb.ImageTextModel,
                mb.SegmentationModel,
                mb.SlideEncoderModel,
                mb.TilePredictionModel,
                mb.TimmModel,
            ],
        ),
    }

    for _, v in MODEL_REGISTRY.items():
        if isinstance(v.task, mb.ModelTask):
            task = [v.task]
        else:
            task = v.task
        for m in task:
            model_sections[m.value][-1].add(v)

    template = [
        ".. _models-section:",
        "",
        "Models",
        "------",
        "",
        ".. currentmodule:: lazyslide.models",
        "",
        ".. autosummary::",
        "    :toctree: _autogen",
        "    :nosignatures:",
        "",
        "    list_models",
        "",
    ]

    for _, models in model_sections.items():
        title, module, models = models
        content = template_model_api(title, module, models)
        template.extend(content)

    # Write to file
    with open(models_file, "w") as f:
        f.write("\n".join(template))

    print(f"Generated {models_file}")


def setup(app):
    """Set up Sphinx extension."""
    # Must hook into the very first event before autosummary executed
    app.connect("config-inited", generate_models_rst)

    # Connect the pull_tutorials function to the builder-inited event
    # This ensures it runs after the configuration is initialized but before the build starts
    app.connect("config-inited", pull_tutorials)

    return {"version": "0.1", "parallel_read_safe": True}
