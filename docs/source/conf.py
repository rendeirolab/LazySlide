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
        "cv_features": (
            "Tile prediction models (CV Features)",
            "tile_prediction.cv_features",
            list(CV_FEATURES.values()),
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
        for m in v.model_type:
            model_sections[m.value][-1].add(v.module)

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


def template_model_card(card):
    content = [
        f"   .. grid-item-card:: {card.name}",
        "      :shadow: sm",  # Light shadow
        "      :class-card: model-card",  # Custom class for card styling
        "",
        f"      {card._doc()}",
    ]
    return content


def generate_model_table(app, config):
    """Generate model table for avail_models.rst."""
    # Import here to avoid circular imports
    from lazyslide.models._model_registry import MODEL_REGISTRY

    avail_models_file = Path(app.srcdir) / "avail_models.rst"
    avail_models_skeleton_file = Path(app.srcdir) / "avail_models_skeleton.rst"

    # Read existing content
    with open(avail_models_skeleton_file, "r") as f:
        content = f.read()

    # Split content at the point where we want to insert the table
    # This is after the section about accessing gated models
    split_marker = "Below is a list of available models categorized by their type:"
    if split_marker in content:
        before, after = content.split(split_marker, 1)
        before += split_marker + "\n\n"
    else:
        before = content
        after = ""

    # Generate model table
    table_content = [
        "Model cards",
        "-----------",
        "",
        ".. grid:: 1 2 2 2",
        "   :gutter: 3",  # Spacing between grid items
        "",
    ]

    for key, card in MODEL_REGISTRY.items():
        table_content.extend(template_model_card(card))

    # Ensure there's an empty line after the grid directive for proper RST parsing
    table_content.append("")

    # Combine content
    new_content = before + "\n".join(table_content) + "\n\n" + after

    # Write to file
    with open(avail_models_file, "w") as f:
        f.write(new_content)

    print(f"Updated {avail_models_file} with model table")


def setup(app):
    """Set up Sphinx extension."""
    # Must hook into the very first event before autosummary executed
    app.connect("config-inited", generate_models_rst)
    # app.connect('config-inited', generate_model_table)
    return {"version": "0.1", "parallel_read_safe": True}
