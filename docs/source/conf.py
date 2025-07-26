from datetime import datetime

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
nb_execution_excludepatterns = [
    "02_feature_extraction.ipynb",
    "03_multiple_slides.ipynb",
    "04_genomics_integration.ipynb",
    "05_cell_segmentation.ipynb",
    "07_zero-shot-learning.ipynb",
]
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
