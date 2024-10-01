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
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_click",
    "myst_nb",
]
autoclass_content = "class"
autodoc_docstring_signature = True
autodoc_default_options = {"members": None, "undoc-members": None}
autodoc_typehints = "none"
# setting autosummary
autosummary_generate = True
numpydoc_show_class_members = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/rendeirolab/LazySlide",
    "navigation_with_keys": True,
    "show_prev_next": False,
}
html_sidebars = {"installation": [], "cli": []}

nb_execution_mode = "off"
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

intersphinx_mapping = {
    "wsidata": ("https://wsidata.readthedocs.io/en/latest", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
