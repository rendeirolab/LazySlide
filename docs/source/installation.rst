:octicon:`package;1em;mr-1` Installation
==========================================

PIP
---

To install the released version, simply run:


.. code-block:: bash

    pip install lazyslide


To install the latest development version, run:


.. code-block:: bash

    pip install git+https://github.com/rendeirolab/LazySlide.git


Installation for openslide
--------------------------

By default, LazySlide will try to use :code:`openslide` when available to support wider range of image formats.

It will also load :code:`tiffslide` when :code:`openslide` is not available. Especially on windows platform.

For Linux and OSX users, it's suggested that you install :code:`openslide` with conda or mamba:

.. code-block:: bash

    conda install -c conda-forge openslide-python
    # or
    mamba install -c conda-forge openslide-python


For Windows users, you need to download compiled :code:`openslide` from
`GitHub Release <https://github.com/openslide/openslide-bin/releases>`_.

Before import lazyslide in your code, you need to set the path to the :code:`openslide` library:

.. code-block:: python

    import os
    with os.add_dll_directory("path/to/openslide/bin")):
        import openslide
