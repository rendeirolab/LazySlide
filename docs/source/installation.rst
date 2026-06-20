Installation
============

You can install :code:`lazyslide` with different package manager you prefer.

.. tab-set::

    .. tab-item:: PyPI

        The default installation.

        .. code-block:: bash

            pip install lazyslide

    .. tab-item:: uv

        .. code-block:: bash

            uv add lazyslide

    .. tab-item:: Conda

        .. code-block:: bash

            conda install conda-forge::lazyslide

    .. tab-item:: Mamba

        .. code-block:: bash

            mamba install lazyslide

    .. tab-item:: Development

        If you want to install the latest version from the GitHub repository, you can use the following command:

        .. code-block:: bash

            pip install git+https://github.com/rendeirolab/lazyslide.git


Verify the installation
-----------------------

Confirm the installed version:

.. code-block:: bash

    python -c "import lazyslide as zs; print(zs.__version__)"

Then run a small test that does not download model weights:

.. code-block:: python

    import lazyslide as zs

    wsi = zs.datasets.sample(with_data=False)
    zs.pp.find_tissues(wsi, level=-1)
    print(wsi.shapes.keys())

Continue with :doc:`getting-started/first-analysis`, or see
:doc:`how-to/installation` if a reader or model dependency is unavailable.


Installation of slide readers
-----------------------------

LazySlide uses `wsidata <https://wsidata.readthedocs.io>`_ to handle the IO with the slide files.
To support different file formats, you need to install corresponding slide readers.
The reader will be automatically detected by `wsidata <https://wsidata.readthedocs.io>`_ when you open the slide file.


.. tab-set::

    .. tab-item:: TiffSlide

        `TiffSlide <https://github.com/Bayer-Group/tiffslide>`_ is a cloud native openslide-python replacement
        based on tifffile.

        TiffSlide is installed by default. You don't need to install it manually.

        .. code-block:: bash

            pip install tiffslide

    .. tab-item:: OpenSlide

        `OpenSlide <https://openslide.org/>`_ is a C library that provides a simple interface to read whole-slide images.

        OpenSlide is installed by default, you don't need to install it manually.

        But you can always install from PyPI

        .. code-block:: bash

            pip install openslide-python openslide-bin

        In case your OpenSlide installation is not working, you can install it manually.

        For Linux and OSX users, it's suggested that you install :code:`openslide` with conda or mamba:

        .. code-block:: bash

            conda install -c conda-forge openslide-python
            # or
            mamba install -c conda-forge openslide-python


        For Windows users, you need to download compiled :code:`openslide` from
        `GitHub Release <https://github.com/openslide/openslide-bin/releases>`_.
        If you open the folder, you should find a :code:`bin` folder.

        Make sure you point the :code:`bin` folder for python to locate the :code:`openslide` binary.
        You need to run following code to import the :code:`openslide`,
        it's suggested to run this code before everything:

        .. code-block:: python

            import os
            with os.add_dll_directory("path/to/openslide/bin"):
                import openslide

    .. tab-item:: BioFormats

        `BioFormats <https://www.openmicroscopy.org/bio-formats/>`_ is a standalone Java library
        for reading and writing life sciences image file formats.

        `scyjava <https://github.com/scijava/scyjava>`_ is used to interact with the BioFormats library.

        .. code-block:: bash

            pip install scyjava

    .. tab-item:: CuCIM

        `CuCIM <https://github.com/rapidsai/cucim>`_ is a GPU-accelerated image I/O library.

        Please refer to the `CuCIM GitHub <https://github.com/rapidsai/cucim>`_.

    .. tab-item:: fastslide

        `fastslide <https://github.com/NKI-AI/fastslide>`_ is a high-performance C++ whole-slide image reader with native Python bindings.

        .. code-block:: bash

            pip install fastslide

    .. tab-item:: pyisyntax

        `pyisyntax <https://github.com/anibali/pyisyntax>`_ reads Philips iSyntax pathology images using libisyntax.

        .. code-block:: bash

            pip install pyisyntax

    .. tab-item:: pylibCZIrw

        `pylibCZIrw <https://github.com/ZEISS/pylibczirw>`_ is the official Python binding for reading and writing Zeiss CZI images.
        It is the recommended CZI backend on Apple Silicon because it decodes JPEG-XR natively.

        .. code-block:: bash

            pip install pylibCZIrw

        Wheels are available for Linux (x86_64 and aarch64), Apple Silicon macOS, and Windows x86_64.
        Intel macOS users need to build it from source or use Bio-Formats.

Reader availability and platform support can change independently of LazySlide. See the
`current wsidata reader installation guide <https://wsidata.readthedocs.io/en/latest/installation.html#installation-for-slide-readers>`_
for authoritative backend-specific details.
