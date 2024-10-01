Installation
============

You can install :code:`lazyslide` with different package manager you prefer.

.. tab-set::

    .. tab-item:: PyPI

        The default installation.

        .. code-block:: bash

            pip install lazyslide


    .. tab-item:: Conda

        .. warning::

           Not available yet.

        .. code-block:: bash

            conda install -c conda-forge lazyslide

    .. tab-item:: Mamba

        .. warning::

           Not available yet.

        .. code-block:: bash

            mamba install lazyslide

    .. tab-item:: Development

        If you want to install the latest version from the GitHub repository, you can use the following command:

        .. code-block:: bash

            pip install git+https://github.com/rendeirolab/lazyslide.git


Installation for slide readers
------------------------------

LazySlide uses :code:`wsidata` to handle the IO with the slide files.
To support different file formats, you need to install corresponding slide readers.
The reader will be automatically detected by :code:`wsidata` when you open the slide file.


.. tab-set::

    .. tab-item:: TiffSlide

        `TiffSlide <https://github.com/Bayer-Group/tiffslide>`_ is a cloud native openslide-python replacement
        based on tifffile.

        TiffSlide is installed by default. You don't need to install it manually.

        .. code-block:: bash

            pip install tiffslide

    .. tab-item:: OpenSlide

        `OpenSlide <https://openslide.org/>`_ is a C library that provides a simple interface to read whole-slide images.

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
            with os.add_dll_directory("path/to/openslide/bin")):
                import openslide

    .. tab-item:: BioFormats

        `BioFormats <https://www.openmicroscopy.org/bio-formats/>`_ is a standalone Java library
        for reading and writing life sciences image file formats.

        `scyjava <https://github.com/scijava/scyjava>`_ is used to interact with the BioFormats library.

        .. code-block:: bash

            pip install scyjava

    .. tab-item:: CuCIM

        `CuCIM <https://github.com/rapidsai/cucim>`_ is a GPU-accelerated image I/O library.

        .. warning::

            CuCIM support is not available yet.

        Please refer to the `CuCIM GitHub <https://github.com/rapidsai/cucim>`_.