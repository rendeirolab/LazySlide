LazySlide: Accessible and interoperable whole slide image analysis
==================================================================

.. grid:: 1 2 2 2

   .. grid-item::
       :columns: 12 4 4 4

       .. image:: _static/logo@3x.png
          :align: center
          :width: 150px

   .. grid-item::
      :columns: 12 8 8 8
      :child-align: center

      **LasySlide** LazySlide is a Python framework for whole slide image (WSI) analysis,
      designed to integrate seamlessly with the `scverse`_ ecosystem.

      By adopting standardized data structures and APIs familiar to the single-cell and genomics community,
      LazySlide enables intuitive, interoperable, and reproducible workflows for histological analysis.
      It supports a range of tasks from basic preprocessing to advanced deep learning applications,
      facilitating the integration of histopathology into modern computational biology.

Key features
------------

* **Interoperability**: Built on top of `SpatialData`_, ensuring compatibility with scverse tools like `Scanpy`_, `Anndata`_, and `Squidpy`_. Check out `WSIData`_ for more details.
* **Accessibility**: User-friendly APIs that cater to both beginners and experts in digital pathology.
* **Scalability**: Efficient handling of large WSIs, enabling high-throughput analyses.
* **Multimodal integration**: Combine histological data with transcriptomics, genomics, and textual annotations.
* **Foundation model support**: Native integration with state-of-the-art models (e.g., UNI, CONCH, Gigapath, Virchow) for tasks like zero-shot classification and captioning.
* **Deep learning ready**: Provides PyTorch dataloaders for seamless integration into machine learning pipelines.

Whether you're a novice in digital pathology or an expert computational biologist, LazySlide provides a scalable and modular foundation to accelerate AI-driven discovery in tissue biology and pathology.

.. image:: https://github.com/rendeirolab/LazySlide/raw/main/assets/Figure.png

|

.. toctree::
    :maxdepth: 1
    :hidden:

    installation
    tutorials/index
    avail_models
    api/index
    contributing
    contributors
    references


.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      How to install LazySlide

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Get started with LazySlide

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      Contribute to Lazyslide

   .. grid-item-card:: Contributors
      :link: contributors
      :link-type: doc

      The team behind LazySlide

.. _scverse: https://scverse.org/
.. _WSIData: https://wsidata.readthedocs.io/
.. _SpatialData: https://spatialdata.scverse.org/
.. _Scanpy: https://scanpy.readthedocs.io/
.. _Anndata: https://anndata.readthedocs.io/
.. _Squidpy: https://squidpy.readthedocs.io/

