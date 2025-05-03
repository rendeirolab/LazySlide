.. LazySlide documentation master file, created by
   sphinx-quickstart on Sat Jun 15 08:19:55 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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
      designed to integrate seamlessly with the scverse ecosystem.

      By adopting standardized data structures and APIs familiar to the single-cell and genomics community,
      LazySlide enables intuitive, interoperable, and reproducible workflows for histological analysis.
      It supports a range of tasks from basic preprocessing to advanced deep learning applications,
      facilitating the integration of histopathology into modern computational biology.

.. image:: https://github.com/rendeirolab/LazySlide/blob/main/assets/Figure.png

.. toctree::
    :maxdepth: 1
    :hidden:

    installation
    api/index
    tutorials/index


.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      How to install LazySlide

   .. grid-item-card:: Tutorial
      :link: tutorial/index
      :link-type: doc

      Get started with LazySlide

   .. grid-item-card:: API
      :link: api/index
      :link-type: doc

      LazySlide API documentation
