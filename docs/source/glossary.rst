========
Glossary
========

This glossary contains definitions of terms used throughout the LazySlide documentation.

.. glossary::
   :sorted:

   ``AnnData``
      Annotated data object used for storing and manipulating single-cell and spatial omics data.

   Foundation Model
      A large-scale pre-trained model (e.g., UNI, CONCH) that can be adapted for various downstream tasks.

   ``H&E``
      Hematoxylin and eosin staining, the most common tissue staining method in histopathology.

   Instance Segmentation
      A type of segmentation that identifies and delineates each individual instance of an object (e.g., each cell) in an image.

   Multimodal Model
      A model that can process and integrate multiple types of input data, such as both images and text.

   Patch
      A small rectangular region extracted from a whole slide image for analysis.

   Segmentation
      The task of partitioning an image into meaningful regions, such as identifying individual cells or tissue structures.

   Segmentation Model
      A model that performs image segmentation tasks, partitioning images into meaningful regions.

   Semantic Segmentation
      A type of segmentation that classifies each pixel in an image into a category (e.g., tumor vs. normal tissue) without distinguishing between individual instances.

   ``SpatialData``
      A framework for handling spatially resolved omics data, used as LazySlide's data foundation.

   Tile
      Synonym for patch - a small image region extracted from a WSI for processing.

   Tile Prediction Model
      A model that makes predictions on image tiles or patches, typically for classification or regression tasks.

   Vision Model
      A model designed to process and analyze visual data, typically images.

   ``WSI``
      Whole Slide Image - a high-resolution digital image of an entire tissue section.

   ``WSIData``
      LazySlide's data structure for storing WSI data and associated annotations.

   Zero-shot Learning
      Machine learning approach where models make predictions on classes not seen during training.


Adding New Terms
================

To add a new term to this glossary:

1. Add the term in alphabetical order within the ``.. glossary::`` directive
2. Use the format::

      Term Name
         Definition of the term goes here. Can span multiple lines
         if needed.

3. Reference the term in documentation using ``:term:`Term Name```

