========
Glossary
========

This glossary contains definitions of terms used throughout the LazySlide documentation.


Histopathology 
===============

.. glossary::
   :sorted:

   ``H&E``
      Hematoxylin and eosin staining, the most common tissue staining method in histopathology.

   patch
      A small rectangular region extracted from a whole slide image for analysis.

   segmentation level
      The resolution level of a WSI used for segmentation. It can be set by using the parameter `level` 
      in LazySlide's tissue segmentation function. The optimal level is automatically determined based 
      on the available memory, but can be manually set for consistency.

   tile
      Synonym for patch - a small image region extracted from a WSI for processing.

   ``WSI``
      Whole Slide Image - a high-resolution digital image of an entire tissue section.


Data structures
===============

.. glossary::
   :sorted:

   ``AnnData``
      Annotated data object used for storing and manipulating single-cell and spatial omics data.

   ``SpatialData``
      A framework for handling spatially resolved omics data, used as LazySlide's data foundation.

   ``WSIData``
      LazySlide's data structure for storing WSI data and associated annotations.

Model types & machine learning
===============================

.. glossary::
   :sorted:

   foundation model
      A large-scale pre-trained model (e.g., UNI, CONCH) that can be adapted for various downstream tasks.

   multimodal model
      A model that can process and integrate multiple types of input data, such as both images and text.

   segmentation model
      A model that performs image segmentation tasks, partitioning images into meaningful regions.

   tile prediction model
      A model that makes predictions on image tiles or patches, typically for classification or regression tasks.

   vision model
      A model designed to process and analyze visual data, typically images.

   zero-shot learning
      Machine learning approach where models make predictions on classes not seen during training.

Image analysis 
===============

.. glossary::
   :sorted:

   instance segmentation
      A type of segmentation that identifies and delineates each individual instance of an object (e.g., each cell) in an image.

   segmentation
      The task of partitioning an image into meaningful regions, such as identifying individual cells or tissue structures.

   semantic segmentation
      A type of segmentation that classifies each pixel in an image into a category (e.g., tumor vs. normal tissue) without distinguishing between individual instances.

