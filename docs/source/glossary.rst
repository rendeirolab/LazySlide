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
      Hematoxylin stains cell nuclei blue/purple (binding to DNA/RNA) and eosin stains cytoplasm, extracellular matrix, and other structures pink/red.
      H&E slides are the standard for disease diagnosis and are widely used in digital pathology analysis.

   patch
      A small rectangular region extracted from a whole slide image for analysis. 
      They enable efficient processing of high-resolution :term:`WSIs <WSI>` by breaking them into manageable sub-images

   segmentation level
      The resolution level of a WSI used for segmentation. Level 0 has the higher resolution, 
      and higher levels have progressively lower resolution. It can be set by using the parameter `level` 
      in LazySlide's tissue segmentation function. The optimal level is automatically determined based 
      on the available memory, but can be manually set for consistency.

   magnification
       Zoom level of a :term:`WSI`. It is related to the level of detail in a WSI, 
       typically measured in terms of how many times the original tissue is magnified (e.g., 20x, 40x). Higher magnification levels provide more detail but require more computational resources for analysis.
   
   pyramid
   pyramid structure
       A multi-resolution representation of a WSI, storing images at different zoom levels 
      
   tile
      Synonym for :term:`patch` - a small image region extracted from a WSI for processing.
    
   ``WSI``
      Whole Slide Image - a high-resolution digital image of an entire tissue section.

    contours
       Closed boundaries outlining tissue regions in a WSI (e.g., tumor areas, stroma),
       often represented as polygons. 

    holes
       Empty spaces within a :term:`contour` (e.g., artifacts, fat deposits, or non-tissue regions like lumens).
       They are typically excluded from further analysis.


Data structures
===============

.. glossary::
   :sorted:

   ``AnnData``
      Annotated data object used for storing and manipulating single-cell and spatial omics data. See `Anndata`_ documentation for more details.

   feature embedding
      A numerical representation of data (such as image :term:`patches <patch>`) in a lower-dimensional space, typically produced by neural networks. 
      These embeddings capture semantic information and can be used for downstream tasks like clustering, classification, or similarity search.

   features
      Numerical representations or measurements extracted from data, such as pixel intensities, texture descriptors, or learned representations from neural networks. 
      In histopathology, features can describe visual properties of :term:`patches <patch>` or geometric properties of tissue regions.

   geometric features
      Quantitative measurements of shape and spatial properties of objects, such as area, perimeter, :term:`convexity`, solidity, and eccentricity. 
      In LazySlide, these are computed for tissue :term:`contours` and provide morphological characterization of tissue regions.

   ``Hugging Face``
      A popular platform and ecosystem for sharing and deploying machine learning models, particularly natural language processing and computer vision models. 
      Many :term:`foundation models <foundation model>` in LazySlide are hosted on Hugging Face.

   ``SpatialData``
      A framework for handling spatially resolved omics data, used as LazySlide's data foundation. See `SpatialData`_ documentation for more details.

   ``WSIData``
      LazySlide's data structure for storing WSI data and associated annotations. See `WSIData`_ documentation for more details.

Model types & machine learning
===============================

.. glossary::
   :sorted:

   convexity
      A geometric property measuring how close a shape is to being convex, calculated as the ratio of convex hull area to actual area. 
      Values close to 1 indicate more convex shapes, used in tissue morphology analysis.

   foundation model
      A large-scale pre-trained model (e.g., UNI, CONCH) that can be adapted for various downstream tasks.

   ``Leiden clustering``
      A community detection algorithm for clustering nodes in graphs, commonly used for spatial clustering of :term:`tiles <tile>` or cells. 
      Often applied after constructing a :term:`spatial tile graph` to identify spatially coherent regions.

   multimodal model
      A model that can process and integrate multiple types of input data, such as both images and text.

   pretrained model
      A neural network model that has been trained on a large dataset and can be fine-tuned or used as a feature extractor for new tasks. 
      :term:`Foundation models <foundation model>` are a type of pretrained model designed for broad applicability.

   segmentation model
      A model that performs image segmentation tasks, partitioning images into meaningful regions.

   tile prediction model
      A model that makes predictions on image tiles or patches, typically for classification or regression tasks.

   transform function
      A preprocessing function that converts raw image data into the format expected by a :term:`vision model`, typically including normalization, resizing, and tensor conversion. 
      Each model defines its own transform function via the `get_transform()` method.

   vision model
      A model designed to process and analyze visual data, typically images.

   zero-shot learning
      Machine learning approach where models make predictions on classes not seen during training.

Image analysis 
===============

.. glossary::
   :sorted:

   Delaunay triangulation
      A geometric method for creating a triangular mesh from a set of points, where no point lies inside the circumcircle of any triangle. 
      Used in :term:`spatial tile graph` construction to define neighborhood relationships between :term:`tiles <tile>` based on natural 
      spatial connectivity rather than fixed distance thresholds. Can be set by using the parameter `use_delaunay` in `pp.tile_graph`.

   feature aggregation
      The process of combining :term:`features` from multiple sources or spatial locations, such as aggregating :term:`patch` features within tissue regions 
      or combining features from neighboring :term:`tiles <tile>` in a :term:`spatial tile graph`.

   feature extraction
      The process of computing numerical representations (:term:`features`) from raw data, such as extracting embeddings from image :term:`patches <patch>` 
      using :term:`vision models <vision model>` or computing :term:`geometric features` from tissue :term:`contours`.

   instance segmentation
      A type of segmentation that identifies and delineates each individual instance of an object (e.g., each cell) in an image.

   neighborhood graph construction
      The process of building a graph structure that represents spatial relationships between objects (such as :term:`tiles <tile>` or cells), 
      where edges connect spatially proximate nodes. Used as a foundation for spatial analysis methods.

   segmentation
      The task of partitioning an image into meaningful regions, such as identifying individual cells or tissue structures.

   semantic segmentation
      A type of segmentation that classifies each pixel in an image into a category (e.g., tumor vs. normal tissue) without distinguishing between individual instances.

   spatial feature smoothing
      A technique for reducing noise and creating spatial coherence in :term:`feature` maps by averaging or interpolating values across neighboring locations 
      in a :term:`spatial tile graph`. Helps create smoother spatial patterns and reduce the impact of outlier measurements.

   spatial tile graph
      A spatial graph structure representing neighborhood relationships between :term:`tiles <tile>` extracted from a :term:`WSI`. 
      Each tile becomes a node, and edges connect spatially adjacent tiles based on distance or :term:`Delaunay triangulation`. 
      This transformation enables analysis methods based on graph theory and graph neural networks.

   unsupervised spatial domain segmentation
      A machine learning approach for automatically identifying distinct spatial regions or domains in tissue without prior labeled examples, 
      typically using clustering methods applied to spatial :term:`features` and :term:`neighborhood graph construction`.

.. _scverse: https://scverse.org/
.. _WSIData: https://wsidata.readthedocs.io/
.. _SpatialData: https://spatialdata.scverse.org/
.. _Anndata: https://anndata.readthedocs.io/


