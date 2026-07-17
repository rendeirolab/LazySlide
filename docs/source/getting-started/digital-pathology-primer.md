# Digital pathology primer

Digital pathology turns a glass tissue slide into a **whole-slide image (WSI)**. A WSI can contain billions of pixels, so analysis usually happens at several resolutions and on many smaller image regions.

## The minimum vocabulary

**Pyramid level**
: A WSI stores the same scene at multiple resolutions. Level 0 is normally the highest resolution; larger level numbers are progressively downsampled.

**Microns per pixel (MPP)**
: The physical size represented by one pixel. A smaller MPP means higher resolution. Approximately 0.5 MPP is commonly associated with 20× scanning and 0.25 MPP with 40×, but scanner metadata should be preferred over this approximation.

**Tissue segmentation**
: Separating tissue from the glass background. The result is usually one or more polygons rather than a new cropped image.

**Tile or patch**
: A small rectangular image sampled from the WSI. LazySlide records tile coordinates first and reads their pixels only when an operation needs them.

**Feature embedding**
: A numerical vector describing the appearance of a tile. It can be used for visualization, clustering, prediction, or slide-level aggregation.

**Annotation**
: A point, line, or polygon associated with a region of the slide, often created by a pathologist or another image-analysis tool.

## A typical analysis

```text
slide file -> inspect metadata -> find tissue -> create tiles
           -> quality control -> extract features -> analyze or predict
           -> visualize -> save
```

LazySlide stores the source image, spatial shapes, feature tables, and metadata in a `WSIData` object. Most operations add a result to that object instead of returning a disconnected array. See [How LazySlide stores results](../concepts/data-model) before building a longer pipeline.

## Three resolution decisions

Do not treat pyramid level, magnification, and MPP as interchangeable knobs:

- Use **MPP** when an algorithm or model expects a physical resolution.
- Use a **pyramid level** when directly inspecting or segmenting a particular stored level.
- Use **tile size in pixels** together with MPP to define the physical field of view.

For example, a 256-pixel tile at 0.5 MPP covers 128 microns in each direction. The same tile at 0.25 MPP covers 64 microns.

Next, [analyze your first slide](first-analysis).
