# Slide readers and formats

LazySlide delegates WSI reading to `wsidata`. Format support therefore depends on the installed reader and the exact vendor encoding.

| Reader | Typical use | Additional requirements |
|---|---|---|
| TiffSlide | TIFF-based bright-field WSI formats | Python package; installed by default |
| OpenSlide | Common vendor bright-field formats | OpenSlide library and Python bindings |
| fastslide | High-performance native whole-slide image reading | `fastslide` |
| Bio-Formats | Broad microscopy format coverage | Java and `scyjava` |
| cuCIM | Supported GPU-accelerated image I/O | Compatible NVIDIA/CUDA environment |
| pyisyntax | Philips iSyntax pathology images | `pyisyntax` and libisyntax support |
| pylibCZIrw | Zeiss CZI, including native JPEG-XR decoding | Platform-compatible `pylibCZIrw` wheel |

The common extensions include `.svs`, `.ndpi`, `.mrxs`, and TIFF variants, but an extension is not a complete compatibility guarantee.

## Verify a file

```python
import lazyslide as zs

wsi = zs.open_wsi("path/to/slide")
print(wsi.properties)
print(wsi.fetch.pyramids())
```

Inspect the selected reader, dimensions, physical pixel size, magnification, and pyramid levels. Test representative files from every scanner and acquisition protocol used in a cohort.

## Annotation formats

`zs.io.load_annotations` reads geospatial files supported by GeoPandas, including QuPath GeoJSON, and Hamamatsu `.ndpa` files. Export currently targets QuPath-compatible GeoJSON. See [Annotations](../how-to/annotations).

For reader installation commands, see [Installation](../installation).
