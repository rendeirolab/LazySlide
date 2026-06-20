# Multiple slides and scaling

## How do I process a folder of slides?

Put the single-slide workflow in a function with explicit input, output, and configuration. Do not rely on notebook state.

```python
from pathlib import Path
import lazyslide as zs

def process_slide(slide: Path, output_dir: Path) -> Path:
    store = output_dir / f"{slide.stem}.zarr"
    wsi = zs.open_wsi(slide, backed_file=store)
    zs.pp.find_tissues(wsi, level=-1)
    zs.pp.tile_tissues(wsi, 256, mpp=0.5)
    zs.tl.feature_extraction(wsi, "resnet50", batch_size=32)
    wsi.write(overwrite=True)
    return store

slides = sorted(Path("slides").glob("*.svs"))
for slide in slides:
    process_slide(slide, Path("results"))
```

Validate on a small, diverse subset before launching the full cohort.

## How do I make a batch workflow reproducible?

Record at least:

- LazySlide, `wsidata`, and `lazyslide-models` versions;
- source slide identifier and checksum or immutable path;
- reader and source MPP;
- tissue, tiling, QC, model, and aggregation parameters;
- model weight revision and access source;
- hardware and mixed-precision settings;
- status, elapsed time, warnings, and failure message per slide.

Store this configuration in a version-controlled text file and write a per-slide manifest instead of relying only on log output.

## How do I resume an interrupted cohort?

Use one output store and one status record per slide. Before processing, check that all expected output keys exist rather than checking only whether the directory exists.

```python
expected_tables = {"resnet50_tiles"}
complete = expected_tables.issubset(set(wsi.tables))
```

Write to a temporary or clearly marked in-progress location when atomic completion matters.

## How do I use Dask?

Parallelize at the slide level when possible: each worker opens, processes, writes, and closes one slide. Avoid sending open reader objects or large WSI arrays between workers. See the [multiple-slides tutorial](../tutorials/multiple_slides) for a complete distributed example.

## How do I aggregate slide-level features?

First aggregate tile features within each slide:

```python
zs.tl.feature_aggregation(wsi, feature_key="resnet50", encoder="mean")
```

Collect the resulting slide-level vectors into an AnnData or tabular object in the cohort loop, keeping the slide identifier as the observation index. Validate that every slide uses the same feature model and aggregation settings before concatenating the results.

## How do I harmonize slides from different scanners?

Use physical resolution rather than a fixed pyramid level, validate MPP metadata, and hold tile field of view constant. Treat stain normalization as a model- and study-specific choice: compare downstream behavior before applying it to the complete cohort. Preserve scanner and site metadata so residual batch effects can be measured.
