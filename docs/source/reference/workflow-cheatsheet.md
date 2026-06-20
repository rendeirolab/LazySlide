# Workflow cheat sheet

| I want to… | Use | Result |
|---|---|---|
| Open a slide | `zs.open_wsi(...)` | `WSIData` |
| Use sample data | `zs.datasets.sample(...)` | `WSIData` |
| Find tissue with image processing | `zs.pp.find_tissues(...)` | tissue shapes |
| Find tissue with a model | `zs.seg.tissue(...)` | tissue shapes |
| Create tiles | `zs.pp.tile_tissues(...)` | tile shapes and specification |
| Compute tissue properties | `zs.tl.tissue_props(...)` | columns on tissue shapes |
| Build tile neighbors | `zs.pp.tile_graph(...)` | spatial graph metadata |
| Segment cells | `zs.seg.cells(...)` | cell shapes, optionally features |
| Run semantic segmentation | `zs.seg.semantic(...)` | class shapes |
| Extract tile embeddings | `zs.tl.feature_extraction(...)` | feature table |
| Predict from tile images | `zs.tl.tile_prediction(...)` | columns on tile shapes |
| Predict from embeddings | `zs.tl.feature_prediction(...)` | prediction table |
| Aggregate features | `zs.tl.feature_aggregation(...)` | data inside feature table |
| Identify spatial domains | `zs.tl.spatial_domain(...)` | columns on tile shapes |
| Visualize tissue | `zs.pl.tissue(...)` | plot |
| Visualize tile values | `zs.pl.tiles(...)` | plot |
| Visualize spatial shapes | `zs.pl.annotations(...)` | plot |
| Import annotations | `zs.io.load_annotations(...)` | annotation shapes |
| Export annotations | `zs.io.export_annotations(...)` | GeoDataFrame or GeoJSON |
| Score segmentation | `zs.metrics.segmentation` | metric values |

Use the [API reference](../api/index) for complete signatures and the [How-To guides](../how-to/index) for decisions and failure modes.
