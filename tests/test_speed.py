"""
Benchmark WSI backends

Run with:
python -m fire tests/test_speed.py benchmark
python -m fire tests/test_speed.py benchmark_joint
python -m fire tests/test_speed.py plot
"""

from pathlib import Path
import tempfile
from timeit import default_timer as timer

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from fastai.vision.data import ImageBlock
from fastai.data.block import DataBlock
from torchvision.transforms import ToTensor

from wsi_core import WholeSlideImage
import lazyslide as zs
from lazyslide.loader import FeatureExtractionDataset
from lazyslide.loader import SlidesBalancedLoader


def download_slide(gtex_id: str, slide_path: Path | None = None) -> Path:
    if slide_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".svs", mode="r")
        slide_path = Path(tmp.name)
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{gtex_id}"

    print(f"Downloading {url} to {slide_path}")
    zs.utils.download_file(url, slide_path)
    return slide_path


def prepare_disk(slide_path: Path) -> Path:
    from wsi_core.utils import Path

    slide = WholeSlideImage(slide_path)
    slide.segment(method="manual")
    slide.tile()
    dir_ = Path(slide_path.parent / slide_path.stem)
    dir_.mkdir(exist_ok=True)
    slide.save_tile_images(dir_, attributes=False)
    return dir_


def prepare_clam(slide_path: Path) -> WholeSlideImage:
    slide = WholeSlideImage(slide_path)
    slide.segment(method="manual")
    slide.tile()
    return slide


def prepare_lazyslide(
    slide_path: Path, coords: np.ndarray | None = None, reader="auto"
) -> zs.WSI:
    wsi = zs.WSI(slide_path, reader=reader)
    if coords is None:
        wsi.create_tissue_mask()
        wsi.create_tiles(tile_px=224, mpp=0.5)
    else:
        wsi.new_tiles(coords, 224, 224)
    return wsi


def run_inference(
    dl,
    model_name="resnet18",
    model_hub="pytorch/vision",
    device: str | None = None,
    category: bool = False,
) -> tuple[float, int, np.ndarray]:
    device = device or "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(model_hub, model_name, weights="DEFAULT").eval().to(device)

    n = 0
    time = timer()
    features = list()
    with torch.no_grad():
        for batch in tqdm(dl):
            if category:
                batch = batch[0]
            features.append(model(batch.to(device)).cpu().numpy())
            n += batch.shape[0]
    elapsed = timer() - time
    features = np.concatenate(features)

    return elapsed, n, features


def benchmark(
    model_name: str = "resnet18",
    batch_sizes: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
    gtex_ids: list[str] = [
        "GTEX-OIZI-0826",
        "GTEX-15CHS-0426",
    ],
):
    res = list()
    for gtex_id in gtex_ids:
        slide_path = Path(".") / f"{gtex_id}.svs"
        if not slide_path.exists():
            url = f"https://brd.nci.nih.gov/brd/imagedownload/{gtex_id}"
            zs.utils.download_file(url, slide_path)
        file_size = slide_path.stat().st_size / 1e6
        # print(file_size)

        for batch_size in batch_sizes:
            for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
                # CLAM
                cl = prepare_clam(slide_path)
                dl = cl.as_data_loader(batch_size=batch_size)
                # print(f"Running inference on {slide_path}")
                time, n, _ = run_inference(dl, model_name, device=device)
                # print(f"CLAM: {time} seconds")
                res.append(
                    [
                        "CLAM",
                        gtex_id,
                        file_size,
                        batch_size,
                        device,
                        len(dl),
                        time,
                        n,
                    ]
                )
                tqdm.write(str(res[-1]))

                # LazySlide
                for reader in ["openslide", "vips"]:
                    lz = prepare_lazyslide(
                        slide_path, cl.get_tile_coordinates(), reader=reader
                    )
                    ds = FeatureExtractionDataset(lz, resize=224, color_normalize=None)
                    dl = DataLoader(dataset=ds, batch_size=batch_size)
                    # print(f"Running inference on {slide_path}")
                    time, n, _ = run_inference(dl, model_name, device=device)
                    # print(f"Lazyslide: {time} seconds")
                    res.append(
                        [
                            f"LazySlide ({reader})",
                            gtex_id,
                            file_size,
                            batch_size,
                            device,
                            len(dl),
                            time,
                            n,
                        ]
                    )
                    tqdm.write(str(res[-1]))

                # Disk
                dir_ = Path(slide_path.parent / slide_path.stem)
                if not dir_.exists():
                    dir_ = prepare_disk(slide_path)
                db = DataBlock(
                    blocks=ImageBlock,
                    item_tfms=[ToTensor()],
                )
                dl = db.dataloaders(list(dir_.glob("*.jpg")), bs=batch_size)
                # print(f"Running inference on {slide_path}")
                time, n, _ = run_inference(
                    dl[0], model_name, device=device, category=True
                )
                # print(f"Disk: {time} seconds")
                res.append(
                    [
                        "Disk",
                        gtex_id,
                        file_size,
                        batch_size,
                        device,
                        len(dl),
                        time,
                        n,
                    ]
                )
                tqdm.write(str(res[-1]))

    df = pd.DataFrame(
        res,
        columns=[
            "method",
            "slide_id",
            "file_size",
            "batch_size",
            "device",
            "batch_count",
            "time",
            "total_images",
        ],
    )
    # df.loc[df["method"] == "LazySlide", "method"] = "LazySlide (vips)"
    df.to_csv("speed.csv", index=False)


def benchmark_joint(
    model_name: str = "resnet18",
    batch_sizes: list[int] = [8, 32, 128],
    slide_sizes: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
    gtex_id_file: Path = Path("/projects/histopath/sc_slide_list.txt"),
):
    gtex_ids = gtex_id_file.open().read().strip().split("\n")
    # Download slides
    slide_paths = list()
    for gtex_id in tqdm(gtex_ids[: max(slide_sizes)]):
        slide_path = Path(".") / f"{gtex_id}.svs"
        if not slide_path.exists():
            of = Path(f"/projects/histopath/data/gtex/svs/{gtex_id}.svs")
            if of.exists():
                of.symlink_to(slide_path)
            else:
                url = f"https://brd.nci.nih.gov/brd/imagedownload/{gtex_id}"
                zs.utils.download_file(url, slide_path)
        slide_paths.append(slide_path)

    res = list()
    for reader in ["openslide"]:
        lzs = [
            prepare_lazyslide(slide_path, reader=reader)
            for slide_path in tqdm(slide_paths)
        ]
        for batch_size in batch_sizes:
            for slides in slide_sizes:
                # TODO: check this makes shuffled batches with images from multiple slides
                dl = SlidesBalancedLoader(
                    lzs[:slides], max_taken=480, batch_size=batch_size
                )
                for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
                    s = [f"LazySlide ({reader})", batch_size, slides, device, len(dl)]
                    if s in [x[:-2] for x in res]:
                        print("Skipping")
                        continue
                    time, n, _ = run_inference(dl, model_name, device=device)
                    res.append(
                        [
                            f"LazySlide ({reader})",
                            batch_size,
                            slides,
                            device,
                            len(dl),
                            time,
                            n,
                        ]
                    )
                    tqdm.write(str(res[-1]))

                    df = pd.DataFrame(
                        res,
                        columns=[
                            "method",
                            "batch_size",
                            "slide_count",
                            "device",
                            "batch_count",
                            "time",
                            "total_images",
                        ],
                    )
                    df.to_csv("speed_join.csv", index=False)


def plot():
    df = pd.read_csv("speed.csv")
    df = df.query("batch_size < 512")
    # df["time_per_image"] = df["time"] / (df["batch_size"] * df["batch_count"])
    df["time_per_image"] = df["time"] / (df["total_images"])
    df["time_total_adj"] = df["time_per_image"] * df["total_images"]

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="batch_size",
        y="time_total_adj",
        hue="method",
        style="device",
        markers=True,
        ax=ax,
    )
    ax.loglog()
    ax.set(ylabel="Time per slide (seconds)", xlabel="Batch size")
    ax.set_title("WSI backend speed comparison")
    fig.savefig("speed.svg", dpi=300, bbox_inches="tight")
    fig.savefig("speed.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="batch_size",
        y="time_per_image",
        hue="method",
        style="device",
        markers=True,
        ax=ax,
    )
    ax.loglog()
    ax.set(ylabel="Time per image (seconds)", xlabel="Batch size")
    ax.set_title("WSI backend speed comparison")
    fig.savefig("speed.per_image.svg", dpi=300, bbox_inches="tight")
    fig.savefig("speed.per_image.png", dpi=300, bbox_inches="tight")

    if not Path("speed_join.csv").exists():
        return
    df = pd.read_csv("speed_join.csv")

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="slide_count",
        y="time",
        hue="batch_size",
        style="device",
        markers=True,
        ax=ax,
    )
    ax.set(xlabel="Number of slides", ylabel="Time for all slides (seconds)")
    ax.set_title("WSI backend speed comparison")
    fig.savefig("speed_join.svg", dpi=300, bbox_inches="tight")
    ax.loglog()
    fig.savefig("speed_join.loglog.svg", dpi=300, bbox_inches="tight")
