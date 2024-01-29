import numpy as np
import pandas as pd

from lazyslide.utils import TileOps
from lazyslide.io import IOGroup, H5ZSFile


class TestIOGroup:
    def test_init(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        assert iogroup.file.exists()

    def test_save_dataset(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        iogroup._save_dataset("test", np.array([1, 2, 3]))
        assert iogroup._read_dataset("test").tolist() == [1, 2, 3]

    def test_save_dataset_with_group(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        iogroup._save_dataset("test", np.array([1, 2, 3]), group="group")
        assert iogroup._read_dataset("test", group="group").tolist() == [1, 2, 3]

    def test_save_dataframe(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["7", "8", "9"]})
        iogroup._save_dataframe("test", df)
        assert iogroup._read_dataframe("test").equals(df)

    def test_save_attributes(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        attrs = {"int": 1, "float": 2.0, "string": "text", "null": None}
        iogroup._save_attributes("test_attrs", attrs)
        assert iogroup._read_attributes("test_attrs") == attrs

    def test_delete_group(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        iogroup._save_dataset("test", np.array([1, 2, 3]), group="group")
        assert iogroup._read_dataset("test", group="group").tolist() == [1, 2, 3]
        iogroup._delete_group("group")
        assert iogroup._read_dataset("test", group="group") is None

    def test_delete_dataset(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        iogroup._save_dataset("test", np.array([1, 2, 3]), group="group")
        assert iogroup._read_dataset("test", group="group").tolist() == [1, 2, 3]
        iogroup._delete_dataset("test", group="group")
        assert iogroup._read_dataset("test", group="group") is None

    def test_get_group_keys(self, tmp_path):
        iogroup = IOGroup(tmp_path / "test.h5")
        iogroup._save_dataset("test1", np.array([1, 2, 3]), group="group")
        iogroup._save_dataset("test2", np.array([1, 2, 3]), group="group")
        iogroup._save_dataset("test3", np.array([1, 2, 3]), group="group")
        assert iogroup._get_group_keys("group") == ["test1", "test2", "test3"]


class TestH5ZSFile:
    def test_init(self, tmp_path):
        h5zsfile = H5ZSFile(tmp_path / "test.h5")
        assert h5zsfile.file.exists()

    def test_set_coords(self, tmp_path):
        h5zsfile = H5ZSFile(tmp_path / "test.h5")
        coords = np.array([[0, 0], [1, 1], [10, 10]])
        h5zsfile.set_coords(coords)
        assert np.array_equal(h5zsfile.get_coords(), coords)
        assert np.array_equal(h5zsfile.get_index(), np.arange(3))

    def test_set_table(self, tmp_path):
        h5zsfile = H5ZSFile(tmp_path / "test.h5")
        table = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["7", "8", "9"]})
        h5zsfile.set_coords(np.array([[0, 0], [1, 1], [10, 10]]))
        h5zsfile.set_table(table)
        assert h5zsfile.get_table().equals(table)

    def test_set_tile_ops(self, tmp_path):
        h5zsfile = H5ZSFile(tmp_path / "test.h5")
        tile_ops = TileOps(
            level=0,
            downsample=1,
            mpp=0.25,
            height=100,
            width=100,
            ops_height=100,
            ops_width=100,
            mask_name="mask",
        )
        h5zsfile.set_tile_ops(tile_ops)
        assert h5zsfile.get_tile_ops() == tile_ops

    def test_set_masks(self, tmp_path):
        h5zsfile = H5ZSFile(tmp_path / "test.h5")
        masks = {"mask": np.array([[0, 0], [1, 1], [10, 10]])}
        masks_levels = {"mask": 0}
        h5zsfile.set_masks(masks, masks_levels)
        assert np.array_equal(h5zsfile.get_masks()[0]["mask"], masks["mask"])
        assert h5zsfile.get_masks()[1]["mask"] == masks_levels["mask"]

    def test_set_contours_holes(self, tmp_path):
        h5zsfile = H5ZSFile(tmp_path / "test.h5")
        contours = [np.array([[0, 0], [1, 1], [10, 10]])]
        holes = [np.array([[0, 0], [1, 1], [10, 10]])]
        h5zsfile.set_contours_holes(contours, holes)
        assert np.array_equal(h5zsfile.get_contours_holes()[0], contours)
        assert np.array_equal(h5zsfile.get_contours_holes()[1], holes)

    def test_set_md_field(self, tmp_path):
        h5zsfile = H5ZSFile(tmp_path / "test.h5")
        field = "test"
        value = np.array([1, 2, 3])
        h5zsfile.set_coords(np.array([[0, 0], [1, 1], [10, 10]]))
        h5zsfile.set_feature_field(field, value)
        assert np.array_equal(h5zsfile.get_feature_field(field), value)
