from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from scripts.prepare_sparse_camera_dataset import (
  fill_placeholders,
  remove_placeholders
)


def _write_image(filename, value):
  image = np.full((48, 64, 3), value, dtype=np.uint8)
  assert cv2.imwrite(str(filename), image)


def test_fill_and_remove_sparse_camera_placeholders():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    for camera in ("C1", "C2"):
      (root / camera).mkdir()
    _write_image(root / "C1" / "frame_0001.jpg", 20)
    _write_image(root / "C1" / "frame_0002.jpg", 30)
    _write_image(root / "C2" / "frame_0002.jpg", 40)
    _write_image(root / "C2" / "frame_0003.jpg", 50)

    result = fill_placeholders(root, ["C1", "C2"], gray=127)
    assert result["frame_count"] == 3
    assert result["missing_count"] == 2
    assert (root / "C1" / "frame_0003.jpg").is_file()
    assert (root / "C2" / "frame_0001.jpg").is_file()
    assert (root / "placeholder_manifest.json").is_file()

    placeholder = cv2.imread(
      str(root / "C1" / "frame_0003.jpg"), cv2.IMREAD_GRAYSCALE
    )
    assert placeholder.shape == (48, 64)
    assert abs(float(np.mean(placeholder)) - 127.0) < 2.0

    repeated = fill_placeholders(root, ["C1", "C2"], gray=127)
    assert repeated["missing_count"] == 0

    removed = remove_placeholders(root)
    assert removed["removed_count"] == 2
    assert not (root / "C1" / "frame_0003.jpg").exists()
    assert not (root / "C2" / "frame_0001.jpg").exists()
    assert (root / "C1" / "frame_0001.jpg").is_file()
    assert (root / "C2" / "frame_0003.jpg").is_file()
