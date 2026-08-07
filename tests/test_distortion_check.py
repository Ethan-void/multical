from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from multical.camera import Camera
from multical.io.distortion_check import (
  export_distortion_checks,
  representative_indices
)


def test_representative_indices_are_deterministic_and_distributed():
  assert representative_indices(0) == []
  assert representative_indices(1) == [0]
  assert representative_indices(2) == [0, 1]
  assert representative_indices(10) == [0, 4, 9]


def test_distortion_check_writes_comparison_images_only():
  width, height = 320, 240
  camera = Camera(
    image_size=(width, height),
    intrinsic=np.array([
      [260.0, 0.0, width / 2.0],
      [0.0, 260.0, height / 2.0],
      [0.0, 0.0, 1.0]
    ]),
    dist=np.array([[0.12, -0.04, 0.001, -0.002, 0.0]])
  )
  images = []
  for offset in range(5):
    image = np.zeros((height, width), dtype=np.uint8)
    for x in range(20 + offset, width, 40):
      cv2.line(image, (x, 0), (x, height - 1), 180, 1)
    for y in range(20, height, 40):
      cv2.line(image, (0, y), (width - 1, y), 220, 1)
    images.append(image)

  with TemporaryDirectory() as temporary:
    legacy_directory = (
      Path(temporary) / "distortion_check" / "CamA"
    )
    legacy_directory.mkdir(parents=True)
    (
      Path(temporary) / "distortion_check" / "manifest.json"
    ).write_text("{}", encoding="utf-8")
    (
      legacy_directory / "view_01_undistorted_full.jpg"
    ).write_bytes(b"legacy")

    manifest, output_directory = export_distortion_checks(
      temporary,
      ["CamA"],
      [camera],
      [images],
      [[
        "CamA/{:06d}.jpg".format(index)
        for index in range(len(images))
      ]],
      stage="calibration"
    )

    assert output_directory == Path(temporary) / "distortion_check"
    assert not (output_directory / "manifest.json").exists()
    assert not (
      legacy_directory / "view_01_undistorted_full.jpg"
    ).exists()
    assert manifest["stage"] == "calibration"
    assert manifest["cameras"]["CamA"]["fisheye"] is False
    assert len(manifest["cameras"]["CamA"]["views"]) == 3
    assert manifest["cameras"]["CamA"]["views"][1]["source_index"] == 2
    assert manifest["cameras"]["CamA"]["views"][2]["source_index"] == 4

    camera_directory = Path(temporary) / "distortion_check" / "CamA"
    comparison = cv2.imread(
      str(camera_directory / "view_01_comparison.jpg")
    )
    assert comparison is not None
    assert comparison.shape == (height, width * 3, 3)
    assert sorted(path.name for path in camera_directory.glob("*.jpg")) == [
      "view_01_comparison.jpg",
      "view_02_comparison.jpg",
      "view_03_comparison.jpg"
    ]
    assert manifest["cameras"]["CamA"]["views"][0] == {
      "source_index": 0,
      "source_image": "CamA/000000.jpg",
      "comparison": "view_01_comparison.jpg"
    }
