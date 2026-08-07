from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from multical.board import load_config
from tests.generate_calibration_simulation import generate


def test_generator_supports_all_board_types_and_modes():
  cases = [
    ("checkerboard", "plane", "combined"),
    ("charuco", "l", "split"),
    ("aprilgrid", "triangle", "combined")
  ]
  with TemporaryDirectory() as temporary:
    for board_type, board_style, dataset_mode in cases:
      output = Path(temporary) / "{}-{}-{}".format(
        board_type, board_style, dataset_mode
      )
      result = generate(
        output_dir=output,
        board_type=board_type,
        board_style=board_style,
        dataset_mode=dataset_mode,
        intrinsic_frames=1,
        extrinsic_frames=2,
        seed=4
      )

      assert Path(result["commands"]).is_file()
      assert (output / "boards.yaml").is_file()
      assert (output / "ground_truth.json").is_file()
      assert result["overview_files"]
      assert all(
        Path(path).is_file() for path in result["overview_files"]
      )
      assert all(
        cv2.imread(path) is not None for path in result["overview_files"]
      )
      assert len(load_config(output / "boards.yaml")) == {
        "plane": 1, "l": 2, "triangle": 3
      }[board_style]

      if dataset_mode == "combined":
        assert len(list((output / "C1").glob("*.jpg"))) == 8
        assert result["overview_image_count"] == 48
        assert (
          output / "overviews" / "overview_combined_001.jpg"
        ).is_file()
      else:
        assert len(list((output / "intrinsic" / "C1").glob("*.jpg"))) == 1
        assert len(list((output / "extrinsic" / "C1").glob("*.jpg"))) == 2
        assert result["overview_image_count"] == 18
        assert (
          output / "overviews" / "overview_intrinsic_001.jpg"
        ).is_file()
        assert (
          output / "overviews" / "overview_extrinsic_001.jpg"
        ).is_file()


def test_aprilgrid_draw_and_opencv_detection():
  with TemporaryDirectory() as temporary:
    output = Path(temporary) / "april"
    generate(
      output_dir=output,
      board_type="aprilgrid",
      board_style="plane",
      dataset_mode="split",
      intrinsic_frames=1,
      extrinsic_frames=1
    )
    board = next(iter(load_config(output / "boards.yaml").values()))
    image = board.draw(pixels_mm=3, margin_mm=0)
    detection = board.detect(image)

    assert detection.ids.size == board.num_points
    assert board.has_min_detections(detection)


def test_checkerboard_rejects_uncoded_multiface_styles():
  with TemporaryDirectory() as temporary:
    try:
      generate(
        output_dir=Path(temporary) / "invalid",
        board_type="checkerboard",
        board_style="l",
        dataset_mode="combined",
        intrinsic_frames=1,
        extrinsic_frames=1
      )
    except ValueError as error:
      assert "coded faces" in str(error)
    else:
      raise AssertionError("checkerboard L target should be rejected")
