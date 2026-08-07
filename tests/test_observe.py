from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from multical.app.observe import (
  append_automatic_point,
  automatic_point_work_items,
  annotate_observations,
  compose_mosaic,
  compose_point_sidebar,
  load_existing_observations,
  map_mosaic_click,
  nudge_observation,
  observation_work_items,
  sidebar_hit_test,
  write_observations
)


def test_mosaic_click_maps_to_original_pixels():
  first = np.zeros((480, 640, 3), dtype=np.uint8)
  second = np.zeros((720, 1280, 3), dtype=np.uint8)
  _, placements = compose_mosaic(
    ["C1", "C2"], [first, second], {}, columns=2, tile_width=400
  )

  first_click = map_mosaic_click(
    placements,
    int(placements[0]["x"] + 320 * placements[0]["scale"]),
    int(placements[0]["y"] + 240 * placements[0]["scale"])
  )
  second_click = map_mosaic_click(
    placements,
    int(placements[1]["x"] + 640 * placements[1]["scale"]),
    int(placements[1]["y"] + 360 * placements[1]["scale"])
  )
  assert first_click[0] == "C1"
  assert np.allclose(first_click[1], [320, 240], atol=2)
  assert second_click[0] == "C2"
  assert np.allclose(second_click[1], [640, 360], atol=2)


def test_magnifier_click_refines_original_pixel():
  image = np.zeros((480, 640, 3), dtype=np.uint8)
  _, placements = compose_mosaic(
    ["C1"], [image], {"C1": [321.0, 241.0]},
    columns=1, tile_width=400
  )
  magnifier = placements[0]["magnifier"]
  expected = [325.0, 244.0]
  click_x = int(round(
    magnifier["x"] +
    (expected[0] - magnifier["raw_x"]) *
    magnifier["size"] / magnifier["raw_width"]
  ))
  click_y = int(round(
    magnifier["y"] +
    (expected[1] - magnifier["raw_y"]) *
    magnifier["size"] / magnifier["raw_height"]
  ))
  camera, refined = map_mosaic_click(
    placements, click_x, click_y
  )
  assert camera == "C1"
  assert np.allclose(refined, expected, atol=0.3)


def test_keyboard_nudge_moves_and_clamps_raw_pixel():
  image = np.zeros((48, 64, 3), dtype=np.uint8)
  _, placements = compose_mosaic(
    ["C1"], [image], {"C1": [63.0, 0.0]},
    columns=1, tile_width=400
  )
  observations = {"C1": [63.0, 0.0]}
  assert nudge_observation(
    observations, "C1", placements, 1.0, -1.0
  )
  assert observations["C1"] == [63.0, 0.0]
  assert nudge_observation(
    observations, "C1", placements, -1.0, 1.0
  )
  assert observations["C1"] == [62.0, 1.0]


def test_observation_yaml_is_triangulate_compatible_and_resumable():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    output_path = root / "observations.yaml"
    annotations = {
      "000001.jpg": {
        "C1": [100.12345, 200.98765],
        "C2": [110.0, 201.0]
      },
      "000002.jpg": {
        "C1": [120.0, 220.0]
      }
    }
    output, destination = write_observations(
      output_path,
      root,
      ["C1", "C2"],
      ["000001.jpg", "000002.jpg"],
      annotations
    )

    assert destination == output_path.resolve()
    assert len(output["frames"]) == 1
    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert loaded["frames"][0]["observations"]["C1"] == [
      100.123, 200.988
    ]
    resumed = load_existing_observations(output_path)
    assert set(resumed) == {"000001.jpg"}
    assert set(resumed["000001.jpg"]) == {"C1", "C2"}


def test_single_camera_observation_can_be_saved_for_click_testing():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    output_path = root / "single.yaml"
    output, _ = write_observations(
      output_path,
      root / "test.jpg",
      ["C1"],
      ["test.jpg"],
      {"test.jpg": {"C1": [12.5, 34.5]}},
      minimum_cameras=1
    )
    assert output["frames"] == [{
      "frame": "test.jpg",
      "observations": {"C1": [12.5, 34.5]}
    }]


def test_multiple_named_points_share_one_source_frame():
  items = observation_work_items(
    ["000000.jpg"], ["P01", "P02", "P03", "P04"]
  )
  assert items == [
    {
      "frame": "P01",
      "source_frame": "000000.jpg",
      "point_name": "P01"
    },
    {
      "frame": "P02",
      "source_frame": "000000.jpg",
      "point_name": "P02"
    },
    {
      "frame": "P03",
      "source_frame": "000000.jpg",
      "point_name": "P03"
    },
    {
      "frame": "P04",
      "source_frame": "000000.jpg",
      "point_name": "P04"
    }
  ]

  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    output_path = root / "observations.yaml"
    annotations = {
      item["frame"]: {
        "C1": [100.0, 200.0],
        "C2": [110.0, 201.0]
      }
      for item in items
    }
    output, _ = write_observations(
      output_path,
      root,
      ["C1", "C2"],
      [item["frame"] for item in items],
      annotations,
      source_frames={
        item["frame"]: item["source_frame"] for item in items
      }
    )

    assert [frame["frame"] for frame in output["frames"]] == [
      "P01", "P02", "P03", "P04"
    ]
    assert all(
      frame["source_frame"] == "000000.jpg"
      for frame in output["frames"]
    )
    assert set(load_existing_observations(output_path)) == {
      "P01", "P02", "P03", "P04"
    }


def test_automatic_point_names_start_and_resume_sequentially():
  items, index = automatic_point_work_items("000000.jpg", {})
  assert index == 0
  assert items == [{
    "frame": "P01",
    "source_frame": "000000.jpg",
    "point_name": "P01"
  }]
  append_automatic_point(items)
  assert items[-1]["point_name"] == "P02"

  resumed, index = automatic_point_work_items(
    "000000.jpg",
    {
      "P01": {"C1": [1.0, 2.0], "C2": [2.0, 2.0]},
      "P02": {"C1": [3.0, 4.0], "C2": [4.0, 4.0]}
    }
  )
  assert index == 2
  assert [item["point_name"] for item in resumed] == [
    "P01", "P02", "P03"
  ]


def test_saved_point_sidebar_is_clickable():
  canvas = np.zeros((300, 500, 3), dtype=np.uint8)
  combined, hitboxes = compose_point_sidebar(
    canvas,
    ["P01", "P02", "P03"],
    {
      "P01": {"C1": [1.0, 2.0]},
      "P02": {"C1": [3.0, 4.0]}
    },
    current_index=2
  )
  assert combined.shape == (300, 500 + 190, 3)
  assert [value["point_name"] for value in hitboxes] == ["P01", "P02"]
  p02 = hitboxes[1]
  assert sidebar_hit_test(
    hitboxes,
    (p02["x0"] + p02["x1"]) // 2,
    (p02["y0"] + p02["y1"]) // 2
  ) == 1
  assert sidebar_hit_test(hitboxes, 10, 10) is None


def test_missing_single_image_path_has_clear_error():
  with TemporaryDirectory() as temporary:
    missing = Path(temporary) / "missing.jpg"
    try:
      annotate_observations(
        missing, ["C1"], Path(temporary) / "output.yaml"
      )
    except ValueError as error:
      assert "image_path does not exist" in str(error)
      assert str(missing) in str(error)
    else:
      raise AssertionError("missing image path should fail")
