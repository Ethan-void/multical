import json
from pathlib import Path

import numpy as np
import pytest

from multical.app.worldgroups import merge_world_groups
from multical.io.calibration_utils import (
  camera_pose_matrices,
  transform_from_json,
  transform_to_json
)


def _camera(index):
  return {
    "model": "standard",
    "K": [[1000.0 + index, 0.0, 640.0],
          [0.0, 1000.0 + index, 360.0],
          [0.0, 0.0, 1.0]],
    "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
    "image_size": [1280, 720]
  }


def _translation(x, y, z):
  transform = np.eye(4)
  transform[:3, 3] = [x, y, z]
  return transform


def _write_json(path, value):
  path.write_text(json.dumps(value), encoding="utf-8")
  return str(path)


def _group_calibration(cameras, first, second, baseline):
  return {
    "cameras": cameras,
    "camera_poses": {
      first: transform_to_json(np.eye(4)),
      "{}_to_{}".format(second, first): transform_to_json(baseline)
    }
  }


def _world_result(world_to_group, baseline, first, second):
  world_to_first = world_to_group
  world_to_second = baseline @ world_to_group
  return {
    "convention": (
      "x_camera = R_world_to_camera * x_world + T_world_to_camera"
    ),
    "world_units": "meters",
    "joint": {"inlier_count": 8, "observation_count": 10},
    "cameras": {
      first: {"world_to_camera": transform_to_json(world_to_first)},
      second: {"world_to_camera": transform_to_json(world_to_second)}
    }
  }


def _inputs(tmp_path):
  cameras = {"cam{}".format(index): _camera(index) for index in range(4)}
  intrinsic = _write_json(
    tmp_path / "intrinsic.json", {"cameras": cameras}
  )
  baseline01 = _translation(1.2, 0.0, 0.0)
  baseline23 = _translation(-1.1, 0.1, 0.0)
  calibration01 = _write_json(
    tmp_path / "calibration01.json",
    _group_calibration(
      {name: cameras[name] for name in ("cam0", "cam1")},
      "cam0", "cam1", baseline01
    )
  )
  calibration23 = _write_json(
    tmp_path / "calibration23.json",
    _group_calibration(
      {name: cameras[name] for name in ("cam2", "cam3")},
      "cam2", "cam3", baseline23
    )
  )
  world01_data = _world_result(
    _translation(0.0, 2.0, 8.0), baseline01, "cam0", "cam1"
  )
  world23_data = _world_result(
    _translation(20.0, -2.0, 8.5), baseline23, "cam2", "cam3"
  )
  world01 = _write_json(tmp_path / "world01.json", world01_data)
  world23 = _write_json(tmp_path / "world23.json", world23_data)
  return (
    intrinsic,
    [calibration01, calibration23],
    [world01, world23],
    world01_data,
    world23_data
  )


def test_merge_world_groups_and_derive_global_calibration(tmp_path):
  intrinsic, calibrations, worlds, world01, world23 = _inputs(tmp_path)
  output_path = tmp_path / "world.json"
  calibration_path = tmp_path / "combined.json"

  result, _, derived_path = merge_world_groups(
    intrinsic,
    calibrations,
    worlds,
    output_path,
    calibration_path,
    master="cam0",
    group_names=["pair01", "pair23"]
  )

  assert list(result["cameras"]) == ["cam0", "cam1", "cam2", "cam3"]
  assert [group["name"] for group in result["groups"]] == [
    "pair01", "pair23"
  ]
  assert derived_path == calibration_path.resolve()
  derived = json.loads(calibration_path.read_text(encoding="utf-8"))
  poses = camera_pose_matrices(derived)
  world_to_cam0 = transform_from_json(
    world01["cameras"]["cam0"]["world_to_camera"]
  )
  for world, names in ((world01, ("cam0", "cam1")),
                       (world23, ("cam2", "cam3"))):
    for name in names:
      expected = transform_from_json(
        world["cameras"][name]["world_to_camera"]
      ) @ np.linalg.inv(world_to_cam0)
      assert np.allclose(poses[name], expected)


def test_merge_world_groups_rejects_duplicate_camera(tmp_path):
  intrinsic, calibrations, worlds, _, world23 = _inputs(tmp_path)
  duplicate_calibration = json.loads(
    Path(calibrations[1]).read_text(encoding="utf-8")
  )
  duplicate_calibration["cameras"]["cam0"] = _camera(0)
  duplicate_calibration["camera_poses"] = {
    "cam0": transform_to_json(np.eye(4)),
    "cam2_to_cam0": transform_to_json(np.eye(4)),
    "cam3_to_cam0": transform_to_json(_translation(-1.1, 0.1, 0.0))
  }
  world23["cameras"]["cam0"] = {
    "world_to_camera": transform_to_json(_translation(20.0, -2.0, 8.5))
  }
  calibrations[1] = _write_json(
    tmp_path / "duplicate.json", duplicate_calibration
  )
  worlds[1] = _write_json(tmp_path / "duplicate_world.json", world23)

  with pytest.raises(ValueError, match="multiple groups"):
    merge_world_groups(
      intrinsic, calibrations, worlds, tmp_path / "output.json"
    )


def test_merge_world_groups_rejects_changed_local_baseline(tmp_path):
  intrinsic, calibrations, worlds, _, world23 = _inputs(tmp_path)
  transform = transform_from_json(
    world23["cameras"]["cam3"]["world_to_camera"]
  )
  transform[0, 3] += 0.01
  world23["cameras"]["cam3"]["world_to_camera"] = transform_to_json(
    transform
  )
  worlds[1] = _write_json(tmp_path / "changed_world.json", world23)

  with pytest.raises(ValueError, match="relative translation changed"):
    merge_world_groups(
      intrinsic, calibrations, worlds, tmp_path / "output.json"
    )
