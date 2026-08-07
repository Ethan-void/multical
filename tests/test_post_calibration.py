import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from multical.app.rectify import generate_rectification
from multical.app.triangulate import (
  build_camera_models,
  triangulate_frame,
  triangulate_observations
)
from multical.app.world import world_extrinsics
from multical.app.worldmulti import multicamera_world_extrinsics
from multical.io.calibration_utils import (
  transform_from_rt,
  transform_to_json
)
from multical.io.export_calib import (
  _pair_pose_statistics,
  _pair_quality_status,
  _tree_transform,
  export_calibration_quality
)


def synthetic_calibration():
  intrinsic = [
    [700.0, 0.0, 320.0],
    [0.0, 700.0, 240.0],
    [0.0, 0.0, 1.0]
  ]
  camera = {
    "model": "standard",
    "image_size": [640, 480],
    "K": intrinsic,
    "dist": [[0.0, 0.0, 0.0, 0.0, 0.0]]
  }
  return {
    "quality": {
      "RMS": 0.31,
      "RMS_all": 0.48,
      "unit": "px",
      "inlier_observation_count": 1200,
      "observation_count": 1240,
      "outlier_filter_applied": True
    },
    "cameras": {
      "C1": camera,
      "C2": camera,
      "C3": camera,
      "C4": camera
    },
    "camera_poses": {
      "C1": {
        "R": np.eye(3).tolist(),
        "T": [0.0, 0.0, 0.0]
      },
      "C2_to_C1": {
        "R": np.eye(3).tolist(),
        "T": [-0.25, 0.0, 0.0]
      },
      "C3_to_C1": {
        "R": np.eye(3).tolist(),
        "T": [0.0, -0.30, 0.0]
      },
      "C4_to_C1": {
        "R": np.eye(3).tolist(),
        "T": [0.20, 0.0, 0.0]
      }
    }
  }


def test_calibration_quality_exports_final_rms():
  class CalibrationResult:
    reprojection_error = np.array([1.0, 2.0, 10.0])
    reprojection_inliers = np.array([1.0, 2.0])
    inlier_mask = np.array([True, True, False])

  quality = export_calibration_quality(CalibrationResult())

  assert abs(quality.RMS - np.sqrt(2.5)) < 1e-12
  assert abs(quality.RMS_all - np.sqrt(35.0)) < 1e-12
  assert quality.unit == "px"
  assert quality.inlier_observation_count == 2
  assert quality.observation_count == 3
  assert quality.outlier_filter_applied


def test_extrinsic_quality_status_and_tree_transform():
  assert _pair_quality_status(30, 0.1, 0.02) == "good"
  assert _pair_quality_status(20, 0.4, 0.10) == "caution"
  assert _pair_quality_status(17, 1.66, 0.656) == "weak"
  assert _pair_quality_status(0, None, None) == "no_overlap"

  first_to_second = transform_from_rt(
    np.eye(3), [1.0, 0.0, 0.0]
  )
  second_to_third = transform_from_rt(
    np.eye(3), [0.0, 2.0, 0.0]
  )
  adjacency = {
    0: [(1, first_to_second)],
    1: [
      (0, np.linalg.inv(first_to_second)),
      (2, second_to_third)
    ],
    2: [(1, np.linalg.inv(second_to_third))]
  }
  assert np.allclose(
    _tree_transform(adjacency, 0, 2),
    second_to_third @ first_to_second
  )


def test_pair_pose_statistics_reports_final_edge_residuals():
  frame_poses = np.repeat(np.eye(4)[None], 4, axis=0)
  frame_poses[:, 0, 3] = np.arange(4, dtype=np.float64)
  pose_table = type("PoseTable", (), {
    "valid": np.ones((2, 4), dtype=bool),
    "poses": np.stack([frame_poses, frame_poses])
  })()
  angle = np.deg2rad(1.0)
  rotation = np.array([
    [np.cos(angle), -np.sin(angle), 0.0],
    [np.sin(angle), np.cos(angle), 0.0],
    [0.0, 0.0, 1.0]
  ])
  final_transform = transform_from_rt(rotation, [0.1, 0.0, 0.0])

  with patch(
      "multical.io.export_calib.matrix.align_transforms_robust",
      return_value=(np.eye(4), np.ones(4, dtype=bool))):
    statistics = _pair_pose_statistics(
      pose_table, 0, 1, final_transform=final_transform
    )

  assert statistics.common_pose_count == 4
  assert statistics.pose_inlier_count == 4
  assert statistics.rotation_scatter_deg < 1e-10
  assert statistics.translation_scatter < 1e-10
  assert abs(statistics.final_edge_rotation_residual_deg - 1.0) < 1e-10
  assert abs(statistics.final_edge_translation_residual - 0.1) < 1e-10
  assert abs(statistics.final_rotation_residual_rms_deg - 1.0) < 1e-10
  assert abs(statistics.final_translation_residual_rms - 0.1) < 1e-10


def test_stereo_rectification_writes_parameters_and_maps():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
      json.dumps(synthetic_calibration()), encoding="utf-8"
    )

    result, output_path = generate_rectification(
      calibration_path, ["C1:C2"], alpha=0.0
    )

    pair = result["pairs"]["C1:C2"]
    assert output_path.is_file()
    assert result["quality"] == synthetic_calibration()["quality"]
    assert abs(pair["baseline"] - 0.25) < 1e-12
    expected_e = np.array([
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.25],
      [0.0, -0.25, 0.0]
    ])
    assert np.allclose(pair["E"], expected_e)
    expected_f = (
      np.linalg.inv(np.asarray(synthetic_calibration()["cameras"]["C2"]["K"])).T
      @ expected_e
      @ np.linalg.inv(
        np.asarray(synthetic_calibration()["cameras"]["C1"]["K"])
      )
    )
    assert np.allclose(pair["F"], expected_f)
    assert np.asarray(pair["Q"]).shape == (4, 4)
    maps_path = output_path.parent / pair["maps"]
    assert maps_path.is_file()
    with np.load(maps_path) as maps:
      assert maps["left_map_x"].shape == (480, 640)
      assert maps["right_map_y"].shape == (480, 640)
      assert np.isfinite(maps["left_map_x"]).all()


def test_world_extrinsics_from_planar_control_points():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    calibration = synthetic_calibration()
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
      json.dumps(calibration), encoding="utf-8"
    )

    world_points = np.array([
      [-1.0, -1.0, 0.0],
      [0.0, -1.0, 0.0],
      [1.0, -1.0, 0.0],
      [-1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [-1.0, 1.0, 0.0],
      [0.0, 1.0, 0.0],
      [1.0, 1.0, 0.0]
    ])
    rotation_vector = np.array([0.12, -0.08, 0.03])
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.array([0.2, -0.1, 5.0])
    intrinsic = np.asarray(calibration["cameras"]["C1"]["K"])
    image_points, _ = cv2.projectPoints(
      world_points,
      rotation_vector,
      translation,
      intrinsic,
      np.zeros(5)
    )
    correspondence_path = root / "world_points.json"
    correspondence_path.write_text(json.dumps({
      "camera": "C1",
      "world_units": "meters",
      "world_points": world_points.tolist(),
      "image_points": image_points.reshape(-1, 2).tolist()
    }), encoding="utf-8")

    result, output_path = world_extrinsics(
      calibration_path, correspondence_path
    )

    expected_c1 = transform_from_rt(rotation, translation)
    expected_c2 = transform_from_rt(
      np.eye(3), [-0.25, 0.0, 0.0]
    ) @ expected_c1
    estimated_c1 = np.asarray(
      result["cameras"]["C1"]["world_to_camera"]["matrix"]
    )
    estimated_c2 = np.asarray(
      result["cameras"]["C2"]["world_to_camera"]["matrix"]
    )
    assert output_path.is_file()
    assert result["anchor"]["inlier_count"] == len(world_points)
    assert result["anchor"]["reprojection_rms_px"] < 1e-5
    assert np.allclose(estimated_c1, expected_c1, atol=1e-6)
    assert np.allclose(estimated_c2, expected_c2, atol=1e-6)


def test_joint_multicamera_world_extrinsics():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    calibration = synthetic_calibration()
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
      json.dumps(calibration), encoding="utf-8"
    )
    world_points = np.array([
      [-1.0, -1.0, 0.0],
      [0.0, -1.0, 0.2],
      [1.0, -1.0, 0.0],
      [-1.0, 0.0, 0.4],
      [1.0, 0.0, 0.1],
      [-1.0, 1.0, 0.0],
      [0.0, 1.0, 0.3],
      [1.0, 1.0, 0.0]
    ], dtype=np.float64)
    rotation_vector = np.array([0.12, -0.08, 0.03])
    rotation, _ = cv2.Rodrigues(rotation_vector)
    world_to_rig = transform_from_rt(
      rotation, [0.2, -0.1, 5.0]
    )
    rig_poses = {
      "C1": transform_from_rt(np.eye(3), [0.0, 0.0, 0.0]),
      "C2": transform_from_rt(np.eye(3), [-0.25, 0.0, 0.0]),
      "C3": transform_from_rt(np.eye(3), [0.0, -0.30, 0.0]),
      "C4": transform_from_rt(np.eye(3), [0.20, 0.0, 0.0])
    }
    observations = []
    # Each point may be visible in only a subset. C4 deliberately has no
    # world-anchor observations and must inherit the shared rig transform.
    visible = {
      "C1": range(0, 6),
      "C2": range(2, 8),
      "C3": (0, 1, 4, 5, 6, 7)
    }
    intrinsic = np.asarray(calibration["cameras"]["C1"]["K"])
    for camera_name, indices in visible.items():
      transform = rig_poses[camera_name] @ world_to_rig
      rvec, _ = cv2.Rodrigues(transform[:3, :3])
      selected = world_points[list(indices)]
      pixels, _ = cv2.projectPoints(
        selected, rvec, transform[:3, 3], intrinsic, np.zeros(5)
      )
      for point_index, pixel in zip(indices, pixels.reshape(-1, 2)):
        observations.append({
          "camera": camera_name,
          "world_point": world_points[point_index].tolist(),
          "image_point": pixel.tolist()
        })
    correspondence_path = root / "world_multicam.yaml"
    correspondence_path.write_text(
      yaml.safe_dump({
        "world_units": "meters",
        "observations": observations
      }, sort_keys=False),
      encoding="utf-8"
    )

    result, output_path = multicamera_world_extrinsics(
      calibration_path, correspondence_path,
      ransac_threshold=1.0
    )

    assert output_path.name == "world_extrinsics_multicam.json"
    assert result["method"] == "joint_multicamera_world_anchor"
    assert result["joint"]["participating_cameras"] == ["C1", "C2", "C3"]
    assert result["joint"]["reprojection_rms_px"] < 1e-5
    assert "C4" in result["cameras"]
    for camera_name in calibration["cameras"]:
      estimated = np.asarray(
        result["cameras"][camera_name]["world_to_camera"]["matrix"]
      )
      expected = rig_poses[camera_name] @ world_to_rig
      assert np.allclose(estimated, expected, atol=1e-6)


def test_multiview_triangulation_rejects_outlier_and_reports_failures():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    calibration = synthetic_calibration()
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
      json.dumps(calibration), encoding="utf-8"
    )

    transforms = {
      "C1": transform_from_rt(np.eye(3), [0.0, 0.0, 0.0]),
      "C2": transform_from_rt(np.eye(3), [-0.25, 0.0, 0.0]),
      "C3": transform_from_rt(np.eye(3), [0.0, -0.30, 0.0]),
      "C4": transform_from_rt(np.eye(3), [0.20, 0.0, 0.0])
    }
    world_path = root / "world_extrinsics.json"
    world_path.write_text(json.dumps({
      "world_units": "meters",
      "convention": (
        "x_camera = R_world_to_camera * x_world + T_world_to_camera"
      ),
      "cameras": {
        name: {"world_to_camera": transform_to_json(transform)}
        for name, transform in transforms.items()
      }
    }), encoding="utf-8")

    intrinsic = np.asarray(calibration["cameras"]["C1"]["K"])

    def observe(point, camera_name):
      camera_point = (
        transforms[camera_name][:3, :3] @ point +
        transforms[camera_name][:3, 3]
      )
      pixel = intrinsic @ camera_point
      return (pixel[:2] / pixel[2]).tolist()

    point = np.array([0.12, -0.08, 4.5])
    observations = {
      name: observe(point, name) for name in transforms
    }
    observations["C4"] = (
      np.asarray(observations["C4"]) + [80.0, -50.0]
    ).tolist()
    far_point = np.array([0.0, 0.0, 1000.0])
    observation_path = root / "observations.json"
    observation_path.write_text(json.dumps({
      "sequence": "test-ball",
      "frames": [
        {
          "frame": "frame_0000",
          "observations": observations
        },
        {
          "frame": "frame_0001",
          "observations": {
            "C1": observe(point, "C1"),
            "C2": observe(point, "C2")
          }
        },
        {
          "frame": "frame_0002",
          "observations": {
            "C1": observe(point, "C1")
          }
        },
        {
          "frame": "frame_0003",
          "observations": {
            "C1": observe(far_point, "C1"),
            "C2": observe(far_point, "C2")
          }
        }
      ]
    }), encoding="utf-8")

    result, output_path = triangulate_observations(
      calibration_path,
      world_path,
      observation_path,
      reprojection_threshold=2.0,
      min_ray_angle_deg=1.0
    )

    assert output_path.is_file()
    assert result["summary"]["reconstructed_count"] == 2
    assert result["summary"]["failed_count"] == 2
    first = result["frames"][0]
    assert first["status"] == "ok"
    assert "C4" in first["cameras_rejected"]
    assert np.allclose(first["point_world"], point, atol=1e-8)
    assert first["refinement"]["enabled"]
    assert (
      first["refinement"]["final_reprojection_rms_px"]
      <= first["refinement"]["initial_reprojection_rms_px"] + 1e-12
    )
    assert result["settings"]["refine"]
    assert result["frames"][2]["status"] == "failed"
    assert "fewer than two" in result["frames"][2]["reason"]
    assert result["frames"][3]["status"] == "failed"
    assert "ray angle" in result["frames"][3]["reason"]


def test_nonlinear_triangulation_refinement_reduces_pixel_error():
  calibration = synthetic_calibration()
  calibration["cameras"]["C1"]["dist"] = [[
    -0.12, 0.03, 0.001, -0.0005, 0.0
  ]]
  for name in ["C2", "C3", "C4"]:
    calibration["cameras"][name] = dict(
      calibration["cameras"]["C1"]
    )

  transforms = {
    "C1": transform_from_rt(np.eye(3), [0.0, 0.0, 0.0]),
    "C2": transform_from_rt(np.eye(3), [-0.35, 0.0, 0.0]),
    "C3": transform_from_rt(np.eye(3), [0.0, -0.40, 0.0])
  }
  world_extrinsics_data = {
    "cameras": {
      name: {"world_to_camera": transform_to_json(transform)}
      for name, transform in transforms.items()
    }
  }
  models = build_camera_models(calibration, world_extrinsics_data)
  point = np.array([0.65, -0.35, 3.2])
  noise = {
    "C1": [0.8, -0.4],
    "C2": [-0.3, 0.7],
    "C3": [0.2, -0.6]
  }
  observations = {}
  for name, model in models.items():
    rotation = model["world_to_camera"][:3, :3]
    translation = model["world_to_camera"][:3, 3]
    rotation_vector, _ = cv2.Rodrigues(rotation)
    pixel, _ = cv2.projectPoints(
      point.reshape(1, 3),
      rotation_vector,
      translation,
      model["K"],
      model["dist"]
    )
    observations[name] = (
      pixel.reshape(2) + np.asarray(noise[name])
    ).tolist()

  result = triangulate_frame(
    {"frame": "noisy", "observations": observations},
    models,
    reprojection_threshold=3.0,
    refine=True
  )

  assert result["status"] == "ok"
  assert result["refinement"]["applied"]
  assert (
    result["refinement"]["final_reprojection_rms_px"]
    < result["refinement"]["initial_reprojection_rms_px"]
  )
