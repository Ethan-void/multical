import json
import importlib.util
from pathlib import Path

import numpy as np

from multical.io.calibration_utils import transform_to_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_worldgroups.py"
SPEC = importlib.util.spec_from_file_location("analyze_worldgroups", SCRIPT)
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)
analyze_worldgroups = analyzer.analyze_worldgroups


def _write(path, value):
  path.write_text(json.dumps(value), encoding="utf-8")
  return str(path)


def _camera_record(x):
  world_to_camera = np.eye(4)
  world_to_camera[0, 3] = -x
  camera_to_world = np.linalg.inv(world_to_camera)
  return {
    "world_to_camera": transform_to_json(world_to_camera),
    "camera_to_world": transform_to_json(camera_to_world),
    "position_world": camera_to_world[:3, 3].tolist()
  }


def _marker_checks(group_offset):
  checks = []
  for index in range(4):
    checks.append({
      "capture": "point{}".format(index),
      "camera": "cam{}".format(group_offset),
      "points": [{
        "world_point": [float(index), 0.0, 1.0],
        "reprojection_error_px": 0.5 + group_offset * 0.1,
        "inlier": True
      }]
    })
  return checks


def _inputs(tmp_path):
  groups = []
  cameras = {}
  for group_index, names in enumerate((('cam0', 'cam1'), ('cam2', 'cam3'))):
    calibration = _write(
      tmp_path / "calibration{}.json".format(group_index),
      {
        "quality": {
          "RMS": 0.2,
          "RMS_all": 0.7,
          "observation_count": 100,
          "inlier_observation_count": 90
        }
      }
    )
    groups.append({
      "name": "group{}".format(group_index),
      "calibration": calibration,
      "world_extrinsics": str(tmp_path / "world{}.json".format(group_index)),
      "cameras": list(names),
      "joint": {
        "observation_count": 8,
        "inlier_count": 8,
        "reprojection_rms_px": 0.8,
        "all_points_rms_px": 0.9,
        "all_points_max_px": 1.2,
        "optimization_success": True,
        "marker_checks": _marker_checks(group_index * 2)
      },
      "local_consistency": {
        "max_rotation_error_deg": 0.0,
        "max_translation_error": 0.0
      }
    })
    for name in names:
      cameras[name] = _camera_record(float(name[-1]))
  world = _write(tmp_path / "merged.json", {
    "method": "grouped_world_anchor",
    "world_units": "meters",
    "groups": groups,
    "cameras": cameras
  })
  evaluation = _write(tmp_path / "evaluation.json", {
    "points": [{
      "frame": "P01",
      "cameras_used": ["cam0", "cam2"],
      "error_3d": 0.02,
      "reprojection_rms_px": 0.4,
      "error_xyz": [0.01, 0.01, 0.01]
    }],
    "failed_reconstructions": []
  })
  return world, evaluation


def test_analyze_worldgroups_reports_cross_group_quality(tmp_path):
  world, evaluation = _inputs(tmp_path)
  report = analyze_worldgroups(
    world,
    evaluation,
    min_cross_group_3d_points=1
  )

  assert report["summary"]["camera_pair_count"] == 6
  assert report["summary"]["cross_group_pair_count"] == 4
  assert report["summary"]["shared_control_point_count"] == 4
  assert report["cross_group_3d"]["summary"]["mean"] == 0.02
  assert report["acceptance"]["passed"] is True


def test_analyze_worldgroups_requires_independent_3d_evidence(tmp_path):
  world, _ = _inputs(tmp_path)
  report = analyze_worldgroups(world)

  assert report["acceptance"]["passed"] is False
  assert any(
    "未提供独立跨组" in failure
    for failure in report["acceptance"]["failures"]
  )
