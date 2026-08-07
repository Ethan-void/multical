import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml

from multical.app.world import (
  diagonal_center,
  discover_ordered_capture_images,
  extract_marker_center_correspondences,
  resolve_capture_image,
  world_extrinsics
)
from multical.app.worldmulti import (
  _marker_observations,
  _marker_quality_options,
  marker_geometry_quality,
  multicamera_world_extrinsics
)


def _paste_marker(image, dictionary, marker_id, center, size=44):
  marker = cv2.aruco.drawMarker(dictionary, marker_id, size)
  marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
  x = int(round(center[0] - size / 2))
  y = int(round(center[1] - size / 2))
  image[y:y + size, x:x + size] = marker


def test_marker_geometry_quality_rejects_oblique_and_tiny_markers():
  intrinsic = np.array([
    [700.0, 0.0, 320.0],
    [0.0, 700.0, 240.0],
    [0.0, 0.0, 1.0]
  ])
  camera = {
    "K": intrinsic.tolist(),
    "dist": [[0.0, 0.0, 0.0, 0.0, 0.0]]
  }
  square = np.asarray([
    [-0.5, 0.5, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, -0.5, 0.0],
    [-0.5, -0.5, 0.0]
  ], dtype=np.float64)
  options = _marker_quality_options({})

  frontal, _ = cv2.projectPoints(
    square, np.zeros(3), np.array([0.0, 0.0, 5.0]),
    intrinsic, np.zeros(5)
  )
  frontal_quality = marker_geometry_quality(frontal, camera, options)
  assert frontal_quality["status"] == "accepted"
  assert frontal_quality["view_angle_deg"] < 0.1

  oblique, _ = cv2.projectPoints(
    square, np.array([0.0, np.radians(65.0), 0.0]),
    np.array([0.0, 0.0, 5.0]), intrinsic, np.zeros(5)
  )
  oblique_quality = marker_geometry_quality(oblique, camera, options)
  assert oblique_quality["status"] == "rejected"
  assert "max_view_angle_deg" in oblique_quality["failed_checks"]
  assert np.isclose(oblique_quality["view_angle_deg"], 65.0, atol=0.2)

  tiny, _ = cv2.projectPoints(
    square, np.zeros(3), np.array([0.0, 0.0, 50.0]),
    intrinsic, np.zeros(5)
  )
  tiny_quality = marker_geometry_quality(tiny, camera, options)
  assert tiny_quality["status"] == "rejected"
  assert "min_edge_px" in tiny_quality["failed_checks"]


def test_marker_quality_rejects_only_bad_occurrence(monkeypatch, tmp_path):
  camera = {
    "K": [
      [700.0, 0.0, 320.0],
      [0.0, 700.0, 240.0],
      [0.0, 0.0, 1.0]
    ],
    "dist": [[0.0, 0.0, 0.0, 0.0, 0.0]]
  }
  cameras = {"C1": camera, "C2": camera}
  images = {}
  for camera_name in cameras:
    image_path = tmp_path / "{}.png".format(camera_name)
    image_path.write_bytes(b"detector is mocked")
    images[camera_name] = image_path.name
  good = {
    "center": [120.0, 100.0],
    "corners": np.array([
      [100.0, 80.0], [140.0, 80.0],
      [140.0, 120.0], [100.0, 120.0]
    ])
  }
  bad = {
    "center": [120.0, 200.5],
    "corners": np.array([
      [100.0, 200.0], [140.0, 200.0],
      [140.0, 201.0], [100.0, 201.0]
    ])
  }

  def fake_detect(*args, **kwargs):
    return None, {23: [good, bad]}, "DICT_6X6_250"

  monkeypatch.setattr(
    "multical.app.worldmulti.detect_marker_centers", fake_detect
  )
  data = {
    "cameras": ["C1", "C2"],
    "captures": [{
      "name": "P01",
      "images": images,
      "markers": [
        {
          "marker_id": 23,
          "occurrence": "upper",
          "world_point": [0.0, 0.0, 1.7]
        },
        {
          "marker_id": 23,
          "occurrence": "lower",
          "world_point": [0.0, 0.0, 0.5]
        }
      ]
    }]
  }
  observations, skipped = _marker_observations(
    data, tmp_path, cameras, _marker_quality_options(data)
  )

  assert [(item["camera"], item["occurrence"])
          for item in observations] == [
    ("C1", "upper"), ("C2", "upper")
  ]
  rejected = [
    item for item in skipped
    if item["reason"] == "marker_quality_rejected"
  ]
  assert [(item["camera"], item["occurrence"])
          for item in rejected] == [
    ("C1", "lower"), ("C2", "lower")
  ]


def test_diagonal_center_handles_projective_quadrilateral():
  corners = np.array([
    [10.0, 10.0],
    [110.0, 20.0],
    [90.0, 100.0],
    [20.0, 80.0]
  ])
  center = diagonal_center(corners)

  line_02 = corners[0] + 0.5 * (corners[2] - corners[0])
  # The diagonal midpoint is not generally the projective center; verify the
  # result lies on both diagonal lines instead.
  cross_02 = np.cross(corners[2] - corners[0], center - corners[0])
  cross_13 = np.cross(corners[3] - corners[1], center - corners[1])
  assert abs(cross_02) < 1e-10
  assert abs(cross_13) < 1e-10
  assert not np.allclose(center, line_02)


def test_resolve_capture_image_supports_common_layouts():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)

    camera_layout = root / "by_camera"
    (camera_layout / "CamA").mkdir(parents=True)
    expected_camera = camera_layout / "CamA" / "position_001.jpg"
    expected_camera.write_bytes(b"test")
    assert resolve_capture_image(
      {
        "camera": "CamA",
        "image_path": "by_camera"
      },
      {"name": "position_001"},
      root
    ) == expected_camera

    capture_layout = root / "by_capture"
    (capture_layout / "position_002").mkdir(parents=True)
    expected_capture = capture_layout / "position_002" / "CamA.png"
    expected_capture.write_bytes(b"test")
    assert resolve_capture_image(
      {
        "camera": "CamA",
        "image_path": "by_capture"
      },
      {"name": "position_002"},
      root
    ) == expected_capture

    custom_layout = root / "custom"
    (custom_layout / "CamA").mkdir(parents=True)
    expected_pattern = custom_layout / "CamA" / "shot-003.jpeg"
    expected_pattern.write_bytes(b"test")
    assert resolve_capture_image(
      {
        "camera": "CamA",
        "image_path": "custom",
        "image_pattern": "{camera}/shot-{capture}.jpeg"
      },
      {"name": "003"},
      root
    ) == expected_pattern


def test_discover_ordered_capture_images_uses_natural_order():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    image_root = root / "images" / "CamA"
    image_root.mkdir(parents=True)
    for filename in ("position_10.jpg", "position_2.jpg", "position_1.jpg"):
      (image_root / filename).write_bytes(b"test")

    discovered = discover_ordered_capture_images(
      {"camera": "CamA", "image_path": "images"},
      root
    )
    assert [path.name for path in discovered] == [
      "position_1.jpg",
      "position_2.jpg",
      "position_10.jpg"
    ]


def test_extract_marker_centers_supports_mixed_families():
  image = np.full((480, 640, 3), 255, dtype=np.uint8)
  upper_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
  )
  lower_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_1000
  )
  _paste_marker(image, upper_dictionary, 23, (200, 180), size=100)
  _paste_marker(image, lower_dictionary, 300, (440, 300), size=100)

  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    image_path = root / "position_001.png"
    assert cv2.imwrite(str(image_path), image)
    data = {
      "camera": "CamA",
      "image_path": ".",
      "captures": [{
        "markers": {
          23: {
            "marker_family": "6X6_250",
            "world_point": [0.0, 0.0, 1.7]
          },
          300: {
            "marker_family": "4X4_1000",
            "world_point": [0.0, 0.0, 0.5]
          }
        }
      }]
    }

    world_points, image_points, observations = (
      extract_marker_center_correspondences(data, root)
    )

    assert np.allclose(world_points, [
      [0.0, 0.0, 1.7],
      [0.0, 0.0, 0.5]
    ])
    assert np.allclose(image_points, [
      [200.0, 180.0],
      [440.0, 300.0]
    ], atol=0.5)
    assert [
      (item["marker_id"], item["marker_family"])
      for item in observations
    ] == [
      (23, "DICT_6X6_250"),
      (300, "DICT_4X4_1000")
    ]


def test_extract_marker_centers_supports_duplicate_id_upper_lower():
  image = np.full((480, 640, 3), 255, dtype=np.uint8)
  dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
  )
  _paste_marker(image, dictionary, 23, (320, 140), size=100)
  _paste_marker(image, dictionary, 23, (320, 340), size=100)

  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    image_path = root / "position_001.png"
    assert cv2.imwrite(str(image_path), image)
    data = {
      "camera": "CamA",
      "image_path": ".",
      "captures": [{
        "markers": [
          {
            "marker_id": 23,
            "marker_family": "6X6_250",
            "occurrence": "upper",
            "world_point": [0.0, 0.0, 1.7]
          },
          {
            "marker_id": 23,
            "marker_family": "6X6_250",
            "occurrence": "lower",
            "world_point": [0.0, 0.0, 0.5]
          }
        ]
      }]
    }

    world_points, image_points, observations = (
      extract_marker_center_correspondences(data, root)
    )

    assert np.allclose(world_points, [
      [0.0, 0.0, 1.7],
      [0.0, 0.0, 0.5]
    ])
    assert np.allclose(image_points, [
      [320.0, 140.0],
      [320.0, 340.0]
    ], atol=0.5)
    assert [
      (item["marker_id"], item["occurrence"])
      for item in observations
    ] == [(23, "upper"), (23, "lower")]


def test_world_extrinsics_from_multiple_marker_center_images():
  width, height = 640, 480
  intrinsic = np.array([
    [700.0, 0.0, 320.0],
    [0.0, 700.0, 240.0],
    [0.0, 0.0, 1.0]
  ])
  camera = {
    "model": "standard",
    "image_size": [width, height],
    "K": intrinsic.tolist(),
    "dist": [[0.0, 0.0, 0.0, 0.0, 0.0]]
  }
  calibration = {
    "cameras": {"CamA": camera},
    "camera_poses": {
      "CamA": {
        "R": np.eye(3).tolist(),
        "T": [0.0, 0.0, 0.0]
      }
    }
  }
  rotation_vector = np.array([1.0, 0.05, -0.02])
  rotation, _ = cv2.Rodrigues(rotation_vector)
  translation = np.array([0.1, -0.2, 7.0])
  positions = [
    (-1.0, -0.5),
    (1.0, -0.5),
    (-1.0, 0.5),
    (1.0, 0.5)
  ]
  dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
  )

  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
      json.dumps(calibration), encoding="utf-8"
    )
    captures = []
    for capture_index, (x, y) in enumerate(positions):
      upper = [x, y, 1.2]
      lower = [x, y, 0.2]
      world_points = np.asarray([upper, lower], dtype=np.float64)
      image_points, _ = cv2.projectPoints(
        world_points,
        rotation_vector,
        translation,
        intrinsic,
        np.zeros(5)
      )
      image = np.full((height, width, 3), 255, dtype=np.uint8)
      _paste_marker(
        image, dictionary, 12, image_points.reshape(-1, 2)[0]
      )
      _paste_marker(
        image, dictionary, 18, image_points.reshape(-1, 2)[1]
      )
      image_name = "position_{:03d}.png".format(capture_index)
      assert cv2.imwrite(str(root / image_name), image)
      captures.append({
        "markers": {
          12: upper,
          18: lower
        }
      })

    correspondence_path = root / "world_markers.yaml"
    correspondence_path.write_text(
      yaml.safe_dump({
        "camera": "CamA",
        "world_units": "meters",
        "marker_family": "6X6_250",
        "image_path": ".",
        "captures": captures
      }, sort_keys=False),
      encoding="utf-8"
    )

    result, output_path = world_extrinsics(
      calibration_path,
      correspondence_path,
      ransac_threshold=2.0
    )

    estimated = np.asarray(
      result["cameras"]["CamA"]["world_to_camera"]["matrix"]
    )
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = translation
    assert output_path.is_file()
    assert result["anchor"]["input_mode"] == "marker_centers"
    assert result["anchor"]["point_count"] == 8
    assert result["anchor"]["inlier_count"] == 8
    assert result["anchor"]["reprojection_rms_px"] < 1.0
    assert np.allclose(estimated[:3, :3], expected[:3, :3], atol=0.02)
    assert np.allclose(estimated[:3, 3], expected[:3, 3], atol=0.08)
    checks = result["anchor"]["marker_checks"]
    assert len(checks) == len(positions)
    for check in checks:
      assert (root / check["validation_image"]).is_file()


def test_multicamera_world_extrinsics_from_marker_centers():
  width, height = 640, 480
  intrinsic = np.array([
    [700.0, 0.0, 320.0],
    [0.0, 700.0, 240.0],
    [0.0, 0.0, 1.0]
  ])
  camera = {
    "model": "standard",
    "image_size": [width, height],
    "K": intrinsic.tolist(),
    "dist": [[0.0, 0.0, 0.0, 0.0, 0.0]]
  }
  calibration = {
    "cameras": {"C1": camera, "C2": camera},
    "camera_poses": {
      "C1": {"R": np.eye(3).tolist(), "T": [0.0, 0.0, 0.0]},
      "C2_to_C1": {
        "R": np.eye(3).tolist(), "T": [-0.4, 0.0, 0.0]
      }
    }
  }
  rotation_vector = np.array([1.0, 0.05, -0.02])
  rotation, _ = cv2.Rodrigues(rotation_vector)
  world_to_rig = np.eye(4)
  world_to_rig[:3, :3] = rotation
  world_to_rig[:3, 3] = [0.1, -0.2, 7.0]
  rig_poses = {
    "C1": np.eye(4),
    "C2": np.array([
      [1.0, 0.0, 0.0, -0.4],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  }
  dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
  )
  positions = [
    (-1.0, -0.5), (1.0, -0.5), (-1.0, 0.5), (1.0, 0.5)
  ]

  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
      json.dumps(calibration), encoding="utf-8"
    )
    captures = []
    for capture_index, (x, y) in enumerate(positions):
      points = np.asarray([
        [x, y, 1.2], [x, y, 0.2]
      ], dtype=np.float64)
      images = {}
      for camera_name in ("C1", "C2"):
        transform = rig_poses[camera_name] @ world_to_rig
        rvec, _ = cv2.Rodrigues(transform[:3, :3])
        pixels, _ = cv2.projectPoints(
          points, rvec, transform[:3, 3], intrinsic, np.zeros(5)
        )
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        _paste_marker(
          image, dictionary, 23, pixels.reshape(-1, 2)[0]
        )
        _paste_marker(
          image, dictionary, 23, pixels.reshape(-1, 2)[1]
        )
        image_name = "{}_{:03d}.png".format(
          camera_name, capture_index
        )
        assert cv2.imwrite(str(root / image_name), image)
        images[camera_name] = image_name
      captures.append({
        "name": "position_{:03d}".format(capture_index),
        "images": images,
        "markers": [
          {
            "marker_id": 23, "occurrence": "upper",
            "world_point": points[0].tolist()
          },
          {
            "marker_id": 23, "occurrence": "lower",
            "world_point": points[1].tolist()
          }
        ]
      })
    correspondence_path = root / "world_multicam.yaml"
    correspondence_path.write_text(
      yaml.safe_dump({
        "world_units": "meters",
        "cameras": ["C1", "C2"],
        "marker_family": "6X6_250",
        "captures": captures
      }, sort_keys=False),
      encoding="utf-8"
    )

    result, output_path = multicamera_world_extrinsics(
      calibration_path, correspondence_path,
      ransac_threshold=3.0
    )

    assert output_path.is_file()
    assert result["joint"]["input_mode"] == "marker_centers"
    assert result["joint"]["participating_cameras"] == ["C1", "C2"]
    assert result["joint"]["observation_count"] == 16
    assert result["joint"]["reprojection_rms_px"] < 1.0
    estimated = np.asarray(
      result["rig"]["world_to_rig"]["matrix"]
    )
    assert np.allclose(
      estimated[:3, :3], world_to_rig[:3, :3], atol=0.02
    )
    assert np.allclose(
      estimated[:3, 3], world_to_rig[:3, 3], atol=0.08
    )
