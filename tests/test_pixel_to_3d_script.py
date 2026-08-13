import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from multical.io.calibration_utils import transform_to_json


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pixel_to_3d.py"


def _write_camera_files(root):
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
  calibration_path = root / "calibration.json"
  calibration_path.write_text(json.dumps({
    "cameras": {"C1": camera, "C2": camera}
  }), encoding="utf-8")

  first = np.eye(4)
  second = np.eye(4)
  second[0, 3] = -0.5
  world_path = root / "world.json"
  world_path.write_text(json.dumps({
    "world_units": "meters",
    "intrinsic": str(calibration_path),
    "cameras": {
      "C1": {"world_to_camera": transform_to_json(first)},
      "C2": {"world_to_camera": transform_to_json(second)}
    }
  }), encoding="utf-8")
  return calibration_path, world_path


def _pixel(point, translation_x):
  camera_point = np.asarray(point, dtype=float) + [translation_x, 0.0, 0.0]
  return [
    700.0 * camera_point[0] / camera_point[2] + 320.0,
    700.0 * camera_point[1] / camera_point[2] + 240.0
  ]


def test_cli_pixels_output_world_coordinate(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = np.array([0.2, -0.1, 3.0])
  first = _pixel(expected, 0.0)
  second = _pixel(expected, -0.5)

  completed = subprocess.run([
    sys.executable,
    str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--pixel", "C1", str(first[0]), str(first[1]),
    "--pixel", "C2", str(second[0]), str(second[1])
  ], check=False, capture_output=True, text=True)

  assert completed.returncode == 0, completed.stderr
  result = json.loads(completed.stdout)
  assert result["status"] == "ok"
  assert result["coordinate_frame"] == "world"
  assert result["world_units"] == "meters"
  assert np.allclose(result["point_world"], expected, atol=1e-9)


def test_stdin_input_and_xyz_output(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = np.array([-0.15, 0.25, 2.5])
  observations = {
    "observations": {
      "C1": _pixel(expected, 0.0),
      "C2": _pixel(expected, -0.5)
    }
  }

  completed = subprocess.run([
    sys.executable,
    str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--input", "-",
    "--format", "xyz"
  ], input=json.dumps(observations), check=False, capture_output=True, text=True)

  assert completed.returncode == 0, completed.stderr
  assert np.allclose(
    [float(value) for value in completed.stdout.split()],
    expected,
    atol=1e-9
  )


def test_single_camera_is_rejected_for_3d(tmp_path):
  _, world_path = _write_camera_files(tmp_path)

  completed = subprocess.run([
    sys.executable,
    str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--pixel", "C1", "320", "240"
  ], check=False, capture_output=True, text=True)

  assert completed.returncode == 2
  error = json.loads(completed.stderr)
  assert error["status"] == "error"
  assert "at least two cameras" in error["reason"]


def test_batch_txt_preserves_ids_and_writes_3d_rows(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = {
    "P01": np.array([0.2, -0.1, 3.0]),
    "P02": np.array([-0.15, 0.25, 2.5])
  }
  input_path = tmp_path / "pixels.txt"
  rows = ["# point_id camera u v camera u v"]
  for point_id, point in expected.items():
    first = _pixel(point, 0.0)
    second = _pixel(point, -0.5)
    rows.append(
      "{} C1 {} {} C2 {} {}".format(
        point_id, first[0], first[1], second[0], second[1]
      )
    )
  input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
  output_path = tmp_path / "points_3d.txt"

  completed = subprocess.run([
    sys.executable,
    str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--input-txt", str(input_path),
    "--output-txt", str(output_path)
  ], check=False, capture_output=True, text=True)

  assert completed.returncode == 0, completed.stderr
  summary = json.loads(completed.stdout)
  assert summary["reconstructed_count"] == 2
  lines = output_path.read_text(encoding="utf-8").splitlines()
  assert lines[0].startswith("point_id\tX\tY\tZ\tstatus")
  for line in lines[1:]:
    fields = line.split("\t")
    assert fields[4] == "ok"
    assert np.allclose(
      [float(value) for value in fields[1:4]],
      expected[fields[0]],
      atol=1e-9
    )


def test_separate_camera_txt_files_match_rows(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = [
    np.array([0.2, -0.1, 3.0]),
    np.array([-0.15, 0.25, 2.5])
  ]
  first_path = tmp_path / "C1.txt"
  second_path = tmp_path / "C2.txt"
  first_path.write_text(
    "# U V\n" + "\n".join(
      "{} {}".format(*_pixel(point, 0.0)) for point in expected
    ) + "\n",
    encoding="utf-8"
  )
  second_path.write_text(
    "\n".join(
      "{},{}".format(*_pixel(point, -0.5)) for point in expected
    ) + "\n",
    encoding="utf-8"
  )
  output_path = tmp_path / "points_3d.txt"

  completed = subprocess.run([
    sys.executable,
    str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--camera-txt", "C1", str(first_path),
    "--camera-txt", "C2", str(second_path),
    "--output-txt", str(output_path)
  ], check=False, capture_output=True, text=True)

  assert completed.returncode == 0, completed.stderr
  lines = output_path.read_text(encoding="utf-8").splitlines()[1:]
  assert len(lines) == 2
  for index, line in enumerate(lines):
    fields = line.split("\t")
    assert fields[0] == "P{:06d}".format(index + 1)
    assert fields[4] == "ok"
    assert np.allclose(
      [float(value) for value in fields[1:4]], expected[index], atol=1e-9
    )


def test_separate_camera_txt_rejects_mismatched_rows(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  first_path = tmp_path / "C1.txt"
  second_path = tmp_path / "C2.txt"
  first_path.write_text("320 240\n321 241\n", encoding="utf-8")
  second_path.write_text("300 240\n", encoding="utf-8")

  completed = subprocess.run([
    sys.executable,
    str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--camera-txt", "C1", str(first_path),
    "--camera-txt", "C2", str(second_path),
    "--output-txt", str(tmp_path / "points_3d.txt")
  ], check=False, capture_output=True, text=True)

  assert completed.returncode == 2
  assert "same number of valid rows" in json.loads(completed.stderr)["reason"]


def test_no_arguments_uses_top_level_camera_txt_configuration(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = np.array([0.2, -0.1, 3.0])
  first = _pixel(expected, 0.0)
  second = _pixel(expected, -0.5)
  (tmp_path / "cam0.txt").write_text(
    "{} {}\n".format(*first), encoding="utf-8"
  )
  (tmp_path / "cam1.txt").write_text(
    "{} {}\n".format(*second), encoding="utf-8"
  )
  configured_script = tmp_path / "pixel_to_3d.py"
  script_text = SCRIPT.read_text(encoding="utf-8")
  start = script_text.index("DEFAULT_WORLD_EXTRINSICS = (")
  end = script_text.index("DEFAULT_REPROJECTION_THRESHOLD", start)
  configured = (
    script_text[:start]
    + "DEFAULT_WORLD_EXTRINSICS = {!r}\n".format(str(world_path))
    + "DEFAULT_CALIBRATION = None\n"
    + "DEFAULT_CAMERA_TXT_FILES = {\n"
    + "  'C1': 'cam0.txt',\n  'C2': 'cam1.txt'\n}\n"
    + "DEFAULT_TRAJECTORY_DIR = None\n"
    + "DEFAULT_DATASET_DIR = None\n"
    + "DEFAULT_CAMERA_NAME_MAP = {}\n"
    + "DEFAULT_OUTPUT_TXT = 'points_3d.txt'\n"
    + "DEFAULT_OUTPUT_DIR = None\n"
    + script_text[end:]
  )
  configured_script.write_text(configured, encoding="utf-8")

  completed = subprocess.run([
    sys.executable, str(configured_script)
  ], cwd=tmp_path, check=False, capture_output=True, text=True)

  assert completed.returncode == 0, completed.stderr
  output_path = tmp_path / "points_3d.txt"
  fields = output_path.read_text(encoding="utf-8").splitlines()[1].split("\t")
  assert np.allclose(
    [float(value) for value in fields[1:4]], expected, atol=1e-9
  )


def test_detector_csv_inputs_use_frame_and_xy_center(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = [np.array([0.2, -0.1, 3.0]), np.array([-0.15, 0.25, 2.5])]
  paths = []
  for camera, offset in (("C1", 0.0), ("C2", -0.5)):
    path = tmp_path / "{}.csv".format(camera)
    rows = ["Frame,Visibility,X,Y,W,H"]
    for index, point in enumerate(expected):
      u, v = _pixel(point, offset)
      rows.append("{},1,{},{},6,6".format(index, u, v))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    paths.extend(["--camera-csv", camera, str(path)])
  output = tmp_path / "csv_3d.txt"
  completed = subprocess.run([
    sys.executable, str(SCRIPT), "--world-extrinsics", str(world_path),
    *paths, "--output-txt", str(output)
  ], check=False, capture_output=True, text=True)
  assert completed.returncode == 0, completed.stderr
  for index, line in enumerate(output.read_text().splitlines()[1:]):
    fields = line.split("\t")
    assert fields[0] == str(index)
    assert np.allclose([float(x) for x in fields[1:4]], expected[index], atol=1e-9)


def test_labelme_json_dirs_use_rectangle_center_and_keep_missing_frame(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = np.array([0.2, -0.1, 3.0])
  directories = []
  for camera, offset in (("C1", 0.0), ("C2", -0.5)):
    directory = tmp_path / camera
    directory.mkdir()
    u, v = _pixel(expected, offset)
    (directory / "0.json").write_text(json.dumps({
      "shapes": [{
        "label": "ball", "shape_type": "rectangle",
        "points": [[u - 3, v - 3], [u + 3, v + 3]]
      }]
    }), encoding="utf-8")
    if camera == "C1":
      (directory / "1.json").write_text(json.dumps({
        "shapes": [{
          "label": "ball", "shape_type": "rectangle",
          "points": [[u - 2, v - 2], [u + 2, v + 2]]
        }]
      }), encoding="utf-8")
    directories.extend(["--camera-json-dir", camera, str(directory)])
  output = tmp_path / "json_3d.txt"
  completed = subprocess.run([
    sys.executable, str(SCRIPT), "--world-extrinsics", str(world_path),
    *directories, "--output-txt", str(output)
  ], check=False, capture_output=True, text=True)
  assert completed.returncode == 1, completed.stderr
  lines = output.read_text().splitlines()[1:]
  assert lines[0].split("\t")[4] == "ok"
  assert lines[1].split("\t")[1:5] == ["nan", "nan", "nan", "failed"]


def test_trajectory_directory_discovers_csvs_and_skips_empty_xy(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  trajectory = tmp_path / "traj_0001"
  expected = np.array([0.2, -0.1, 3.0])
  for source_camera, offset in (("cam0", 0.0), ("cam1", -0.5)):
    directory = trajectory / source_camera
    directory.mkdir(parents=True)
    u, v = _pixel(expected, offset)
    (directory / "ball.csv").write_text(
      "Frame,Visibility,X,Y,W,H\n"
      "0,1,,,,\n"
      "1,1,{},{},6,6\n".format(u, v),
      encoding="utf-8"
    )
  output = tmp_path / "trajectory_3d.txt"
  completed = subprocess.run([
    sys.executable, str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--trajectory-dir", str(trajectory),
    "--camera-map", "cam0", "C1",
    "--camera-map", "cam1", "C2",
    "--output-txt", str(output)
  ], check=False, capture_output=True, text=True)
  assert completed.returncode == 1, completed.stderr
  lines = output.read_text(encoding="utf-8").splitlines()[1:]
  assert lines[0].split("\t")[1:5] == ["nan", "nan", "nan", "failed"]
  fields = lines[1].split("\t")
  assert fields[0] == "1"
  assert fields[4] == "ok"
  assert np.allclose([float(x) for x in fields[1:4]], expected, atol=1e-9)


def test_dataset_directory_processes_all_trajectories(tmp_path):
  _, world_path = _write_camera_files(tmp_path)
  expected = np.array([0.2, -0.1, 3.0])
  dataset = tmp_path / "dataset"
  for trajectory_name in ("traj_0001", "traj_0002"):
    for source_camera, offset in (("cam0", 0.0), ("cam1", -0.5)):
      directory = dataset / trajectory_name / source_camera
      directory.mkdir(parents=True)
      u, v = _pixel(expected, offset)
      (directory / "ball.csv").write_text(
        "Frame,Visibility,X,Y,W,H\n0,1,{},{},6,6\n".format(u, v),
        encoding="utf-8"
      )
  output_root = tmp_path / "results"
  completed = subprocess.run([
    sys.executable, str(SCRIPT),
    "--world-extrinsics", str(world_path),
    "--dataset-dir", str(dataset),
    "--camera-map", "cam0", "C1",
    "--camera-map", "cam1", "C2",
    "--output-dir", str(output_root)
  ], check=False, capture_output=True, text=True)
  assert completed.returncode == 0, completed.stderr
  summary = json.loads(completed.stdout)
  assert summary["trajectory_count"] == 2
  assert summary["reconstructed_count"] == 2
  for trajectory_name in ("traj_0001", "traj_0002"):
    output = output_root / trajectory_name / "points_3d.txt"
    fields = output.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert fields[4] == "ok"
    assert np.allclose([float(x) for x in fields[1:4]], expected, atol=1e-9)
