#!/usr/bin/env python3
"""Reconstruct one world-space 3D point from synchronized camera pixels."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from multical.app.triangulate import (
  build_camera_models,
  load_json_or_yaml,
  parse_observation,
  triangulate_frame
)
from multical.io.calibration_utils import load_calibration_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Editable configuration for direct, argument-free execution.
# Paths may be absolute or relative to the directory where the command runs.
# ---------------------------------------------------------------------------
DEFAULT_WORLD_EXTRINSICS = (
  PROJECT_ROOT / "20260811" / "world" / "world_extrinsic.json"
)
DEFAULT_CALIBRATION = None  # None: read the intrinsic path from world extrinsics
DEFAULT_CAMERA_TXT_FILES = {
  "cam0": "cam0.txt",
  "cam1": "cam1.txt",
  "cam4": "cam4.txt",
  "cam5": "cam5.txt"
}


DEFAULT_TRAJECTORY_DIR = None  # e.g. "/data/20260812/traj_0001"
DEFAULT_DATASET_DIR = "/Users/ethan/Dev/tennis-vision/20260812"
DEFAULT_CAMERA_NAME_MAP = {
  "cam2": "cam4",
  "cam3": "cam5"
}
DEFAULT_OUTPUT_TXT = "points_3d.txt"
DEFAULT_OUTPUT_DIR = "/Users/ethan/Dev/tennis-vision/20260812/3d_results"
DEFAULT_REPROJECTION_THRESHOLD = 1.5
DEFAULT_MIN_RAY_ANGLE_DEG = 8.0
DEFAULT_REFINE = True
DEFAULT_REFINE_LOSS = "soft_l1"
DEFAULT_REFINE_MAX_ITERATIONS = 50


def _existing_file(filename, label):
  path = Path(filename).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError("{} not found: {}".format(label, path))
  return path


def _calibration_path(calibration_file, world_path, world_extrinsics):
  if calibration_file:
    return _existing_file(calibration_file, "calibration")

  intrinsic = world_extrinsics.get("intrinsic")
  if not intrinsic:
    raise ValueError(
      "world extrinsics do not declare an intrinsic calibration; "
      "pass --calibration"
    )
  candidate = Path(intrinsic).expanduser()
  if candidate.is_absolute() and candidate.is_file():
    return candidate.resolve()
  relative_to_world = (world_path.parent / candidate).resolve()
  if relative_to_world.is_file():
    return relative_to_world
  return _existing_file(candidate, "calibration")


def _cli_observations(values):
  observations = {}
  for camera, raw_u, raw_v in values or []:
    if camera in observations:
      raise ValueError("duplicate pixel for camera {}".format(camera))
    try:
      pixel = [float(raw_u), float(raw_v)]
    except ValueError as error:
      raise ValueError(
        "pixel for camera {} must contain numeric U and V".format(camera)
      ) from error
    if not all(math.isfinite(value) for value in pixel):
      raise ValueError(
        "pixel for camera {} must contain finite U and V".format(camera)
      )
    observations[camera] = pixel
  return observations


def _file_observations(filename):
  if filename == "-":
    try:
      data = json.load(sys.stdin)
    except json.JSONDecodeError as error:
      raise ValueError("stdin is not valid JSON: {}".format(error)) from error
  else:
    data = load_json_or_yaml(_existing_file(filename, "pixel input"))
  if isinstance(data, dict) and "observations" in data:
    data = data["observations"]
  if not isinstance(data, dict):
    raise ValueError(
      "pixel input must be a camera-to-[u, v] mapping or contain observations"
    )
  return data


def _normalized_observations(observations):
  normalized = {}
  for camera, value in observations.items():
    point, confidence = parse_observation(value)
    if isinstance(value, dict):
      normalized[str(camera)] = {
        "point": point.tolist(),
        "confidence": confidence
      }
    else:
      normalized[str(camera)] = point.tolist()
  return normalized


def _reconstruction_context(world_extrinsics_file, calibration_file=None):
  world_path = _existing_file(world_extrinsics_file, "world extrinsics")
  world_extrinsics = load_json_or_yaml(world_path)
  if not isinstance(world_extrinsics, dict):
    raise ValueError("world extrinsics must contain a mapping")
  calibration_path = _calibration_path(
    calibration_file, world_path, world_extrinsics
  )
  calibration = load_calibration_json(calibration_path)
  models = build_camera_models(calibration, world_extrinsics)
  if len(models) < 2:
    raise ValueError(
      "at least two cameras need calibration and world extrinsics"
    )
  metadata = {
    "coordinate_frame": "world",
    "world_units": world_extrinsics.get("world_units", "meters"),
    "calibration": str(calibration_path),
    "world_extrinsics": str(world_path)
  }
  return models, metadata


def _reconstruct_with_models(
    models, metadata, observations, reprojection_threshold=1.5,
    min_ray_angle_deg=8.0, refine=True, refine_loss="soft_l1",
    refine_max_iterations=50, point_id="input"):

  observations = _normalized_observations(observations)
  unknown = sorted(set(observations) - set(models))
  if unknown:
    raise ValueError(
      "unknown or unavailable cameras: {}; available cameras: {}".format(
        ", ".join(unknown), ", ".join(sorted(models))
      )
    )
  if len(observations) < 2 and point_id == "input":
    raise ValueError(
      "3D triangulation requires synchronized pixels from at least two cameras"
    )
  frame = {"frame": point_id, "observations": observations}
  triangulated = triangulate_frame(
    frame,
    models,
    reprojection_threshold=reprojection_threshold,
    min_ray_angle_deg=min_ray_angle_deg,
    refine=refine,
    refine_loss=refine_loss,
    refine_max_iterations=refine_max_iterations
  )
  return {
    **metadata,
    "input_observations": observations,
    **triangulated
  }


def reconstruct_point(
    world_extrinsics_file, observations, calibration_file=None,
    reprojection_threshold=1.5, min_ray_angle_deg=8.0,
    refine=True, refine_loss="soft_l1", refine_max_iterations=50):
  """Return one triangulation result in the accepted world coordinate frame."""
  models, metadata = _reconstruction_context(
    world_extrinsics_file, calibration_file
  )
  return _reconstruct_with_models(
    models,
    metadata,
    observations,
    reprojection_threshold,
    min_ray_angle_deg,
    refine,
    refine_loss,
    refine_max_iterations
  )


def _read_txt_points(filename):
  path = _existing_file(filename, "pixel TXT input")
  points = []
  identifiers = set()
  for line_number, raw_line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    fields = line.split()
    if len(fields) < 7 or (len(fields) - 1) % 3:
      raise ValueError(
        "TXT line {} must be: POINT_ID CAMERA U V CAMERA U V ...".format(
          line_number
        )
      )
    point_id = fields[0]
    if point_id in identifiers:
      raise ValueError("duplicate point ID {}".format(point_id))
    identifiers.add(point_id)
    pixels = [
      fields[index:index + 3]
      for index in range(1, len(fields), 3)
    ]
    points.append((point_id, _cli_observations(pixels)))
  if not points:
    raise ValueError("pixel TXT input contains no points")
  return points


def _read_camera_txt(filename, camera):
  path = _existing_file(filename, "pixel TXT for camera {}".format(camera))
  pixels = []
  for line_number, raw_line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    fields = line.replace(",", " ").split()
    if len(fields) != 2:
      raise ValueError(
        "camera {} TXT line {} must contain exactly: U V".format(
          camera, line_number
        )
      )
    pixels.append(_cli_observations([[camera, fields[0], fields[1]]])[camera])
  if not pixels:
    raise ValueError("camera {} TXT contains no pixels".format(camera))
  return pixels


def _read_separate_camera_txt(values):
  camera_pixels = {}
  sources = {}
  for camera, filename in values or []:
    if camera in camera_pixels:
      raise ValueError("duplicate TXT input for camera {}".format(camera))
    camera_pixels[camera] = _read_camera_txt(filename, camera)
    sources[camera] = str(_existing_file(
      filename, "pixel TXT for camera {}".format(camera)
    ))
  if len(camera_pixels) < 2:
    raise ValueError("separate camera TXT mode requires at least two cameras")

  counts = {camera: len(pixels) for camera, pixels in camera_pixels.items()}
  if len(set(counts.values())) != 1:
    details = ", ".join(
      "{}={}".format(camera, count) for camera, count in sorted(counts.items())
    )
    raise ValueError(
      "camera TXT files must have the same number of valid rows: {}".format(
        details
      )
    )
  point_count = next(iter(counts.values()))
  points = [
    (
      "P{:06d}".format(index + 1),
      {
        camera: pixels[index]
        for camera, pixels in camera_pixels.items()
      }
    )
    for index in range(point_count)
  ]
  return points, sources


def _frame_sort_key(frame):
  value = str(frame)
  try:
    return 0, int(value)
  except ValueError:
    return 1, value


def _merge_camera_frames(camera_frames):
  if len(camera_frames) < 2:
    raise ValueError("batch reconstruction requires at least two cameras")
  frame_ids = sorted(
    set().union(*(set(frames) for frames in camera_frames.values())),
    key=_frame_sort_key
  )
  if not frame_ids:
    raise ValueError("camera inputs contain no frames")
  return [
    (
      str(frame_id),
      {
        camera: frames[frame_id]
        for camera, frames in camera_frames.items()
        if frames.get(frame_id) is not None
      }
    )
    for frame_id in frame_ids
  ]


def _read_camera_csv(filename, camera):
  path = _existing_file(filename, "CSV for camera {}".format(camera))
  with path.open(newline="", encoding="utf-8-sig") as stream:
    reader = csv.DictReader(stream)
    required = {"Frame", "Visibility", "X", "Y"}
    missing = required - set(reader.fieldnames or [])
    if missing:
      raise ValueError(
        "camera {} CSV is missing columns: {}".format(
          camera, ", ".join(sorted(missing))
        )
      )
    frames = {}
    seen_frames = set()
    for line_number, row in enumerate(reader, start=2):
      frame = str(row["Frame"]).strip()
      if not frame:
        raise ValueError(
          "camera {} CSV line {} has an empty Frame".format(
            camera, line_number
          )
        )
      if frame in seen_frames:
        raise ValueError(
          "camera {} CSV has duplicate Frame {}".format(camera, frame)
        )
      seen_frames.add(frame)
      visibility = str(row["Visibility"]).strip().lower()
      raw_x = str(row["X"] or "").strip()
      raw_y = str(row["Y"] or "").strip()
      if (
          visibility in {"0", "false", "no", "n", ""}
          or not raw_x or not raw_y):
        frames[frame] = None
        continue
      frames[frame] = _cli_observations([[
        camera, raw_x, raw_y
      ]])[camera]
  return frames


def _trajectory_camera_csvs(trajectory_dir, camera_name_map=None):
  directory = Path(trajectory_dir).expanduser().resolve()
  if not directory.is_dir():
    raise FileNotFoundError("trajectory directory not found: {}".format(directory))
  mapping = dict(camera_name_map or {})
  inputs = []
  for path in sorted(directory.glob("cam*/ball.csv")):
    source_camera = path.parent.name
    calibration_camera = mapping.get(source_camera, source_camera)
    inputs.append([calibration_camera, str(path)])
  if len(inputs) < 2:
    raise ValueError(
      "trajectory directory needs at least two cam*/ball.csv files: {}".format(
        directory
      )
    )
  destination_names = [camera for camera, _ in inputs]
  if len(destination_names) != len(set(destination_names)):
    raise ValueError("camera name map produces duplicate calibration cameras")
  return inputs, directory


def _labelme_ball_center(data, camera, filename):
  shapes = data.get("shapes") if isinstance(data, dict) else None
  if not isinstance(shapes, list):
    raise ValueError("LabelMe JSON has no shapes list: {}".format(filename))
  balls = [shape for shape in shapes if shape.get("label") == "ball"]
  if not balls:
    return None
  if len(balls) > 1:
    raise ValueError("LabelMe JSON has multiple ball shapes: {}".format(filename))
  shape = balls[0]
  points = shape.get("points")
  if not isinstance(points, list) or len(points) != 2:
    raise ValueError(
      "ball shape must contain two rectangle points: {}".format(filename)
    )
  try:
    u = (float(points[0][0]) + float(points[1][0])) / 2.0
    v = (float(points[0][1]) + float(points[1][1])) / 2.0
  except (IndexError, TypeError, ValueError) as error:
    raise ValueError("invalid ball rectangle points: {}".format(filename)) from error
  return _cli_observations([[camera, str(u), str(v)]])[camera]


def _read_camera_json_dir(dirname, camera):
  directory = Path(dirname).expanduser().resolve()
  if not directory.is_dir():
    raise FileNotFoundError(
      "LabelMe JSON directory for camera {} not found: {}".format(
        camera, directory
      )
    )
  frames = {}
  for path in sorted(directory.glob("*.json"), key=lambda item: _frame_sort_key(item.stem)):
    frame = path.stem
    if frame in frames:
      raise ValueError(
        "camera {} JSON directory has duplicate Frame {}".format(camera, frame)
      )
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
      raise ValueError("invalid LabelMe JSON: {}".format(path)) from error
    center = _labelme_ball_center(data, camera, path)
    if center is not None:
      frames[frame] = center
  return frames


def _read_camera_frame_sources(values, reader, label):
  camera_frames = {}
  sources = {}
  for camera, filename in values or []:
    if camera in camera_frames:
      raise ValueError("duplicate {} input for camera {}".format(label, camera))
    camera_frames[camera] = reader(filename, camera)
    sources[camera] = str(Path(filename).expanduser().resolve())
  return _merge_camera_frames(camera_frames), sources


def _write_txt_results(results, output_file):
  lines = [
    "point_id\tX\tY\tZ\tstatus\treprojection_rms_px\t"
    "max_ray_angle_deg\treason"
  ]
  for result in results:
    if result["status"] == "ok":
      coordinates = [
        "{:.12g}".format(value) for value in result["point_world"]
      ]
      rms = "{:.12g}".format(result["reprojection_rms_px"])
      angle = "{:.12g}".format(result["max_ray_angle_deg"])
      reason = ""
    else:
      coordinates = ["nan", "nan", "nan"]
      rms = "nan"
      angle = "nan"
      reason = str(result.get("reason", "failed")).replace("\t", " ")
    lines.append("\t".join([
      str(result["frame"]),
      *coordinates,
      result["status"],
      rms,
      angle,
      reason
    ]))

  destination = Path(output_file).expanduser().resolve()
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return destination


def _reconstruct_points(
    points, world_extrinsics_file, calibration_file=None,
    reprojection_threshold=1.5, min_ray_angle_deg=8.0,
    refine=True, refine_loss="soft_l1", refine_max_iterations=50):
  models, metadata = _reconstruction_context(
    world_extrinsics_file, calibration_file
  )
  return [
    _reconstruct_with_models(
      models,
      metadata,
      observations,
      reprojection_threshold,
      min_ray_angle_deg,
      refine,
      refine_loss,
      refine_max_iterations,
      point_id=point_id
    )
    for point_id, observations in points
  ]


def reconstruct_txt(
    input_file, output_file, world_extrinsics_file,
    calibration_file=None, reprojection_threshold=1.5,
    min_ray_angle_deg=8.0, refine=True, refine_loss="soft_l1",
    refine_max_iterations=50):
  """Reconstruct all TXT rows and write a tab-delimited result file."""
  points = _read_txt_points(input_file)
  results = _reconstruct_points(
    points, world_extrinsics_file, calibration_file,
    reprojection_threshold, min_ray_angle_deg, refine,
    refine_loss, refine_max_iterations
  )
  destination = _write_txt_results(results, output_file)
  return results, destination


def reconstruct_camera_txts(
    camera_txt_files, output_file, world_extrinsics_file,
    calibration_file=None, reprojection_threshold=1.5,
    min_ray_angle_deg=8.0, refine=True, refine_loss="soft_l1",
    refine_max_iterations=50):
  """Reconstruct corresponding rows from one TXT file per camera."""
  points, sources = _read_separate_camera_txt(camera_txt_files)
  results = _reconstruct_points(
    points, world_extrinsics_file, calibration_file,
    reprojection_threshold, min_ray_angle_deg, refine,
    refine_loss, refine_max_iterations
  )
  destination = _write_txt_results(results, output_file)
  return results, destination, sources


def reconstruct_camera_csvs(
    camera_csv_files, output_file, world_extrinsics_file,
    calibration_file=None, reprojection_threshold=1.5,
    min_ray_angle_deg=8.0, refine=True, refine_loss="soft_l1",
    refine_max_iterations=50):
  """Reconstruct frames from one detector CSV file per camera."""
  points, sources = _read_camera_frame_sources(
    camera_csv_files, _read_camera_csv, "CSV"
  )
  results = _reconstruct_points(
    points, world_extrinsics_file, calibration_file,
    reprojection_threshold, min_ray_angle_deg, refine,
    refine_loss, refine_max_iterations
  )
  destination = _write_txt_results(results, output_file)
  return results, destination, sources


def reconstruct_trajectory_dir(
    trajectory_dir, output_file, world_extrinsics_file,
    camera_name_map=None, calibration_file=None,
    reprojection_threshold=1.5, min_ray_angle_deg=8.0,
    refine=True, refine_loss="soft_l1", refine_max_iterations=50):
  """Reconstruct a traj_*/cam*/ball.csv directory."""
  camera_csv_files, _ = _trajectory_camera_csvs(
    trajectory_dir, camera_name_map
  )
  return reconstruct_camera_csvs(
    camera_csv_files, output_file, world_extrinsics_file,
    calibration_file, reprojection_threshold, min_ray_angle_deg,
    refine, refine_loss, refine_max_iterations
  )


def reconstruct_dataset_dir(
    dataset_dir, output_dir, world_extrinsics_file,
    camera_name_map=None, calibration_file=None,
    reprojection_threshold=1.5, min_ray_angle_deg=8.0,
    refine=True, refine_loss="soft_l1", refine_max_iterations=50):
  """Reconstruct every traj_* directory below a dataset directory."""
  root = Path(dataset_dir).expanduser().resolve()
  if not root.is_dir():
    raise FileNotFoundError("dataset directory not found: {}".format(root))
  trajectories = sorted(
    path for path in root.glob("traj_*")
    if path.is_dir() and len(list(path.glob("cam*/ball.csv"))) >= 2
  )
  if not trajectories:
    raise ValueError(
      "dataset directory contains no traj_*/cam*/ball.csv inputs: {}".format(
        root
      )
    )

  destination_root = (
    Path(output_dir).expanduser().resolve() if output_dir else None
  )
  batches = []
  for trajectory in trajectories:
    output_file = (
      destination_root / trajectory.name / DEFAULT_OUTPUT_TXT
      if destination_root is not None
      else trajectory / DEFAULT_OUTPUT_TXT
    )
    results, destination, sources = reconstruct_trajectory_dir(
      trajectory,
      output_file,
      world_extrinsics_file,
      camera_name_map,
      calibration_file,
      reprojection_threshold,
      min_ray_angle_deg,
      refine,
      refine_loss,
      refine_max_iterations
    )
    failed_count = sum(result["status"] != "ok" for result in results)
    batches.append({
      "trajectory": trajectory.name,
      "frame_count": len(results),
      "reconstructed_count": len(results) - failed_count,
      "failed_count": failed_count,
      "output": str(destination),
      "sources": sources
    })
  return batches


def reconstruct_camera_json_dirs(
    camera_json_dirs, output_file, world_extrinsics_file,
    calibration_file=None, reprojection_threshold=1.5,
    min_ray_angle_deg=8.0, refine=True, refine_loss="soft_l1",
    refine_max_iterations=50):
  """Reconstruct frames from one LabelMe JSON directory per camera."""
  points, sources = _read_camera_frame_sources(
    camera_json_dirs, _read_camera_json_dir, "JSON directory"
  )
  results = _reconstruct_points(
    points, world_extrinsics_file, calibration_file,
    reprojection_threshold, min_ray_angle_deg, refine,
    refine_loss, refine_max_iterations
  )
  destination = _write_txt_results(results, output_file)
  return results, destination, sources


def _parser():
  parser = argparse.ArgumentParser(
    description=(
      "Reconstruct one world 3D coordinate from synchronized pixels in "
      "at least two cameras."
    )
  )
  parser.add_argument(
    "--world-extrinsics",
    default=str(DEFAULT_WORLD_EXTRINSICS),
    help="world extrinsics JSON (default: %(default)s)"
  )
  parser.add_argument(
    "--calibration",
    default=DEFAULT_CALIBRATION,
    help=(
      "intrinsic calibration JSON; defaults to the intrinsic path declared "
      "by the world extrinsics"
    )
  )
  source = parser.add_mutually_exclusive_group()
  source.add_argument(
    "--pixel",
    action="append",
    nargs=3,
    metavar=("CAMERA", "U", "V"),
    help="one camera pixel; repeat for every synchronized camera"
  )
  source.add_argument(
    "--input",
    help=(
      "JSON/YAML file containing observations, or '-' to read JSON from stdin"
    )
  )
  source.add_argument(
    "--input-txt",
    help="batch TXT input: POINT_ID CAMERA U V CAMERA U V ... per line"
  )
  source.add_argument(
    "--camera-txt",
    action="append",
    nargs=2,
    metavar=("CAMERA", "FILE"),
    help=(
      "one U V TXT file for a camera; repeat for at least two cameras and "
      "keep corresponding pixels on matching valid rows"
    )
  )
  source.add_argument(
    "--camera-csv",
    action="append",
    nargs=2,
    metavar=("CAMERA", "FILE"),
    help="detector CSV for a camera; repeat for at least two cameras"
  )
  source.add_argument(
    "--camera-json-dir",
    action="append",
    nargs=2,
    metavar=("CAMERA", "DIRECTORY"),
    help="LabelMe JSON directory for a camera; repeat for at least two cameras"
  )
  source.add_argument(
    "--trajectory-dir",
    help="trajectory directory containing cam*/ball.csv"
  )
  source.add_argument(
    "--dataset-dir",
    help="dataset directory containing multiple traj_*/cam*/ball.csv inputs"
  )
  parser.add_argument(
    "--camera-map",
    action="append",
    nargs=2,
    metavar=("INPUT_CAMERA", "CALIBRATION_CAMERA"),
    help="map a trajectory camera directory name to a calibrated camera"
  )
  parser.add_argument(
    "--output-txt",
    help="tab-delimited 3D output path; required with --input-txt"
  )
  parser.add_argument(
    "--output-dir",
    help=(
      "batch output root for --dataset-dir; defaults to writing "
      "points_3d.txt inside each trajectory"
    )
  )
  parser.add_argument(
    "--reprojection-threshold",
    type=float,
    default=DEFAULT_REPROJECTION_THRESHOLD,
    help="maximum inlier reprojection error in pixels (default: %(default)s)"
  )
  parser.add_argument(
    "--min-ray-angle-deg",
    type=float,
    default=DEFAULT_MIN_RAY_ANGLE_DEG,
    help="minimum accepted triangulation ray angle (default: %(default)s)"
  )
  parser.add_argument(
    "--refine-loss",
    choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
    default=DEFAULT_REFINE_LOSS,
    help="nonlinear refinement loss (default: %(default)s)"
  )
  parser.add_argument(
    "--no-refine",
    action="store_true",
    help="disable nonlinear point refinement"
  )
  parser.add_argument(
    "--refine-max-iterations",
    type=int,
    default=DEFAULT_REFINE_MAX_ITERATIONS,
    help="maximum nonlinear refinement evaluations (default: %(default)s)"
  )
  parser.add_argument(
    "--format",
    choices=("json", "xyz"),
    default="json",
    help="stdout format; xyz prints three space-separated values"
  )
  return parser


def main(argv=None):
  args = _parser().parse_args(argv)
  try:
    no_input_arguments = not any((
      args.pixel, args.input, args.input_txt, args.camera_txt,
      args.camera_csv, args.camera_json_dir, args.trajectory_dir,
      args.dataset_dir
    ))
    if no_input_arguments:
      if DEFAULT_DATASET_DIR:
        args.dataset_dir = DEFAULT_DATASET_DIR
      elif DEFAULT_TRAJECTORY_DIR:
        args.trajectory_dir = DEFAULT_TRAJECTORY_DIR
      elif DEFAULT_CAMERA_TXT_FILES:
        args.camera_txt = [
          [camera, filename]
          for camera, filename in DEFAULT_CAMERA_TXT_FILES.items()
        ]
      else:
        raise ValueError(
          "no input specified; configure DEFAULT_DATASET_DIR, "
          "DEFAULT_TRAJECTORY_DIR or "
          "DEFAULT_CAMERA_TXT_FILES at the "
          "top of the script or pass an input argument"
        )
      if not args.output_txt:
        args.output_txt = DEFAULT_OUTPUT_TXT
    if args.reprojection_threshold <= 0:
      raise ValueError("reprojection threshold must be positive")
    if args.min_ray_angle_deg < 0:
      raise ValueError("minimum ray angle must be non-negative")
    if args.refine_max_iterations <= 0:
      raise ValueError("refine max iterations must be positive")
    batch_input = any((
      args.input_txt is not None,
      args.camera_txt is not None,
      args.camera_csv is not None,
      args.camera_json_dir is not None,
      args.trajectory_dir is not None,
      args.dataset_dir is not None
    ))
    if batch_input:
      if (
          not args.output_txt
          and args.trajectory_dir is None
          and args.dataset_dir is None):
        raise ValueError(
          "--output-txt is required with batch input"
        )
      common = {
        "calibration_file": args.calibration,
        "reprojection_threshold": args.reprojection_threshold,
        "min_ray_angle_deg": args.min_ray_angle_deg,
        "refine": DEFAULT_REFINE and not args.no_refine,
        "refine_loss": args.refine_loss,
        "refine_max_iterations": args.refine_max_iterations
      }
      camera_map = dict(DEFAULT_CAMERA_NAME_MAP)
      for source_camera, calibration_camera in args.camera_map or []:
        camera_map[source_camera] = calibration_camera
      if args.dataset_dir is not None:
        if args.output_txt:
          raise ValueError(
            "--output-txt cannot be used with --dataset-dir; use --output-dir"
          )
        batches = reconstruct_dataset_dir(
          args.dataset_dir,
          args.output_dir or DEFAULT_OUTPUT_DIR,
          args.world_extrinsics,
          camera_name_map=camera_map,
          **common
        )
        frame_count = sum(batch["frame_count"] for batch in batches)
        reconstructed_count = sum(
          batch["reconstructed_count"] for batch in batches
        )
        failed_count = sum(batch["failed_count"] for batch in batches)
        print(json.dumps({
          "status": "ok" if not failed_count else "partial",
          "trajectory_count": len(batches),
          "frame_count": frame_count,
          "reconstructed_count": reconstructed_count,
          "failed_count": failed_count,
          "trajectories": batches
        }, ensure_ascii=False, indent=2))
        return 0 if not failed_count else 1
      if args.output_dir:
        raise ValueError("--output-dir can only be used with --dataset-dir")
      if args.trajectory_dir is not None:
        if not args.output_txt:
          args.output_txt = str(
            Path(args.trajectory_dir).expanduser() / DEFAULT_OUTPUT_TXT
          )
        results, destination, _ = reconstruct_trajectory_dir(
          args.trajectory_dir, args.output_txt, args.world_extrinsics,
          camera_name_map=camera_map, **common
        )
      elif args.camera_csv is not None:
        results, destination, _ = reconstruct_camera_csvs(
          args.camera_csv, args.output_txt, args.world_extrinsics, **common
        )
      elif args.camera_json_dir is not None:
        results, destination, _ = reconstruct_camera_json_dirs(
          args.camera_json_dir, args.output_txt,
          args.world_extrinsics, **common
        )
      elif args.camera_txt is not None:
        results, destination, _ = reconstruct_camera_txts(
          args.camera_txt,
          args.output_txt,
          args.world_extrinsics,
          **common
        )
      else:
        results, destination = reconstruct_txt(
          args.input_txt,
          args.output_txt,
          args.world_extrinsics,
          **common
        )
      failed_count = sum(result["status"] != "ok" for result in results)
      print(json.dumps({
        "status": "ok" if not failed_count else "partial",
        "point_count": len(results),
        "reconstructed_count": len(results) - failed_count,
        "failed_count": failed_count,
        "output": str(destination)
      }, ensure_ascii=False))
      return 0 if not failed_count else 1
    if args.output_txt:
      raise ValueError(
        "--output-txt can only be used with batch input"
      )
    if args.output_dir:
      raise ValueError("--output-dir can only be used with --dataset-dir")
    observations = (
      _file_observations(args.input)
      if args.input is not None else _cli_observations(args.pixel)
    )
    result = reconstruct_point(
      args.world_extrinsics,
      observations,
      calibration_file=args.calibration,
      reprojection_threshold=args.reprojection_threshold,
      min_ray_angle_deg=args.min_ray_angle_deg,
      refine=DEFAULT_REFINE and not args.no_refine,
      refine_loss=args.refine_loss,
      refine_max_iterations=args.refine_max_iterations
    )
  except (KeyError, OSError, TypeError, ValueError) as error:
    print(json.dumps({
      "status": "error",
      "reason": str(error)
    }, ensure_ascii=False), file=sys.stderr)
    return 2

  if result["status"] != "ok":
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1
  if args.format == "xyz":
    print(" ".join("{:.12g}".format(value) for value in result["point_world"]))
  else:
    print(json.dumps(result, ensure_ascii=False, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
