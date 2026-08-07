"""Create a coverage-aware calibration/validation image split.

The script detects a checkerboard in every image, describes each view by its
image position, apparent size, plane tilt, and in-plane rotation, then groups
similar views. Validation images are sampled from those groups while retaining
at least one calibration image in every non-empty group.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted
from omegaconf import OmegaConf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"}


def load_checkerboard_size(config_path, board_name=None):
  config = OmegaConf.to_container(
    OmegaConf.load(str(config_path)), resolve=True
  )
  boards = config.get("boards", {})
  if board_name is None:
    candidates = [
      name for name, board in boards.items()
      if board.get("_type_", board.get("type")) == "checkerboard"
    ]
    if len(candidates) != 1:
      raise ValueError(
        "board config must contain exactly one checkerboard; found {}. "
        "Use --board-name to choose one.".format(candidates)
      )
    board_name = candidates[0]
  if board_name not in boards:
    raise ValueError("board {!r} not found in {}".format(board_name, config_path))
  board = boards[board_name]
  board_type = board.get("_type_", board.get("type"))
  if board_type != "checkerboard":
    raise ValueError("board {!r} is not a checkerboard".format(board_name))
  size = tuple(int(value) for value in board["size"])
  if len(size) != 2 or min(size) < 2:
    raise ValueError("invalid checkerboard inner-corner size {}".format(size))
  return board_name, size


def find_images(image_dir):
  paths = [
    path for path in Path(image_dir).iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
  ]
  return [Path(path) for path in natsorted([str(path) for path in paths])]


def detect_checkerboard(gray, pattern_size):
  if hasattr(cv2, "findChessboardCornersSB"):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
    flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
    found, corners = cv2.findChessboardCornersSB(
      gray, pattern_size, flags=flags
    )
  else:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(
      gray, pattern_size, flags=flags
    )
    if found:
      corners = cv2.cornerSubPix(
        gray, corners.astype(np.float32), (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
      )
  if not found or corners is None:
    return None
  return np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def view_features(corners, image_shape, pattern_size):
  """Return interpretable features that do not require calibrated intrinsics."""
  height, width = image_shape[:2]
  centroid = corners.mean(axis=0)
  hull = cv2.convexHull(corners.astype(np.float32))
  area_ratio = max(cv2.contourArea(hull) / float(width * height), 1e-9)

  columns, rows = pattern_size
  object_points = np.zeros((columns * rows, 3), dtype=np.float64)
  object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
  focal_guess = float(max(width, height))
  camera_matrix = np.array([
    [focal_guess, 0.0, width / 2.0],
    [0.0, focal_guess, height / 2.0],
    [0.0, 0.0, 1.0]
  ])
  success, rotation_vector, _ = cv2.solvePnP(
    object_points,
    corners,
    camera_matrix,
    np.zeros(5),
    flags=cv2.SOLVEPNP_ITERATIVE
  )
  if not success:
    raise ValueError("could not estimate approximate checkerboard pose")
  rotation, _ = cv2.Rodrigues(rotation_vector)
  normal = rotation[:, 2]
  if normal[2] < 0:
    normal = -normal
  normal_z = max(float(normal[2]), 1e-6)

  grid = corners.reshape(rows, columns, 2)
  horizontal = (
    (grid[0, -1] - grid[0, 0]) +
    (grid[-1, -1] - grid[-1, 0])
  ) / 2.0
  roll = float(np.arctan2(horizontal[1], horizontal[0]))

  return np.array([
    centroid[0] / width,
    centroid[1] / height,
    np.log(area_ratio),
    normal[0] / normal_z,
    normal[1] / normal_z,
    np.cos(2.0 * roll),
    np.sin(2.0 * roll)
  ], dtype=np.float64)


def standardize(features):
  features = np.asarray(features, dtype=np.float64)
  center = np.median(features, axis=0)
  scale = np.percentile(features, 75, axis=0) - np.percentile(
    features, 25, axis=0
  )
  standard_deviation = features.std(axis=0)
  scale = np.where(scale > 1e-9, scale, standard_deviation)
  scale = np.where(scale > 1e-9, scale, 1.0)
  return (features - center) / scale


def kmeans(features, cluster_count, rng, max_iterations=100):
  """Small NumPy-only k-means implementation with k-means++ seeding."""
  features = np.asarray(features, dtype=np.float64)
  centers = [features[int(rng.integers(len(features)))]]
  while len(centers) < cluster_count:
    distances = np.min(
      np.stack([
        np.sum((features - center) ** 2, axis=1) for center in centers
      ]),
      axis=0
    )
    total = distances.sum()
    if total <= 1e-12:
      remaining = [
        index for index in range(len(features))
        if not any(np.array_equal(features[index], center) for center in centers)
      ]
      centers.append(features[remaining[0] if remaining else 0])
    else:
      centers.append(features[int(rng.choice(len(features), p=distances / total))])
  centers = np.asarray(centers)

  labels = np.zeros(len(features), dtype=np.int64)
  for _ in range(max_iterations):
    distances = np.stack([
      np.sum((features - center) ** 2, axis=1) for center in centers
    ], axis=1)
    new_labels = np.argmin(distances, axis=1)
    if np.array_equal(labels, new_labels):
      break
    labels = new_labels
    for cluster in range(cluster_count):
      members = features[labels == cluster]
      if len(members):
        centers[cluster] = members.mean(axis=0)
  return labels


def allocate_validation_counts(labels, validation_count, rng):
  clusters = np.unique(labels)
  sizes = {cluster: int(np.sum(labels == cluster)) for cluster in clusters}
  capacities = {cluster: max(0, size - 1) for cluster, size in sizes.items()}
  if sum(capacities.values()) < validation_count:
    raise ValueError(
      "cannot reserve {} validation images while retaining one calibration "
      "image per stratum".format(validation_count)
    )

  allocations = {cluster: 0 for cluster in clusters}
  eligible = [cluster for cluster in clusters if capacities[cluster] > 0]
  rng.shuffle(eligible)
  if validation_count >= len(eligible):
    for cluster in eligible:
      allocations[cluster] = 1

  while sum(allocations.values()) < validation_count:
    candidates = [
      cluster for cluster in clusters
      if allocations[cluster] < capacities[cluster]
    ]
    cluster = max(
      candidates,
      key=lambda item: (
        sizes[item] / float(allocations[item] + 1),
        float(rng.random())
      )
    )
    allocations[cluster] += 1
  return allocations


def stratified_split(records, validation_count, seed, cluster_count=None):
  if not 0 < validation_count < len(records):
    raise ValueError(
      "validation count must be between 1 and {}".format(len(records) - 1)
    )
  rng = np.random.default_rng(seed)
  if cluster_count is None:
    cluster_count = min(validation_count, max(3, validation_count // 2))
  cluster_count = int(max(1, min(cluster_count, len(records) - validation_count)))
  features = standardize([record["features"] for record in records])
  labels = kmeans(features, cluster_count, rng)
  allocations = allocate_validation_counts(labels, validation_count, rng)

  validation_indices = set()
  for cluster, count in allocations.items():
    members = np.flatnonzero(labels == cluster)
    chosen = rng.choice(members, size=count, replace=False)
    validation_indices.update(int(index) for index in chosen)

  calibration = []
  validation = []
  for index, (record, cluster) in enumerate(zip(records, labels)):
    output = dict(record)
    output["stratum"] = int(cluster)
    output["features"] = [float(value) for value in output["features"]]
    (validation if index in validation_indices else calibration).append(output)
  return calibration, validation


def copy_split(records, destination):
  destination.mkdir(parents=True, exist_ok=False)
  for record in records:
    shutil.copy2(record["path"], destination / record["filename"])


def parse_args():
  parser = argparse.ArgumentParser(
    description="Split checkerboard images by pose coverage."
  )
  parser.add_argument("image_dir", help="Directory containing input images.")
  parser.add_argument(
    "--boards", default=None,
    help="boards.yaml path (default: image_dir/boards.yaml)."
  )
  parser.add_argument("--board-name", help="Checkerboard name in boards.yaml.")
  parser.add_argument(
    "--validation-count", type=int, default=10,
    help="Number of held-out validation images (default: 10)."
  )
  parser.add_argument(
    "--clusters", type=int,
    help="Number of pose strata (default: half the validation count)."
  )
  parser.add_argument("--seed", type=int, default=20260803)
  parser.add_argument(
    "--manifest", default="intrinsic_split.json",
    help="Output JSON path (default: intrinsic_split.json)."
  )
  parser.add_argument(
    "--copy-to",
    help="Optionally copy images into calibration/ and validation/ directories."
  )
  parser.add_argument(
    "--allow-undetected", action="store_true",
    help="Exclude images without a full checkerboard instead of failing."
  )
  return parser.parse_args()


def main():
  args = parse_args()
  image_dir = Path(args.image_dir).expanduser().resolve()
  if not image_dir.is_dir():
    raise ValueError("image directory does not exist: {}".format(image_dir))
  boards_path = Path(args.boards).expanduser().resolve() if args.boards else (
    image_dir / "boards.yaml"
  )
  board_name, pattern_size = load_checkerboard_size(
    boards_path, args.board_name
  )
  images = find_images(image_dir)
  if not images:
    raise ValueError("no images found in {}".format(image_dir))

  records = []
  rejected = []
  expected_shape = None
  for image_path in images:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
      rejected.append({"filename": image_path.name, "reason": "unreadable"})
      continue
    if expected_shape is None:
      expected_shape = gray.shape
    elif gray.shape != expected_shape:
      rejected.append({
        "filename": image_path.name,
        "reason": "image size differs from {}".format(expected_shape[::-1])
      })
      continue
    corners = detect_checkerboard(gray, pattern_size)
    if corners is None:
      rejected.append({"filename": image_path.name, "reason": "not detected"})
      continue
    records.append({
      "filename": image_path.name,
      "path": str(image_path),
      "features": view_features(corners, gray.shape, pattern_size)
    })

  if rejected and not args.allow_undetected:
    raise ValueError(
      "{} of {} images were rejected; rerun with --allow-undetected to "
      "exclude them. First failures: {}".format(
        len(rejected), len(images), rejected[:5]
      )
    )
  calibration, validation = stratified_split(
    records, args.validation_count, args.seed, args.clusters
  )
  manifest = {
    "version": 1,
    "image_dir": str(image_dir),
    "board_config": str(boards_path),
    "board_name": board_name,
    "pattern_size_inner_corners": list(pattern_size),
    "seed": args.seed,
    "feature_order": [
      "center_x", "center_y", "log_area", "tilt_x", "tilt_y",
      "roll_cos_2x", "roll_sin_2x"
    ],
    "calibration": calibration,
    "validation": validation,
    "rejected": rejected
  }
  manifest_path = Path(args.manifest).expanduser().resolve()
  manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
  )

  if args.copy_to:
    output_root = Path(args.copy_to).expanduser().resolve()
    if output_root.exists():
      raise ValueError("copy destination already exists: {}".format(output_root))
    output_root.mkdir(parents=True)
    try:
      copy_split(calibration, output_root / "calibration")
      copy_split(validation, output_root / "validation")
    except Exception:
      shutil.rmtree(output_root)
      raise

  summary = {
    "calibration_count": len(calibration),
    "validation_count": len(validation),
    "rejected_count": len(rejected),
    "manifest": str(manifest_path),
    "copied_to": str(Path(args.copy_to).expanduser().resolve())
    if args.copy_to else None
  }
  print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
