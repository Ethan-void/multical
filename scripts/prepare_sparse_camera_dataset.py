"""Fill missing multi-camera frames with safe, solid-color placeholders."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}
DEFAULT_MANIFEST = "placeholder_manifest.json"


def image_files(directory):
  return [
    path for path in directory.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
  ]


def camera_image_size(camera_directory):
  files = image_files(camera_directory)
  if not files:
    raise ValueError(
      "camera directory {} contains no real image from which to infer its "
      "resolution".format(camera_directory)
    )
  expected = None
  for image_path in files:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
      raise ValueError("could not read image {}".format(image_path))
    size = (int(image.shape[1]), int(image.shape[0]))
    if expected is None:
      expected = size
    elif size != expected:
      raise ValueError(
        "camera {} contains mixed image sizes {} and {}".format(
          camera_directory.name, expected, size
        )
      )
  return expected


def file_sha256(filename):
  digest = hashlib.sha256()
  with Path(filename).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _safe_dataset_path(root, relative):
  root = root.resolve()
  target = (root / relative).resolve()
  if not target.is_relative_to(root):
    raise ValueError(
      "manifest path escapes dataset root: {}".format(relative)
    )
  return target


def fill_placeholders(
    dataset_root,
    cameras,
    gray=127,
    manifest_name=DEFAULT_MANIFEST,
    dry_run=False
):
  root = Path(dataset_root).expanduser().resolve()
  if not root.is_dir():
    raise ValueError("dataset root does not exist {}".format(root))
  if not cameras:
    raise ValueError("at least one camera is required")
  gray = int(gray)
  if not 0 <= gray <= 255:
    raise ValueError("gray must be between 0 and 255")

  camera_directories = {}
  sizes = {}
  camera_frames = {}
  for camera in cameras:
    directory = root / camera
    if not directory.is_dir():
      raise ValueError(
        "camera directory does not exist {}".format(directory)
      )
    camera_directories[camera] = directory
    sizes[camera] = camera_image_size(directory)
    camera_frames[camera] = {
      path.name for path in image_files(directory)
    }

  frames = natsorted(set().union(*camera_frames.values()))
  missing = [
    (camera, frame)
    for frame in frames
    for camera in cameras
    if frame not in camera_frames[camera]
  ]
  manifest_path = _safe_dataset_path(root, manifest_name)
  previous_entries = []
  if manifest_path.is_file():
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_entries = previous.get("created_placeholders", [])

  created = []
  for camera, frame in missing:
    width, height = sizes[camera]
    target = camera_directories[camera] / frame
    entry = {
      "camera": camera,
      "frame": frame,
      "path": str(target.relative_to(root)),
      "image_size": [width, height],
      "gray": gray
    }
    if not dry_run:
      placeholder = np.full(
        (height, width, 3), gray, dtype=np.uint8
      )
      temporary = target.with_name(
        target.stem + ".placeholder-tmp" + target.suffix
      )
      if not cv2.imwrite(str(temporary), placeholder):
        raise RuntimeError(
          "could not write placeholder {}".format(temporary)
        )
      os.replace(str(temporary), str(target))
      entry["sha256"] = file_sha256(target)
    created.append(entry)

  if not dry_run:
    by_path = {
      entry["path"]: entry
      for entry in previous_entries + created
    }
    manifest = {
      "version": 1,
      "dataset_root": str(root),
      "cameras": list(cameras),
      "camera_image_sizes": {
        camera: list(sizes[camera]) for camera in cameras
      },
      "frame_count": len(frames),
      "created_placeholders": [
        by_path[key] for key in sorted(by_path)
      ]
    }
    manifest_path.write_text(
      json.dumps(manifest, indent=2) + "\n",
      encoding="utf-8"
    )

  return {
    "dataset_root": str(root),
    "frame_count": len(frames),
    "missing_count": len(missing),
    "created": created,
    "dry_run": bool(dry_run),
    "manifest": str(manifest_path)
  }


def remove_placeholders(dataset_root, manifest_name=DEFAULT_MANIFEST):
  root = Path(dataset_root).expanduser().resolve()
  manifest_path = _safe_dataset_path(root, manifest_name)
  if not manifest_path.is_file():
    raise ValueError(
      "placeholder manifest not found {}".format(manifest_path)
    )
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  entries = manifest.get("created_placeholders", [])
  changed = []
  for entry in entries:
    target = _safe_dataset_path(root, entry["path"])
    if not target.is_file():
      continue
    expected_hash = entry.get("sha256")
    if not expected_hash or file_sha256(target) != expected_hash:
      changed.append(str(target))
  if changed:
    raise ValueError(
      "refusing to remove placeholders that were modified: {}".format(
        ", ".join(changed)
      )
    )

  removed = []
  for entry in entries:
    target = _safe_dataset_path(root, entry["path"])
    if target.is_file():
      target.unlink()
      removed.append(str(target))
  manifest_path.unlink()
  return {
    "dataset_root": str(root),
    "removed_count": len(removed),
    "removed": removed
  }


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      "Create solid placeholders for filenames missing from one or more "
      "camera directories."
    )
  )
  parser.add_argument(
    "--image_path", "--dataset-root", dest="image_path", required=True,
    help="Directory containing one subdirectory per camera."
  )
  parser.add_argument(
    "--cameras", nargs="+",
    help="Camera directory names, for example C1 C2 C3 C4."
  )
  parser.add_argument(
    "--gray", type=int, default=127,
    help="Placeholder gray value from 0 to 255 (default: 127)."
  )
  parser.add_argument(
    "--manifest", default=DEFAULT_MANIFEST,
    help="Manifest filename relative to dataset root."
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="Report missing frames without writing placeholders."
  )
  parser.add_argument(
    "--remove", action="store_true",
    help="Remove only unchanged placeholders recorded in the manifest."
  )
  return parser.parse_args()


def main():
  args = parse_args()
  if not args.remove and not args.cameras:
    raise ValueError("--cameras is required unless --remove is used")
  result = (
    remove_placeholders(args.image_path, args.manifest)
    if args.remove else
    fill_placeholders(
      args.image_path,
      args.cameras,
      gray=args.gray,
      manifest_name=args.manifest,
      dry_run=args.dry_run
    )
  )
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
