"""Create a random frame subset without modifying the source dataset.

Examples:
  # 从 [0, 52] 中随机删除 10 帧，复制其余帧
  python scripts/random_frame_dataset.py SOURCE OUTPUT \
    --range 0 52 --mode drop --count 10 --seed 42

  # 从 [0, 52] 中随机抽取 20 帧作为新数据集
  python scripts/random_frame_dataset.py SOURCE OUTPUT \
    --range 0 52 --mode sample --count 20 --seed 42

For a multi-camera dataset, a frame is selected once and the corresponding
image is copied from every selected camera, preserving synchronization.
"""

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

from natsort import natsorted


IMAGE_EXTENSIONS = {
  ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"
}
FRAME_NUMBER = re.compile(r"^\d+$")
DEFAULT_MANIFEST = "random_frame_dataset.json"


def frame_number(path):
  """Return the integer filename stem, or None for a non-numeric filename."""
  if not FRAME_NUMBER.fullmatch(path.stem):
    return None
  return int(path.stem)


def camera_images(directory):
  images = {}
  for path in directory.iterdir():
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
      continue
    number = frame_number(path)
    if number is None:
      continue
    if number in images:
      raise ValueError(
        "camera {} has multiple images for frame {}: {} and {}".format(
          directory.name, number, images[number].name, path.name
        )
      )
    images[number] = path
  return images


def discover_cameras(source, requested=None):
  if requested:
    names = list(dict.fromkeys(requested))
  else:
    names = [
      path.name for path in source.iterdir()
      if path.is_dir() and path.name.lower().startswith("cam")
    ]
    names = natsorted(names)
  if not names:
    raise ValueError("no camera directories found in {}".format(source))

  cameras = {}
  for name in names:
    directory = source / name
    if not directory.is_dir():
      raise ValueError("camera directory does not exist: {}".format(directory))
    images = camera_images(directory)
    if not images:
      raise ValueError(
        "camera directory has no numeric image filenames: {}".format(directory)
      )
    cameras[name] = images
  return cameras


def select_frames(cameras, start, end, mode, count, seed):
  if start > end:
    raise ValueError("range start must not be greater than range end")
  if count < 0:
    raise ValueError("count must be non-negative")

  complete = set.intersection(*(set(images) for images in cameras.values()))
  candidates = sorted(number for number in complete if start <= number <= end)
  if count > len(candidates):
    raise ValueError(
      "cannot {} {} frames from only {} complete frames in [{}, {}]".format(
        mode, count, len(candidates), start, end
      )
    )

  chosen = sorted(random.Random(seed).sample(candidates, count))
  chosen_set = set(chosen)
  if mode == "sample":
    output_frames = chosen
    excluded_frames = sorted(set(candidates) - chosen_set)
  elif mode == "drop":
    output_frames = sorted(set(candidates) - chosen_set)
    excluded_frames = chosen
  else:
    raise ValueError("unknown mode {!r}".format(mode))

  present_in_range = set().union(*(
    {number for number in images if start <= number <= end}
    for images in cameras.values()
  ))
  incomplete = sorted(present_in_range - complete)
  return {
    "candidate_frames": candidates,
    "random_frames": chosen,
    "output_frames": output_frames,
    "excluded_frames": excluded_frames,
    "incomplete_frames": incomplete
  }


def build_dataset(
    source, output, start, end, mode, count, seed=0, cameras=None,
    dry_run=False):
  source = Path(source).expanduser().resolve()
  output = Path(output).expanduser().resolve()
  if not source.is_dir():
    raise ValueError("source dataset does not exist: {}".format(source))
  if output.exists():
    raise ValueError("output path already exists: {}".format(output))
  if output == source or source in output.parents:
    raise ValueError("output must not be inside the source dataset")

  camera_map = discover_cameras(source, cameras)
  selection = select_frames(
    camera_map, int(start), int(end), mode, int(count), int(seed)
  )
  copied = []
  for camera, images in camera_map.items():
    for number in selection["output_frames"]:
      source_path = images[number]
      copied.append({
        "camera": camera,
        "frame": number,
        "source": str(source_path),
        "output": str(output / camera / source_path.name)
      })

  manifest = {
    "version": 1,
    "source": str(source),
    "output": str(output),
    "mode": mode,
    "range": [int(start), int(end)],
    "count": int(count),
    "seed": int(seed),
    "cameras": list(camera_map),
    **selection,
    "output_frame_count": len(selection["output_frames"]),
    "files_copied": len(copied),
    "dry_run": bool(dry_run)
  }

  if dry_run:
    return manifest

  output.mkdir(parents=True)
  try:
    for camera in camera_map:
      (output / camera).mkdir()
    for record in copied:
      shutil.copy2(record["source"], record["output"])
    (output / DEFAULT_MANIFEST).write_text(
      json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8"
    )
  except Exception:
    shutil.rmtree(output)
    raise
  return manifest


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description=(
      "Randomly drop or sample synchronized numeric image frames into a new "
      "dataset. The source dataset is never modified."
    )
  )
  parser.add_argument("source", help="Dataset root containing cam* directories.")
  parser.add_argument("output", help="New dataset path; it must not exist.")
  parser.add_argument(
    "--range", nargs=2, type=int, metavar=("START", "END"), required=True,
    dest="frame_range", help="Inclusive numeric filename range [START, END]."
  )
  parser.add_argument(
    "--mode", choices=("drop", "sample"), required=True,
    help="drop: remove N random frames; sample: keep N random frames."
  )
  parser.add_argument("--count", "-n", type=int, required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--cameras", nargs="+",
    help="Camera directory names (default: discover all cam* directories)."
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="Print the selection without creating the output dataset."
  )
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  try:
    result = build_dataset(
      args.source,
      args.output,
      args.frame_range[0],
      args.frame_range[1],
      args.mode,
      args.count,
      seed=args.seed,
      cameras=args.cameras,
      dry_run=args.dry_run
    )
  except (OSError, ValueError) as error:
    print("error: {}".format(error), file=sys.stderr)
    return 2
  print(json.dumps(result, indent=2, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
