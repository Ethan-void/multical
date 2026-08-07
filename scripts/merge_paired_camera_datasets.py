#!/usr/bin/env python3
"""Merge multiple synchronized two-camera batches into one dataset.

Each source batch contains a left/right camera directory. Images are matched by
relative filename, then copied to the output camera directories with a batch
prefix. The same output filename is always used for both cameras, preserving
frame synchronization without modifying the source data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}
DEFAULT_BATCHES = (("cam2", "cam3"), ("cam2_e1", "cam3_e2"))


class MergeError(RuntimeError):
  """Raised when source batches cannot be merged safely."""


def image_paths(directory: Path) -> dict[str, Path]:
  if not directory.is_dir():
    raise MergeError("camera directory does not exist: {}".format(directory))

  images = {}
  for path in sorted(directory.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
      continue
    relative = path.relative_to(directory).as_posix()
    images[relative] = path
  if not images:
    raise MergeError("camera directory contains no images: {}".format(directory))
  return images


def paired_images(
    root: Path, left_name: str, right_name: str
) -> list[tuple[Path, Path]]:
  left = image_paths(root / left_name)
  right = image_paths(root / right_name)
  left_names = set(left)
  right_names = set(right)
  if left_names != right_names:
    only_left = sorted(left_names - right_names)
    only_right = sorted(right_names - left_names)
    details = []
    if only_left:
      details.append("only in {}: {}".format(
        left_name, ", ".join(only_left[:10])))
    if only_right:
      details.append("only in {}: {}".format(
        right_name, ", ".join(only_right[:10])))
    raise MergeError(
      "{} and {} are not one-to-one ({} vs {} images); {}".format(
        left_name, right_name, len(left), len(right), "; ".join(details)))
  return [(left[name], right[name]) for name in sorted(left)]


def output_name(batch_index: int, frame_index: int, source: Path) -> str:
  """Create a flat, collision-free name shared by both output cameras."""
  return "b{:02d}_{:06d}{}".format(
    batch_index, frame_index, source.suffix.lower())


def ensure_empty_output(directory: Path):
  if directory.exists() and any(directory.iterdir()):
    raise MergeError(
      "output directory is not empty: {} (choose a new --output)".format(
        directory))


def merge_batches(
    root: Path,
    output: Path,
    batches: Iterable[tuple[str, str]],
    output_cameras: tuple[str, str],
    dry_run: bool = False,
) -> int:
  batches = list(batches)
  prepared = []
  total = 0
  for index, (left_name, right_name) in enumerate(batches):
    pairs = paired_images(root, left_name, right_name)
    prepared.append((index, left_name, right_name, pairs))
    total += len(pairs)

  if not dry_run:
    ensure_empty_output(output)
    for camera in output_cameras:
      (output / camera).mkdir(parents=True, exist_ok=True)

  for index, left_name, right_name, pairs in prepared:
    print("batch {:02d}: {} <-> {} ({} synchronized frames)".format(
      index, left_name, right_name, len(pairs)))
    if dry_run:
      continue
    for frame_index, (left_source, right_source) in enumerate(pairs):
      name = output_name(index, frame_index, left_source)
      shutil.copy2(left_source, output / output_cameras[0] / name)
      shutil.copy2(right_source, output / output_cameras[1] / name)

  action = "Would write" if dry_run else "Wrote"
  print("{} {} synchronized frames per camera to {}".format(
    action, total, output))
  return total


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Merge synchronized camera batches without changing sources.")
  parser.add_argument(
    "--root", type=Path, default=Path("20260804/extrinsic"),
    help="directory containing source camera folders")
  parser.add_argument(
    "--output", type=Path, default=Path("20260804/extrinsic_merged"),
    help="new merged dataset directory; must be empty")
  parser.add_argument(
    "--batch", nargs=2, action="append", metavar=("LEFT", "RIGHT"),
    help=("synchronized source camera folders; repeat for each batch "
          "(default: cam2 cam3, then cam2_e1 cam3_e2)"))
  parser.add_argument(
    "--output-cameras", nargs=2, default=("cam2", "cam3"),
    metavar=("LEFT", "RIGHT"), help="output camera folder names")
  parser.add_argument(
    "--dry-run", action="store_true",
    help="validate inputs and show counts without copying files")
  return parser


def main() -> int:
  args = make_parser().parse_args()
  batches = args.batch or list(DEFAULT_BATCHES)
  try:
    merge_batches(
      args.root, args.output, batches, tuple(args.output_cameras), args.dry_run)
  except (MergeError, OSError) as error:
    print("merge error: {}".format(error), file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
