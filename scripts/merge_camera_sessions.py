#!/usr/bin/env python3
"""Merge multi-camera capture sessions into one multical image dataset.

Expected input layout::

    source/
      session_1/cam0/0001.png
      session_1/cam1/0001.png
      session_2/cam0/0001.png
      session_2/cam1/0001.png

The output contains flat ``camN`` directories with a continuous six-digit
frame number. Files with the same relative path inside one session receive the
same frame number across cameras.

uv run python scripts/merge_camera_sessions.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff"}
CAMERA_PATTERN = re.compile(r"^ca?m(\d+)$", re.IGNORECASE)

# macOS configuration: edit these two paths if you prefer not to pass paths
# on the command line. Keep DEFAULT_OUTPUT_PATH empty to use SOURCE/merged.
DEFAULT_SOURCE_PATH = "data/pending"
DEFAULT_OUTPUT_PATH = "data/merged"

# Empty lists mean "all". Names must match directories under the source.
# Example: ["capture1", "capture3"] and ["cam0", "cam5"].
DEFAULT_SESSION_NAMES: list[str] = ["capture3", "capture4"]
DEFAULT_CAMERA_NAMES: list[str] = ["cam4", "cam5"]


class MergeError(RuntimeError):
  """Raised when a dataset cannot be merged safely."""


@dataclass(frozen=True)
class Session:
  name: str
  cameras: dict[int, Path]


@dataclass(frozen=True)
class CopyOperation:
  camera: int
  output_name: str
  session: str
  source_camera: str
  relative_path: str
  source: Path


def natural_key(value: str) -> tuple[object, ...]:
  """Return a case-insensitive key that sorts 2 before 10."""
  return tuple(
    int(part) if part.isdigit() else part.casefold()
    for part in re.split(r"(\d+)", value)
  )


def resolve_path(path: Path) -> Path:
  return path.expanduser().resolve()


def parse_camera_names(names: Iterable[str]) -> set[int]:
  cameras = set()
  invalid = []
  for name in names:
    match = CAMERA_PATTERN.fullmatch(name)
    if match is None:
      invalid.append(name)
    else:
      cameras.add(int(match.group(1)))
  if invalid:
    raise MergeError(
      "invalid camera names (expected camN or cmN): {}".format(
        ", ".join(invalid)
      )
    )
  return cameras


def discover_sessions(
    source: Path,
    output: Path,
    selected_sessions: Iterable[str] = (),
    selected_cameras: Iterable[str] = (),
) -> list[Session]:
  requested_sessions = set(selected_sessions)
  requested_cameras = parse_camera_names(selected_cameras)
  found_session_names = set()
  found_camera_numbers = set()
  sessions = []
  for directory in sorted(source.iterdir(), key=lambda path: natural_key(path.name)):
    if not directory.is_dir() or directory.resolve() == output:
      continue
    if requested_sessions and directory.name not in requested_sessions:
      continue
    found_session_names.add(directory.name)

    cameras = {}
    for candidate in directory.iterdir():
      if not candidate.is_dir():
        continue
      match = CAMERA_PATTERN.fullmatch(candidate.name)
      if match is None:
        continue
      camera = int(match.group(1))
      if requested_cameras and camera not in requested_cameras:
        continue
      if camera in cameras:
        raise MergeError(
          "session {!r} has duplicate camera {} directories: {} and {}".format(
            directory.name, camera, cameras[camera].name, candidate.name
          )
        )
      cameras[camera] = candidate
      found_camera_numbers.add(camera)

    if cameras:
      sessions.append(Session(directory.name, cameras))

  missing_sessions = requested_sessions - found_session_names
  if missing_sessions:
    raise MergeError(
      "selected session directories do not exist: {}".format(
        ", ".join(sorted(missing_sessions, key=natural_key))
      )
    )
  missing_cameras = requested_cameras - found_camera_numbers
  if missing_cameras:
    raise MergeError(
      "selected camera directories were not found: {}".format(
        ", ".join("cam{}".format(camera) for camera in sorted(missing_cameras))
      )
    )
  if not sessions:
    raise MergeError(
      "no matching session/camN directory structure found in {}".format(source)
    )
  return sessions


def files_in_camera(directory: Path, all_files: bool) -> dict[str, Path]:
  files = {}
  for path in directory.rglob("*"):
    if not path.is_file():
      continue
    if not all_files and path.suffix.lower() not in IMAGE_SUFFIXES:
      continue
    relative = path.relative_to(directory).as_posix()
    folded = relative.casefold()
    if folded in files:
      raise MergeError(
        "case-insensitive duplicate path in {}: {}".format(directory, relative)
      )
    files[folded] = path
  return files


def plan_merge(
    sessions: Iterable[Session],
    all_files: bool = False,
    require_complete: bool = False,
) -> tuple[list[CopyOperation], int]:
  operations = []
  next_frame = 0

  for session in sessions:
    files_by_camera = {
      camera: files_in_camera(directory, all_files)
      for camera, directory in session.cameras.items()
    }
    relative_names = {
      key: path.relative_to(session.cameras[camera]).as_posix()
      for camera, files in files_by_camera.items()
      for key, path in files.items()
    }
    frame_keys = set(relative_names)

    if require_complete and frame_keys:
      missing = []
      for key in sorted(frame_keys, key=lambda item: natural_key(relative_names[item])):
        absent = [
          "cam{}".format(camera)
          for camera, files in sorted(files_by_camera.items())
          if key not in files
        ]
        if absent:
          missing.append("{} missing {}".format(relative_names[key], ", ".join(absent)))
      if missing:
        raise MergeError(
          "session {!r} is not synchronized: {}".format(
            session.name, "; ".join(missing[:10])
          )
        )

    for key in sorted(frame_keys, key=lambda item: natural_key(relative_names[item])):
      output_stem = "{:06d}".format(next_frame)
      for camera, camera_files in sorted(files_by_camera.items()):
        source_file = camera_files.get(key)
        if source_file is None:
          continue
        operations.append(CopyOperation(
          camera=camera,
          output_name=output_stem + source_file.suffix.lower(),
          session=session.name,
          source_camera=session.cameras[camera].name,
          relative_path=source_file.relative_to(session.cameras[camera]).as_posix(),
          source=source_file,
        ))
      next_frame += 1

  if not operations:
    kind = "files" if all_files else "supported images"
    raise MergeError("camera directories contain no {}".format(kind))
  return operations, next_frame


def ensure_empty_output(output: Path) -> None:
  if output.exists() and not output.is_dir():
    raise MergeError("output path exists and is not a directory: {}".format(output))
  if output.exists() and next(output.iterdir(), None) is not None:
    raise MergeError("output directory is not empty: {}".format(output))


def execute_merge(output: Path, operations: list[CopyOperation]) -> Path:
  ensure_empty_output(output)
  output.mkdir(parents=True, exist_ok=True)
  manifest = output / "merge_manifest.csv"

  cameras = sorted({operation.camera for operation in operations})
  for camera in cameras:
    (output / "cam{}".format(camera)).mkdir()

  with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow([
      "output_camera", "output_file", "source_session",
      "source_camera_dir", "source_relative_path", "source_full_path",
    ])
    for index, operation in enumerate(operations, start=1):
      destination = output / "cam{}".format(operation.camera) / operation.output_name
      if destination.exists():
        raise MergeError("output file collision: {}".format(destination))
      shutil.copy2(operation.source, destination)
      writer.writerow([
        "cam{}".format(operation.camera), operation.output_name,
        operation.session, operation.source_camera, operation.relative_path,
        str(operation.source),
      ])
      if index % 100 == 0:
        print("Copied {} files...".format(index))
  return manifest


def camera_counts(operations: Iterable[CopyOperation]) -> dict[int, int]:
  counts = {}
  for operation in operations:
    counts[operation.camera] = counts.get(operation.camera, 0) + 1
  return counts


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Merge capture-session camera folders for multical."
  )
  parser.add_argument(
    "source", type=Path, nargs="?",
    help=("source folder containing one subfolder per capture session; "
          "when omitted, DEFAULT_SOURCE_PATH at the top of this script is used")
  )
  parser.add_argument(
    "--output", type=Path,
    help="output folder (default: SOURCE/merged); it must be empty"
  )
  parser.add_argument(
    "--sessions", nargs="+",
    help=("capture-session directory names to merge; when omitted, "
          "DEFAULT_SESSION_NAMES is used, and an empty default means all")
  )
  parser.add_argument(
    "--cameras", nargs="+",
    help=("camera directories to merge, for example cam0 cam5; when omitted, "
          "DEFAULT_CAMERA_NAMES is used, and an empty default means all")
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="validate and show the planned result without copying"
  )
  parser.add_argument(
    "--require-complete", action="store_true",
    help="fail if any camera is missing a relative image path"
  )
  parser.add_argument(
    "--all-files", action="store_true",
    help="copy every file type instead of only supported image files"
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = make_parser().parse_args(argv)

  try:
    configured_source = args.source or (
      Path(DEFAULT_SOURCE_PATH) if DEFAULT_SOURCE_PATH.strip() else None
    )
    if configured_source is None:
      raise MergeError(
        "no source path supplied; pass SOURCE on the command line or edit "
        "DEFAULT_SOURCE_PATH at the top of this script"
      )
    source = resolve_path(configured_source)
    configured_output = args.output or (
      Path(DEFAULT_OUTPUT_PATH) if DEFAULT_OUTPUT_PATH.strip() else None
    )
    output = resolve_path(configured_output) if configured_output else source / "merged"

    if not source.is_dir():
      raise MergeError("source directory does not exist: {}".format(source))
    if output == source:
      raise MergeError("output directory cannot be the source directory")
    selected_sessions = (
      args.sessions if args.sessions is not None else DEFAULT_SESSION_NAMES
    )
    selected_cameras = (
      args.cameras if args.cameras is not None else DEFAULT_CAMERA_NAMES
    )
    sessions = discover_sessions(
      source,
      output,
      selected_sessions=selected_sessions,
      selected_cameras=selected_cameras,
    )
    operations, frame_count = plan_merge(
      sessions, all_files=args.all_files,
      require_complete=args.require_complete,
    )
    ensure_empty_output(output)

    action = "Would copy" if args.dry_run else "Copied"
    manifest = None
    if not args.dry_run:
      manifest = execute_merge(output, operations)

    print("{} {} files in {} frames from {} sessions to {}".format(
      action, len(operations), frame_count, len(sessions), output
    ))
    for camera, count in sorted(camera_counts(operations).items()):
      print("  cam{}: {} files".format(camera, count))
    if manifest is not None:
      print("Manifest: {}".format(manifest))
  except (MergeError, OSError) as error:
    print("merge error: {}".format(error), file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
