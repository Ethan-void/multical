#!/usr/bin/env python3
"""Run a configurable Multical workflow with safe resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import yaml


MULTICAL_COMMANDS = {
  "boards", "intrinsic", "calibrate", "world", "worldmulti", "observe",
  "triangulate", "evaluate3d", "rectify", "vis"
}
INPUT_ARGUMENTS = {
  "image_path", "boards", "calibration", "correspondences",
  "world_extrinsics", "observations", "reconstruction", "ground_truth",
  "workspace", "workspace_file", "intrinsic", "extrinsic",
  "intrinsic_detections"
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}


class PipelineError(RuntimeError):
  pass


class KeepUnknown(dict):
  def __missing__(self, key):
    return "{" + key + "}"


def expand_string(value: str, variables: Mapping[str, Any]) -> str:
  previous = value
  for _ in range(10):
    current = previous.format_map(KeepUnknown(variables))
    if current == previous:
      return current
    previous = current
  return previous


def expand_value(value: Any, variables: Mapping[str, Any]) -> Any:
  if isinstance(value, str):
    return expand_string(value, variables)
  if isinstance(value, list):
    return [expand_value(item, variables) for item in value]
  if isinstance(value, dict):
    return {
      key: expand_value(item, variables) for key, item in value.items()
    }
  return value


def load_config(filename: Path) -> Dict[str, Any]:
  try:
    data = yaml.safe_load(filename.read_text(encoding="utf-8")) or {}
  except (OSError, yaml.YAMLError) as error:
    raise PipelineError("cannot load config {}: {}".format(filename, error))
  if not isinstance(data, dict):
    raise PipelineError("pipeline config must be a YAML mapping")

  raw_variables = dict(data.get("variables", {}))
  raw_variables.update({
    "config_dir": str(filename.parent.resolve()),
    "repo": str(Path.cwd().resolve())
  })
  variables: Dict[str, Any] = {}
  for _ in range(10):
    variables = {
      key: expand_value(value, {**raw_variables, **variables})
      for key, value in raw_variables.items()
    }
  expanded = expand_value(data, variables)
  expanded["variables"] = variables
  return expanded


def stage_items(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
  stages = config.get("stages")
  if not isinstance(stages, dict) or not stages:
    raise PipelineError("config must define at least one stage under 'stages'")
  result = []
  for name, raw in stages.items():
    if not isinstance(raw, dict):
      raise PipelineError("stage {!r} must be a mapping".format(name))
    stage = dict(raw)
    stage["name"] = name
    stage.setdefault("enabled", True)
    stage.setdefault("group", stage.get("command", name))
    args = stage.get("args", {})
    if not isinstance(args, dict):
      raise PipelineError("stage {!r} args must be a mapping".format(name))
    stage["args"] = dict(args)
    result.append(stage)
  return result


def parse_requested(values: Iterable[str]) -> List[str]:
  result = []
  for value in values:
    result.extend(item.strip() for item in value.split(",") if item.strip())
  return result or ["all"]


def select_stages(
    stages: List[Dict[str, Any]], requested: List[str], camera: str | None,
    from_stage: str | None, to_stage: str | None) -> List[Dict[str, Any]]:
  enabled = [stage for stage in stages if stage["enabled"]]
  if requested == ["all"] or "all" in requested:
    selected = enabled
  else:
    selected = [
      stage for stage in stages
      if stage["name"] in requested
      or stage["group"] in requested
      or stage.get("command") in requested
    ]
    missing = [
      item for item in requested
      if not any(
        item in (stage["name"], stage["group"], stage.get("command"))
        for stage in stages
      )
    ]
    if missing:
      raise PipelineError("unknown stage(s): {}".format(
        ", ".join(missing)))

  names = [stage["name"] for stage in selected]
  if from_stage:
    if from_stage not in names:
      raise PipelineError("--from-stage {} is not selected".format(from_stage))
    selected = selected[names.index(from_stage):]
  names = [stage["name"] for stage in selected]
  if to_stage:
    if to_stage not in names:
      raise PipelineError("--to-stage {} is not selected".format(to_stage))
    selected = selected[:names.index(to_stage) + 1]

  if camera:
    camera_selected = []
    for stage in selected:
      if stage["group"] != "validation":
        continue
      cameras = stage["args"].get("cameras", [])
      if camera in cameras or camera in stage["name"]:
        copied = dict(stage)
        copied["args"] = dict(stage["args"])
        copied["args"]["cameras"] = [camera]
        copied["args"]["master"] = camera
        camera_selected.append(copied)
    if not camera_selected:
      raise PipelineError(
        "no selected validation stage contains camera {}".format(camera))
    selected = camera_selected
  return selected


def include_dependencies(
    all_stages: List[Dict[str, Any]],
    selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  by_name = {stage["name"]: stage for stage in all_stages}
  explicitly_selected = {stage["name"] for stage in selected}
  required = set()
  visiting = set()

  def add(name: str):
    if name in required:
      return
    if name in visiting:
      raise PipelineError("stage dependency cycle includes {}".format(name))
    stage = by_name.get(name)
    if stage is None:
      raise PipelineError("unknown dependency stage {}".format(name))
    if not stage["enabled"] and name not in explicitly_selected:
      raise PipelineError("required dependency stage {} is disabled".format(name))
    visiting.add(name)
    needs = stage.get("needs", [])
    if isinstance(needs, str):
      needs = [needs]
    for dependency in needs:
      add(str(dependency))
    visiting.remove(name)
    required.add(name)

  for stage in selected:
    add(stage["name"])
  selected_versions = {stage["name"]: stage for stage in selected}
  return [
    selected_versions.get(stage["name"], stage)
    for stage in all_stages if stage["name"] in required
  ]


def cli_arguments(values: Mapping[str, Any]) -> List[str]:
  result: List[str] = []
  for key, value in values.items():
    if value is None:
      continue
    option = key if str(key).startswith("-") else "--" + str(key)
    if isinstance(value, bool):
      if value:
        result.append(option)
      else:
        result.extend([option, "false"])
    elif isinstance(value, (list, tuple)):
      if value:
        result.append(option)
        result.extend(str(item) for item in value)
    else:
      result.extend([option, str(value)])
  return result


def command_for(stage: Mapping[str, Any]) -> List[str]:
  command = stage.get("command")
  if command in MULTICAL_COMMANDS:
    prefix = [sys.executable, "-m", "multical.app.multical", command]
  elif command == "analyze":
    prefix = [sys.executable, "scripts/analyze_calibration.py"]
  elif command == "python":
    script = stage.get("script")
    if not script:
      raise PipelineError("python stage {!r} needs 'script'".format(
        stage["name"]))
    prefix = [sys.executable, str(script)]
  else:
    raise PipelineError("stage {!r} has unsupported command {!r}".format(
      stage["name"], command))
  return prefix + cli_arguments(stage["args"]) + [
    str(value) for value in stage.get("extra_args", [])
  ]


def output_paths(stage: Mapping[str, Any]) -> List[Path]:
  explicit = stage.get("outputs")
  if explicit is not None:
    if not isinstance(explicit, list):
      explicit = [explicit]
    return [Path(value) for value in explicit]

  args = stage["args"]
  command = stage.get("command")
  if command == "intrinsic":
    folder = Path(args.get("output_path") or args.get("image_path", "."))
    return [folder / (str(args.get("name", "intrinsic")) + ".json")]
  if command == "calibrate":
    folder = Path(args.get("output_path") or args.get("image_path", "."))
    name = str(args.get("name", "calibration"))
    return [folder / (name + ".json"), folder / (name + ".pkl")]
  if command in ("world", "worldmulti"):
    if args.get("output"):
      return [Path(args["output"])]
    suffix = "world_extrinsics_multicam.json" if command == "worldmulti" else "world_extrinsics.json"
    return [Path(args["calibration"]).parent / suffix]
  if command == "observe":
    return [Path(args["output"])]
  if command == "triangulate":
    destination = Path(args.get("output") or (
      Path(args["world_extrinsics"]).parent / "triangulation.json"))
    return [destination]
  if command == "evaluate3d":
    destination = Path(args.get("output") or (
      Path(args["reconstruction"]).parent / "evaluation3d.json"))
    return [destination, destination.with_suffix(".xlsx")]
  if command == "analyze":
    return [Path(args["output"])]
  return []


def ensure_output_directories(stage: Mapping[str, Any], outputs: List[Path]):
  args = stage["args"]
  if args.get("output_path"):
    Path(args["output_path"]).mkdir(parents=True, exist_ok=True)
  for output in outputs:
    output.parent.mkdir(parents=True, exist_ok=True)


def subset_calibration(stage: MutableMapping[str, Any], dry_run: bool):
  if stage.get("command") != "calibrate":
    return
  args = stage["args"]
  cameras = args.get("cameras")
  calibration_name = args.get("calibration")
  if not cameras or not calibration_name:
    return
  calibration_path = Path(calibration_name)
  if not calibration_path.is_file():
    return
  try:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise PipelineError("cannot read calibration {}: {}".format(
      calibration_path, error))
  available = calibration.get("cameras", {})
  requested = list(cameras)
  if set(available) == set(requested):
    return
  missing = sorted(set(requested) - set(available))
  if missing:
    raise PipelineError("calibration {} is missing camera(s): {}".format(
      calibration_path, ", ".join(missing)))
  if "camera_poses" in calibration:
    raise PipelineError(
      "automatic camera subsetting only supports intrinsic-only JSON files; "
      "stage {!r} calibration contains camera_poses".format(stage["name"]))

  output_folder = Path(
    args.get("output_path") or args.get("image_path", "."))
  subset_path = output_folder / ".pipeline_inputs" / (
    "{}_{}.json".format(stage["name"], "_".join(requested)))
  subset = {
    "cameras": {name: available[name] for name in requested}
  }
  if not dry_run:
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(subset, indent=2) + "\n"
    existing = (
      subset_path.read_text(encoding="utf-8")
      if subset_path.is_file() else None
    )
    # Keep mtime stable so --resume can recognize an unchanged generated input.
    if existing != content:
      subset_path.write_text(content, encoding="utf-8")
  args["calibration"] = str(subset_path)
  print("  auto calibration subset: {}".format(subset_path))


def path_signature(path_value: str) -> Dict[str, Any]:
  path = Path(path_value)
  result: Dict[str, Any] = {"path": str(path)}
  try:
    stat = path.stat()
  except OSError:
    result["missing"] = True
    return result
  result.update({
    "size": stat.st_size,
    "mtime_ns": stat.st_mtime_ns,
    "directory": path.is_dir()
  })
  return result


def image_dataset_signature(
    path_value: str, args: Mapping[str, Any]) -> Dict[str, Any]:
  """Fingerprint image files, excluding outputs stored beside the dataset."""
  base = Path(path_value)
  result: Dict[str, Any] = {"path": str(base), "image_dataset": True}
  if not base.is_dir():
    result["missing"] = True
    return result

  cameras = args.get("cameras") or []
  if isinstance(cameras, str):
    cameras = [cameras]
  camera_pattern = args.get("camera_pattern") or "{camera}"
  if cameras:
    camera_directories = [
      base / camera_pattern.format(camera=camera) for camera in cameras
    ]
  else:
    camera_directories = sorted(
      path for path in base.iterdir() if path.is_dir()
    )

  images = []
  missing_cameras = []
  for camera_directory in camera_directories:
    if not camera_directory.is_dir():
      missing_cameras.append(str(camera_directory))
      continue
    for image in sorted(camera_directory.iterdir()):
      if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
        continue
      stat = image.stat()
      images.append({
        "path": str(image.relative_to(base)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns
      })
  result["images"] = images
  if missing_cameras:
    result["missing_cameras"] = missing_cameras
  return result


def missing_inputs(stage: Mapping[str, Any]) -> List[str]:
  missing = []
  for key, value in stage["args"].items():
    if key not in INPUT_ARGUMENTS or value is None:
      continue
    values = value if isinstance(value, list) else [value]
    for item in values:
      path = Path(str(item))
      if not path.exists():
        missing.append("{}={}".format(key, path))
  return missing


def fingerprint(stage: Mapping[str, Any], command: List[str]) -> str:
  inputs = []
  for key, value in stage["args"].items():
    if key not in INPUT_ARGUMENTS or value is None:
      continue
    values = value if isinstance(value, list) else [value]
    if key == "image_path":
      inputs.extend(
        image_dataset_signature(str(item), stage["args"])
        for item in values
      )
    else:
      inputs.extend(path_signature(str(item)) for item in values)
  payload = json.dumps({
    "command": command,
    "inputs": inputs
  }, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state(filename: Path) -> Dict[str, Any]:
  if not filename.is_file():
    return {"version": 1, "stages": {}}
  try:
    state = json.loads(filename.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise PipelineError("cannot read state file {}: {}".format(filename, error))
  state.setdefault("version", 1)
  state.setdefault("stages", {})
  return state


def save_state(filename: Path, state: Mapping[str, Any]):
  filename.parent.mkdir(parents=True, exist_ok=True)
  temporary = filename.with_suffix(filename.suffix + ".tmp")
  temporary.write_text(
    json.dumps(state, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8")
  os.replace(temporary, filename)


def can_resume(
    record: Mapping[str, Any] | None, digest: str, outputs: List[Path]) -> bool:
  return bool(
    record
    and record.get("status") == "success"
    and record.get("fingerprint") == digest
    and outputs
    and all(output.exists() for output in outputs)
  )


def display_command(command: List[str]) -> str:
  visible = list(command)
  if visible[:3] == [sys.executable, "-m", "multical.app.multical"]:
    visible = ["multical"] + visible[3:]
  elif visible and visible[0] == sys.executable:
    visible = ["python"] + visible[1:]
  return shlex.join(visible)


def run_pipeline(options) -> int:
  config_path = Path(options.config).resolve()
  config = load_config(config_path)
  stages = stage_items(config)

  if options.list_stages:
    for stage in stages:
      status = "enabled" if stage["enabled"] else "disabled"
      print("{:<24} {:<12} {}".format(
        stage["name"], stage["group"], status))
    return 0

  requested = parse_requested(options.stage)
  selected = select_stages(
    stages, requested, options.camera, options.from_stage, options.to_stage)
  if options.with_deps:
    selected = include_dependencies(stages, selected)
  if not selected:
    raise PipelineError("no stages selected")

  state_path = Path(config.get(
    "state_file", config_path.with_suffix(".state.json")))
  state = load_state(state_path)
  environment = os.environ.copy()
  environment.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
  environment.update({
    str(key): str(value) for key, value in config.get("env", {}).items()
  })

  print("Pipeline config: {}".format(config_path))
  print("State file:     {}".format(state_path))
  print("Stages:         {}".format(", ".join(
    stage["name"] for stage in selected)))

  for index, original in enumerate(selected, start=1):
    stage = dict(original)
    stage["args"] = dict(original["args"])
    print("\n[{}/{}] {} ({})".format(
      index, len(selected), stage["name"], stage["command"]))
    subset_calibration(stage, options.dry_run)
    outputs = output_paths(stage)
    command = command_for(stage)
    digest = fingerprint(stage, command)
    record = state["stages"].get(stage["name"])
    print("  command: {}".format(display_command(command)))
    if outputs:
      print("  outputs: {}".format(", ".join(str(path) for path in outputs)))

    if (
        options.resume
        and not options.force
        and not stage.get("interactive", False)
        and can_resume(record, digest, outputs)
    ):
      print("  status: skipped (successful output is up to date)")
      continue
    if options.dry_run:
      print("  status: dry-run")
      continue

    unavailable = missing_inputs(stage)
    if unavailable:
      raise PipelineError("stage {!r} input(s) do not exist: {}".format(
        stage["name"], ", ".join(unavailable)))
    ensure_output_directories(stage, outputs)
    started = time.time()
    state["stages"][stage["name"]] = {
      "status": "running",
      "fingerprint": digest,
      "command": display_command(command),
      "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "outputs": [str(path) for path in outputs]
    }
    save_state(state_path, state)
    result = subprocess.run(command, env=environment)
    duration = time.time() - started
    if result.returncode != 0:
      state["stages"][stage["name"]].update({
        "status": "failed",
        "returncode": result.returncode,
        "duration_seconds": round(duration, 3)
      })
      save_state(state_path, state)
      raise PipelineError("stage {!r} failed with exit code {}".format(
        stage["name"], result.returncode))
    missing_outputs = [str(path) for path in outputs if not path.exists()]
    if missing_outputs:
      state["stages"][stage["name"]].update({
        "status": "failed",
        "duration_seconds": round(duration, 3),
        "missing_outputs": missing_outputs
      })
      save_state(state_path, state)
      raise PipelineError("stage {!r} finished but outputs are missing: {}".format(
        stage["name"], ", ".join(missing_outputs)))
    state["stages"][stage["name"]].update({
      "status": "success",
      "returncode": 0,
      "duration_seconds": round(duration, 3),
      "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    })
    save_state(state_path, state)
    print("  status: success ({:.1f}s)".format(duration))

  print("\nPipeline completed successfully.")
  return 0


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run Multical stages from one YAML configuration.")
  parser.add_argument("--config", required=True, help="pipeline YAML file")
  parser.add_argument(
    "--stage", action="append", default=[],
    help="stage name/group/command, comma separated; default: all")
  parser.add_argument("--camera", help="select one camera validation stage")
  parser.add_argument(
    "--with-deps", action="store_true",
    help="also run prerequisite stages; default: selected stages only")
  parser.add_argument("--from-stage", help="start at this selected stage")
  parser.add_argument("--to-stage", help="stop after this selected stage")
  parser.add_argument(
    "--resume", action="store_true",
    help="skip successful stages when command, inputs and outputs match")
  parser.add_argument(
    "--force", action="store_true",
    help="run selected stages even if --resume could skip them")
  parser.add_argument(
    "--dry-run", action="store_true",
    help="print commands without creating files or running programs")
  parser.add_argument(
    "--list-stages", action="store_true", help="list configured stages")
  return parser


def main() -> int:
  try:
    return run_pipeline(make_parser().parse_args())
  except PipelineError as error:
    print("pipeline error: {}".format(error), file=sys.stderr)
    return 2
  except KeyboardInterrupt:
    print("\npipeline interrupted", file=sys.stderr)
    return 130


if __name__ == "__main__":
  raise SystemExit(main())
