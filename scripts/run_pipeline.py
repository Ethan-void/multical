#!/usr/bin/env python3
"""Run a configurable Multical workflow with safe resume support."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import yaml


MULTICAL_COMMANDS = {
  "boards", "intrinsic", "calibrate", "world", "worldmulti", "observe",
  "worldgroups", "worldgroupba", "triangulate", "evaluate3d", "rectify",
  "vis", "worldpoints"
}
INPUT_ARGUMENTS = {
  "image_path", "boards", "calibration", "correspondences",
  "world_extrinsics", "observations", "observe", "reconstruction",
  "ground_truth",
  "workspace", "workspace_file", "intrinsic", "extrinsic",
  "intrinsic_detections", "calibrations", "evaluation", "workspaces",
  "initial_world_extrinsics", "world_points", "world_correspondences",
  "world_board", "marker_board"
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INIT_TEMPLATE = (
  REPO_ROOT / "configs" /
  "pipeline.default.yaml"
)
EXPERIMENT_DIRECTORIES = (
  "intrinsic",
  "extrinsic",
  "world",
  "world/world_images",
  "observe",
  "observe/measured_points",
)
DEFAULT_BOARDS = "boards/charuco_1600x1200.yaml"
DEFAULT_WORLD_BOARD = "boards/world_boards/标定板图案设计.yaml"


class FlowList(list):
  """A YAML sequence that should be rendered in compact flow style."""


class CameraList(FlowList):
  """The shared camera sequence rendered with the stable cameras anchor."""


class PipelineDumper(yaml.SafeDumper):
  def generate_anchor(self, node):
    if getattr(node, "camera_anchor", False):
      return "cameras"
    return super().generate_anchor(node)


def _represent_flow_list(dumper, values):
  node = dumper.represent_sequence(
    "tag:yaml.org,2002:seq", values, flow_style=True)
  if isinstance(values, CameraList):
    node.camera_anchor = True
  return node


PipelineDumper.add_representer(FlowList, _represent_flow_list)
PipelineDumper.add_representer(CameraList, _represent_flow_list)


def format_pipeline_yaml(text: str, stages: Mapping[str, Any]) -> str:
  """Add visual separation before stages and between stage groups."""
  lines = text.splitlines()
  result = []
  in_stages = False
  previous_group = None
  for line in lines:
    if line == "stages:":
      if result and result[-1] != "":
        result.append("")
      result.append(line)
      in_stages = True
      continue
    if in_stages:
      match = re.fullmatch(r"  ([^ ].*):", line)
      if match and match.group(1) in stages:
        stage_group = stages[match.group(1)].get("group")
        if previous_group is not None and stage_group != previous_group:
          if result and result[-1] != "":
            result.append("")
        previous_group = stage_group
    result.append(line)
  return "\n".join(result) + "\n"


class PipelineError(RuntimeError):
  pass


class KeepUnknown(dict):
  def __missing__(self, key):
    return "{" + key + "}"


def parse_camera_groups(
    cameras: List[str], specifications: List[str]
) -> List[tuple[str, List[str]]]:
  """Parse NAME=cam0,cam1 or cam0,cam1 group specifications."""
  if not cameras:
    raise PipelineError("--cameras must contain at least one camera")
  if len(set(cameras)) != len(cameras):
    raise PipelineError("--cameras contains duplicate names")
  if not specifications:
    raise PipelineError("at least one --group is required")

  result: List[tuple[str, List[str]]] = []
  assigned = set()
  labels = set()
  for specification in specifications:
    explicit_label, separator, members_text = specification.partition("=")
    if not separator:
      members_text = explicit_label
      explicit_label = ""
    members = [item.strip() for item in members_text.split(",") if item.strip()]
    if len(members) < 2:
      raise PipelineError(
        "camera group must contain at least two cameras: {}".format(
          specification))
    unknown = [camera for camera in members if camera not in cameras]
    if unknown:
      raise PipelineError("group contains unknown camera(s): {}".format(
        ", ".join(unknown)))
    repeated = [camera for camera in members if camera in assigned]
    if repeated:
      raise PipelineError("camera appears in multiple groups: {}".format(
        ", ".join(repeated)))

    if explicit_label:
      raw_label = explicit_label.strip()
    elif all(re.fullmatch(r"cam[0-9A-Za-z]+", camera) for camera in members):
      raw_label = "".join(camera[3:] for camera in members)
    else:
      raw_label = "_".join(members)
    label = re.sub(r"[^0-9A-Za-z_-]+", "_", raw_label).strip("_")
    if not label or label in labels:
      raise PipelineError("camera groups must have unique non-empty labels")
    labels.add(label)
    assigned.update(members)
    result.append((label, members))

  missing = [camera for camera in cameras if camera not in assigned]
  if missing:
    raise PipelineError("camera(s) missing from --group: {}".format(
      ", ".join(missing)))
  return result


def render_experiment_config(
    template_text: str, dataset: Path, output_root: Path, boards: str,
    cameras: List[str], groups: List[tuple[str, List[str]]],
    auto_analyze: bool = True,
    world_board: str = DEFAULT_WORLD_BOARD,
    court: str = "tennis") -> str:
  if court not in ("tennis", "badminton"):
    raise PipelineError("court must be tennis or badminton")
  try:
    data = yaml.safe_load(template_text) or {}
  except yaml.YAMLError as error:
    raise PipelineError("cannot parse pipeline template: {}".format(error))
  if not isinstance(data, dict) or not isinstance(data.get("stages"), dict):
    raise PipelineError("pipeline template must define a stages mapping")

  variables = data.setdefault("variables", {})
  variables["dataset"] = str(dataset)
  variables["output_root"] = str(output_root)
  variables["boards"] = boards
  variables["world_board"] = world_board
  variables["court"] = court
  shared_cameras = CameraList(cameras)
  variables["cameras"] = shared_cameras
  data["state_file"] = "{output_root}/pipeline_state.json"

  old_stages = data["stages"]
  try:
    extrinsic_template = next(
      stage for name, stage in old_stages.items()
      if name.startswith("extrinsic_"))
    world_template = next(
      stage for name, stage in old_stages.items()
      if name.startswith("world_") and name != "worldgroups")
  except StopIteration:
    raise PipelineError(
      "pipeline template needs at least one extrinsic_* and world_* stage")
  analysis_template = next((
    stage for name, stage in old_stages.items()
    if name.startswith("analyze_extrinsic_")), None)
  intrinsic_analysis = copy.deepcopy(old_stages.get("analyze_intrinsic") or {
    "command": "analyze",
    "group": "intrinsic_analysis",
    "args": {
      "intrinsic": "{output_root}/intrinsic/intrinsic.json",
      "output": "{output_root}/reports/intrinsic_analysis.xlsx",
    },
  })
  intrinsic_analysis["enabled"] = False
  intrinsic_analysis["needs"] = ["intrinsic"]
  if auto_analyze:
    old_stages["intrinsic"]["then"] = ["analyze_intrinsic"]
  else:
    old_stages["intrinsic"].pop("then", None)

  generated_extrinsics = {}
  generated_worlds = {}
  generated_analyses = {}
  generated_worldpoints = {}
  for label, members in groups:
    extrinsic = copy.deepcopy(extrinsic_template)
    extrinsic.pop("then", None)
    extrinsic["needs"] = ["intrinsic"]
    extrinsic["args"].update({
      "cameras": FlowList(members),
      "master": members[0],
      "output_path": "{{output_root}}/extrinsic_{}".format(label),
      "calibration": "{output_root}/intrinsic/intrinsic.json",
    })
    generated_extrinsics["extrinsic_{}".format(label)] = extrinsic

    world = copy.deepcopy(world_template)
    world["needs"] = ["extrinsic_{}".format(label)]
    world["args"].update({
      "calibration": "{{output_root}}/extrinsic_{}/calibration.json".format(
        label),
      "correspondences": "{{dataset}}/world/world_markers_{}.yaml".format(
        label),
      "output": "{{output_root}}/world/group{}.json".format(label),
    })
    generated_worlds["world_{}".format(label)] = world

    generated_worldpoints["worldpoints_marker_{}".format(label)] = {
      "enabled": False,
      "interactive": True,
      "command": "worldpoints",
      "group": "worldpoints_marker",
      "args": {
        "mode": "markers",
        "court": "{court}",
        "image_path": "{dataset}/world/world_images",
        "world_board": "{world_board}",
        "output": "{{dataset}}/world/world_markers_{}.yaml".format(label),
      },
    }

    if analysis_template is not None:
      analysis = copy.deepcopy(analysis_template)
      analysis["enabled"] = False
      analysis["needs"] = ["extrinsic_{}".format(label)]
      analysis["args"].update({
        "extrinsic": "{{output_root}}/extrinsic_{}/calibration.json".format(
          label),
        "workspace": "{{output_root}}/extrinsic_{}/calibration.pkl".format(
          label),
        "output": "{{output_root}}/reports/extrinsic_analysis_{}.xlsx".format(
          label),
      })
      generated_analyses["analyze_extrinsic_{}".format(label)] = analysis
      if auto_analyze:
        extrinsic["then"] = ["analyze_extrinsic_{}".format(label)]

  group_labels = [label for label, _ in groups]
  worldgroups = old_stages.get("worldgroups")
  worldgroupba = old_stages.get("worldgroupba")
  if not isinstance(worldgroups, dict) or not isinstance(worldgroupba, dict):
    raise PipelineError("pipeline template needs worldgroups and worldgroupba")
  worldgroups["needs"] = ["world_{}".format(label) for label in group_labels]
  worldgroups["args"].update({
    "calibrations": [
      "{{output_root}}/extrinsic_{}/calibration.json".format(label)
      for label in group_labels
    ],
    "world_extrinsics": [
      "{{output_root}}/world/group{}.json".format(label)
      for label in group_labels
    ],
    "group_names": ["group{}".format(label) for label in group_labels],
    "master": cameras[0],
  })
  worldgroupba["args"].update({
    "calibrations": [
      "{{output_root}}/extrinsic_{}/calibration.json".format(label)
      for label in group_labels
    ],
    "workspaces": [
      "{{output_root}}/extrinsic_{}/calibration.pkl".format(label)
      for label in group_labels
    ],
    "correspondences": [
      "{{dataset}}/world/world_markers_{}.yaml".format(label)
      for label in group_labels
    ],
    "group_names": ["group{}".format(label) for label in group_labels],
    "master": cameras[0],
    "relative_prior_weights": [2.0] * len(groups),
  })
  old_stages["intrinsic"]["args"]["cameras"] = shared_cameras
  if isinstance(old_stages.get("observe"), dict):
    old_stages["observe"]["args"]["cameras"] = shared_cameras

  new_stages = {}
  inserted = set()
  for name, stage in old_stages.items():
    if (
        name.startswith("worldpoints_marker_")
        or name in ("worldpoints_measured", "worldpoints_measured_single")
    ):
      continue
    if name == "analyze_intrinsic":
      continue
    if name == "intrinsic":
      new_stages[name] = stage
      new_stages["analyze_intrinsic"] = intrinsic_analysis
      continue
    if name.startswith("extrinsic_"):
      if "extrinsic" not in inserted:
        for generated_name, generated_stage in generated_extrinsics.items():
          new_stages[generated_name] = generated_stage
          label = generated_name.removeprefix("extrinsic_")
          analysis_name = "analyze_extrinsic_{}".format(label)
          if analysis_name in generated_analyses:
            new_stages[analysis_name] = generated_analyses[analysis_name]
        inserted.add("extrinsic")
      continue
    if name.startswith("world_"):
      if "world" not in inserted:
        new_stages.update(generated_worlds)
        inserted.add("world")
      continue
    if name.startswith("analyze_extrinsic_"):
      continue
    new_stages[name] = stage
  new_stages.update(generated_worldpoints)
  new_stages["worldpoints_measured"] = {
    "enabled": False,
    "interactive": True,
    "command": "worldpoints",
    "group": "worldpoints_measured",
    "args": {
      "mode": "measured",
      "court": "{court}",
      "observe": "{dataset}/observe/measured_observations.yaml",
      "output": "{dataset}/measured_world_points.yaml",
    },
  }
  data["stages"] = new_stages
  rendered = yaml.dump(
    data, Dumper=PipelineDumper, sort_keys=False, allow_unicode=True)
  return format_pipeline_yaml(rendered, new_stages)


def update_default_pipeline_config(config_path: Path) -> None:
  """Point the repository's convenience script at the initialized config."""
  repository_configs = (REPO_ROOT / "configs").resolve()
  if config_path.resolve().parent != repository_configs:
    # Tests and API callers may intentionally generate configs elsewhere;
    # those must never modify the real repository launcher.
    return
  launcher = REPO_ROOT / "pipeline"
  if not launcher.is_file():
    return
  try:
    default_path = config_path.resolve().relative_to(REPO_ROOT.resolve())
  except ValueError:
    default_path = config_path.resolve()
  replacement = "CONFIG=${MULTICAL_PIPELINE_CONFIG:-" + (
    shlex.quote(str(default_path)) + "}")
  try:
    original = launcher.read_text(encoding="utf-8")
  except OSError as error:
    raise PipelineError("cannot read pipeline launcher {}: {}".format(
      launcher, error))
  updated, count = re.subn(
    r"^CONFIG=\$\{MULTICAL_PIPELINE_CONFIG:-.*\}$",
    replacement,
    original,
    count=1,
    flags=re.MULTILINE,
  )
  if count != 1:
    raise PipelineError(
      "cannot find default CONFIG setting in {}".format(launcher))
  if updated != original:
    try:
      launcher.write_text(updated, encoding="utf-8")
    except OSError as error:
      raise PipelineError("cannot update pipeline launcher {}: {}".format(
        launcher, error))


def initialize_experiment(
    dataset: Path, template: Path | None = None,
    force: bool = False, config_dir: Path | None = None,
    cameras: List[str] | None = None,
    groups: List[tuple[str, List[str]]] | None = None,
    boards: str = DEFAULT_BOARDS,
    world_board: str = DEFAULT_WORLD_BOARD,
    output_root: Path | None = None,
    auto_analyze: bool = True,
    court: str = "tennis") -> Path:
  """Create a dataset tree and a pipeline config rendered for that tree."""
  dataset = dataset.expanduser().resolve()
  output_root = (output_root or dataset).expanduser().resolve()
  template = (template or DEFAULT_INIT_TEMPLATE).expanduser().resolve()
  config_dir = (config_dir or (REPO_ROOT / "configs")).expanduser().resolve()
  if not template.is_file():
    raise PipelineError("pipeline template does not exist: {}".format(template))
  if dataset.exists() and not dataset.is_dir():
    raise PipelineError("dataset path is not a directory: {}".format(dataset))

  config_path = config_dir / "pipeline.worldgroups.{}.yaml".format(dataset.name)
  dataset.mkdir(parents=True, exist_ok=True)
  for relative_path in EXPERIMENT_DIRECTORIES:
    (dataset / relative_path).mkdir(parents=True, exist_ok=True)

  if config_path.exists() and not force:
    update_default_pipeline_config(config_path)
    print("Experiment directories ready: {}".format(dataset))
    print("Pipeline config kept:       {}".format(config_path))
    print("Default pipeline config:    {}".format(config_path))
    return config_path

  config_dir.mkdir(parents=True, exist_ok=True)

  try:
    template_text = template.read_text(encoding="utf-8")
  except OSError as error:
    raise PipelineError("cannot read template {}: {}".format(template, error))
  if cameras is not None and groups is not None:
    rendered = render_experiment_config(
      template_text, dataset, output_root, boards, cameras, groups,
      auto_analyze=auto_analyze, world_board=world_board, court=court)
  else:
    if "DATASET" not in template_text:
      raise PipelineError(
        "pipeline template must contain the DATASET placeholder: {}".format(
          template))
    # JSON strings are valid YAML scalars and safely preserve spaces or colons.
    rendered = template_text.replace("DATASET", json.dumps(str(dataset)))
  try:
    config_path.write_text(rendered, encoding="utf-8")
  except OSError as error:
    raise PipelineError("cannot write config {}: {}".format(
      config_path, error))

  update_default_pipeline_config(config_path)
  print("Experiment initialized: {}".format(dataset))
  print("Pipeline config:       {}".format(config_path))
  print("Default pipeline config updated.")
  print("Run with: ./pipeline stage intrinsic")
  return config_path


def initialize_from_settings(
    filename: Path, force: bool = False,
    template_override: Path | None = None) -> Path:
  filename = filename.expanduser().resolve()
  try:
    settings = yaml.safe_load(filename.read_text(encoding="utf-8")) or {}
  except (OSError, yaml.YAMLError) as error:
    raise PipelineError("cannot load init settings {}: {}".format(
      filename, error))
  if not isinstance(settings, dict):
    raise PipelineError("init settings must be a YAML mapping")
  if not settings.get("dataset"):
    raise PipelineError("init settings must define dataset")

  raw_groups = settings.get("groups")
  if not isinstance(raw_groups, dict) or not raw_groups:
    raise PipelineError("init settings groups must be a non-empty mapping")
  specifications = []
  cameras = []
  for label, members in raw_groups.items():
    if not isinstance(members, list):
      raise PipelineError("init group {!r} must be a camera list".format(label))
    for member in members:
      camera = str(member)
      if camera not in cameras:
        cameras.append(camera)
    specifications.append("{}={}".format(
      label, ",".join(str(member) for member in members)))
  # The group order defines the camera order; a separate cameras list would
  # duplicate the same source of truth.
  groups = parse_camera_groups(cameras, specifications)

  template_value = template_override or settings.get("template")
  template = Path(str(template_value)) if template_value else None
  court = settings.get("court", "tennis")
  if court not in ("tennis", "badminton"):
    raise PipelineError("init settings court must be tennis or badminton")
  dataset = Path(str(settings["dataset"]))
  dataset_stem = re.sub(r"[._-](?:tennis|badminton)$", "", dataset.name)
  dataset = dataset.with_name("{}.{}".format(dataset_stem, court))
  output_value = settings.get("output_root")
  output_root = Path(str(output_value)) if output_value else dataset
  auto_analyze = settings.get("auto_analyze", True)
  if not isinstance(auto_analyze, bool):
    raise PipelineError("init settings auto_analyze must be true or false")
  return initialize_experiment(
    dataset,
    court=court,
    template=template,
    force=force,
    cameras=cameras,
    groups=groups,
    boards=str(settings.get("boards", DEFAULT_BOARDS)),
    world_board=str(settings.get("world_board", DEFAULT_WORLD_BOARD)),
    output_root=output_root,
    auto_analyze=auto_analyze,
  )


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


def resolve_config_path(value: str | Path) -> Path:
  """Resolve a path or a unique filename keyword under configs/."""
  requested = Path(value).expanduser()
  if requested.is_file():
    return requested.resolve()

  keyword = str(value)
  config_dir = REPO_ROOT / "configs"
  matches = sorted(
    path.resolve() for path in config_dir.glob("*.yaml")
    if keyword in path.name
  )
  if len(matches) == 1:
    print("Config keyword {!r} matched: {}".format(keyword, matches[0]))
    return matches[0]
  if not matches:
    raise PipelineError(
      "pipeline config does not exist and no configs/*.yaml file matches "
      "{!r}".format(keyword))

  choices = "\n".join("  {}".format(
    path.relative_to(REPO_ROOT)) for path in matches)
  raise PipelineError(
    "multiple pipeline configs match {!r}; choose a more specific keyword "
    "or full path:\n{}".format(keyword, choices))


def set_default_pipeline_config(value: str | Path) -> Path:
  config_path = resolve_config_path(value)
  if config_path.parent != (REPO_ROOT / "configs").resolve():
    raise PipelineError(
      "default pipeline config must be located under {}".format(
        REPO_ROOT / "configs"))
  update_default_pipeline_config(config_path)
  print("Default pipeline config: {}".format(config_path))
  return config_path


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


def include_followups(
    all_stages: List[Dict[str, Any]],
    selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Add explicitly configured post-stages, including disabled analyses."""
  by_name = {stage["name"]: stage for stage in all_stages}
  included = {stage["name"] for stage in selected}
  pending = list(selected)
  while pending:
    stage = pending.pop(0)
    followups = stage.get("then", [])
    if isinstance(followups, str):
      followups = [followups]
    for name in followups:
      followup = by_name.get(str(name))
      if followup is None:
        raise PipelineError("unknown follow-up stage {}".format(name))
      if followup["name"] not in included:
        included.add(followup["name"])
        pending.append(followup)
  selected_versions = {stage["name"]: stage for stage in selected}
  return [
    selected_versions.get(stage["name"], stage)
    for stage in all_stages if stage["name"] in included
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
  elif command == "analyze_worldgroups":
    prefix = [sys.executable, "scripts/analyze_worldgroups.py"]
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
  if command in ("worldgroups", "worldgroupba"):
    outputs = [Path(args["output"])]
    if args.get("calibration_output"):
      outputs.append(Path(args["calibration_output"]))
    return outputs
  if command == "observe":
    return [Path(args["output"])]
  if command == "worldpoints":
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
  if command == "analyze_worldgroups":
    destination = Path(args["output"])
    return [destination, destination.with_suffix(".xlsx")]
  return []


def calibration_copy_paths(stage: Mapping[str, Any], config: Mapping[str, Any]):
  """Collect final calibration JSONs under outputs/<dataset name>."""
  args = stage["args"]
  if stage.get("command") == "intrinsic":
    folder = Path(args.get("output_path") or args.get("image_path", "."))
    source = folder / (str(args.get("name", "intrinsic")) + ".json")
  elif stage.get("command") == "worldgroupba":
    source = Path(args["output"])
  else:
    return None
  dataset = config.get("variables", {}).get("dataset")
  dataset_name = Path(str(dataset)).name if dataset is not None else source.parent.parent.resolve().name
  return source, REPO_ROOT / "outputs" / dataset_name / source.name


def copy_calibration(paths, render_layout=False, court=None):
  if paths is None:
    return
  source, destination = paths
  try:
    if source.resolve() != destination.resolve():
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source, destination)
  except OSError as error:
    raise PipelineError("cannot copy {} to {}: {}".format(
      source, destination, error)) from error
  print("  copied: {}".format(destination))
  if render_layout:
    from multical.image.camera_layout import render_camera_layout
    try:
      image = render_camera_layout(destination, court=court)
    except (OSError, ValueError, KeyError) as error:
      raise PipelineError("cannot render camera layout for {}: {}".format(
        destination, error)) from error
    print("  camera layout: {}".format(image))


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
    if (
        key in ("image_path", "observe")
        and stage.get("command") == "worldpoints"
        and (
          (key == "image_path" and stage["args"].get("mode") == "markers")
          or (key == "observe" and stage["args"].get("mode") == "measured")
        )
        and Path(str(stage["args"].get("output", ""))).is_file()
    ):
      # An existing output YAML is sufficient for the editor's YAML-only mode.
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
  config_path = resolve_config_path(options.config)
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
  selected = include_followups(stages, selected)
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
    calibration_copy = calibration_copy_paths(stage, config)
    render_layout = stage.get("command") == "worldgroupba"
    command = command_for(stage)
    digest = fingerprint(stage, command)
    record = state["stages"].get(stage["name"])
    print("  command: {}".format(display_command(command)))
    if outputs:
      print("  outputs: {}".format(", ".join(str(path) for path in outputs)))
    if calibration_copy:
      print("  copy to: {}".format(calibration_copy[1]))
      if render_layout:
        print("  camera layout: {}".format(calibration_copy[1].with_suffix(".png")))

    if (
        options.resume
        and not options.force
        and not stage.get("interactive", False)
        and can_resume(record, digest, outputs)
    ):
      if not options.dry_run:
        copy_calibration(calibration_copy, render_layout=render_layout,
                         court=config.get("variables", {}).get("court"))
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
    try:
      copy_calibration(calibration_copy, render_layout=render_layout,
                         court=config.get("variables", {}).get("court"))
    except PipelineError as error:
      state["stages"][stage["name"]].update({
        "status": "failed",
        "duration_seconds": round(duration, 3),
        "error": str(error)
      })
      save_state(state_path, state)
      raise
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
    description="Initialize an experiment or run Multical pipeline stages.")
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--config", help="pipeline YAML file")
  source.add_argument(
    "--init", metavar="DATASET",
    help="create an experiment directory tree and pipeline YAML")
  source.add_argument(
    "--init-settings", metavar="YAML",
    help="initialize an experiment from a short settings YAML")
  source.add_argument(
    "--set-default-config", metavar="PATH_OR_KEYWORD",
    help="select the launcher's default config by path or unique keyword")
  parser.add_argument(
    "--template", type=Path,
    help="pipeline template for --init (default: 6-camera worldgroups)")
  parser.add_argument(
    "--boards", default=DEFAULT_BOARDS,
    help="board YAML written by --init (default: %(default)s)")
  parser.add_argument(
    "--world-board", default=DEFAULT_WORLD_BOARD,
    help="world marker board YAML written by --init (default: %(default)s)")
  parser.add_argument(
    "--cameras", nargs="+",
    help="camera names for --init, for example: cam0 cam1 cam2 cam3")
  parser.add_argument(
    "--group", dest="groups", action="append", default=[],
    help=("camera group for --init; repeat for each group, for example "
          "--group cam0,cam1; optional form: 01=cam0,cam1"))
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
    help="run despite resume, or overwrite an existing --init config")
  parser.add_argument(
    "--dry-run", action="store_true",
    help="print commands without creating files or running programs")
  parser.add_argument(
    "--list-stages", action="store_true", help="list configured stages")
  return parser


def main() -> int:
  try:
    options = make_parser().parse_args()
    if options.set_default_config:
      set_default_pipeline_config(options.set_default_config)
      return 0
    if options.init_settings:
      initialize_from_settings(
        Path(options.init_settings), force=options.force,
        template_override=options.template)
      return 0
    if options.init:
      groups = parse_camera_groups(options.cameras or [], options.groups)
      initialize_experiment(
        Path(options.init), template=options.template, force=options.force,
        cameras=options.cameras, groups=groups, boards=options.boards,
        world_board=options.world_board)
      return 0
    if options.template or options.cameras or options.groups:
      raise PipelineError(
        "--template, --cameras and --group can only be used with --init")
    return run_pipeline(options)
  except PipelineError as error:
    print("pipeline error: {}".format(error), file=sys.stderr)
    return 2
  except KeyboardInterrupt:
    print("\npipeline interrupted", file=sys.stderr)
    return 130


if __name__ == "__main__":
  raise SystemExit(main())
