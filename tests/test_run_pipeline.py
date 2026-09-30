import importlib.util
import json
import pytest
import yaml
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_initialize_experiment_creates_tree_and_rendered_config(
    tmp_path, monkeypatch):
  dataset = tmp_path / "experiment 01"
  config_dir = tmp_path / "configs"
  monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)
  launcher = tmp_path / "pipeline"
  launcher.write_text(
    "#!/bin/sh\n"
    "CONFIG=${MULTICAL_PIPELINE_CONFIG:-configs/old.yaml}\n",
    encoding="utf-8",
  )
  cameras = ["cam0", "cam1", "cam4", "cam5"]
  groups = pipeline.parse_camera_groups(
    cameras, ["cam0,cam1", "cam4,cam5"])

  config_path = pipeline.initialize_experiment(
    dataset, cameras=cameras, groups=groups)

  rendered = config_path.read_text(encoding="utf-8")
  assert "cameras: &cameras [cam0, cam1, cam4, cam5]" in rendered
  assert rendered.count("cameras: *cameras") == 2
  assert "cameras: [cam0, cam1]" in rendered
  assert "cameras: [cam4, cam5]" in rendered
  assert "\n\nstages:\n" in rendered
  assert "\n\n  analyze_intrinsic:" in rendered
  assert "\n\n  extrinsic_01:" in rendered
  assert "\n\n  analyze_extrinsic_45:" in rendered
  assert "\n\n  world_01:" in rendered
  assert config_path == (
    config_dir / "pipeline.worldgroups.experiment 01.yaml"
  )
  assert launcher.read_text(encoding="utf-8").endswith(
    "CONFIG=${MULTICAL_PIPELINE_CONFIG:-"
    "'configs/pipeline.worldgroups.experiment 01.yaml'}\n"
  )
  for relative_path in pipeline.EXPERIMENT_DIRECTORIES:
    assert (dataset / relative_path).is_dir()
  config = pipeline.load_config(config_path)
  assert config["variables"]["dataset"] == str(dataset)
  assert config["variables"]["output_root"] == str(dataset)
  assert config["variables"]["boards"] == (
    "boards/charuco_1600x1200.yaml"
  )
  assert config["variables"]["cameras"] == cameras
  assert config["stages"]["intrinsic"]["args"]["image_path"] == (
    str(dataset / "intrinsic")
  )
  assert config["stages"]["observe"]["args"]["image_path"] == (
    str(dataset / "observe" / "measured_points")
  )
  stages = config["stages"]
  assert "extrinsic_01" in stages
  assert "extrinsic_45" in stages
  assert "extrinsic_24" not in stages
  assert stages["extrinsic_45"]["args"]["cameras"] == ["cam4", "cam5"]
  assert stages["worldgroups"]["needs"] == ["world_01", "world_45"]
  assert stages["worldgroupba"]["args"]["group_names"] == [
    "group01", "group45"
  ]
  assert stages["worldpoints_marker_01"]["enabled"] is False
  assert stages["worldpoints_marker_01"]["interactive"] is True
  assert stages["worldpoints_marker_01"]["args"]["output"] == (
    str(dataset / "world" / "world_markers_01.yaml")
  )
  assert stages["worldpoints_marker_45"]["args"]["world_board"] == (
    "boards/world_boards/标定板图案设计.yaml"
  )
  assert stages["worldpoints_measured"]["args"]["observe"] == (
    str(dataset / "observe" / "measured_observations.yaml")
  )
  stage_names = list(stages)
  assert stage_names.index("analyze_intrinsic") == (
    stage_names.index("intrinsic") + 1
  )
  assert stage_names.index("analyze_extrinsic_01") == (
    stage_names.index("extrinsic_01") + 1
  )
  assert stages["intrinsic"]["then"] == ["analyze_intrinsic"]
  assert stages["analyze_intrinsic"]["enabled"] is False
  assert stages["analyze_intrinsic"]["needs"] == ["intrinsic"]
  assert stages["extrinsic_45"]["then"] == ["analyze_extrinsic_45"]
  assert stages["analyze_extrinsic_45"]["enabled"] is False
  assert stages["analyze_extrinsic_45"]["needs"] == ["extrinsic_45"]


def test_parse_camera_groups_supports_explicit_labels_and_validates_coverage():
  assert pipeline.parse_camera_groups(
    ["left", "right"], ["stereo=left,right"]
  ) == [("stereo", ["left", "right"])]

  try:
    pipeline.parse_camera_groups(
      ["cam0", "cam1", "cam2"], ["cam0,cam1"])
  except pipeline.PipelineError as error:
    assert "cam2" in str(error)
  else:
    raise AssertionError("missing camera should be rejected")


@pytest.mark.parametrize("court", [None, "tennis", "badminton"])
def test_initialize_from_settings_file(tmp_path, monkeypatch, court):
  dataset = tmp_path / "capture"
  settings = tmp_path / "experiment.init.yaml"
  settings.write_text(
    "dataset: {!s}\n"
    "boards: boards/charuco_1600x1200.yaml\n"
    "groups:\n"
    "  '01': [cam0, cam1]\n"
    "  '45': [cam4, cam5]\n".format(dataset),
    encoding="utf-8",
  )
  monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)

  if court is not None:
    with settings.open("a") as stream:
      stream.write("court: {}\n".format(court))
  config_path = pipeline.initialize_from_settings(settings)
  rendered = pipeline.load_config(config_path)
  assert rendered["variables"]["court"] == (court or "tennis")
  for stage in pipeline.stage_items(rendered):
    if stage.get("command") == "worldpoints":
      assert stage["args"]["court"] == (court or "tennis")
      command = pipeline.command_for(stage)
      assert command[command.index("--court") + 1] == (court or "tennis")

  dataset = dataset.with_name("capture.{}".format(court or "tennis"))
  assert config_path == tmp_path / "configs" / (
    "pipeline.worldgroups.{}.yaml".format(dataset.name)
  )
  for relative_path in pipeline.EXPERIMENT_DIRECTORIES:
    assert (dataset / relative_path).is_dir()
  config = pipeline.load_config(config_path)
  assert config["variables"]["dataset"] == str(dataset)
  assert config["variables"]["output_root"] == str(dataset)
  assert config["variables"]["cameras"] == [
    "cam0", "cam1", "cam4", "cam5"
  ]
  assert config["stages"]["worldgroups"]["args"]["group_names"] == [
    "group01", "group45"
  ]



@pytest.mark.parametrize("name", [
  "20260924", "20260924.badminton", "20260924_badminton",
  "20260924-badminton",
])
def test_init_settings_normalizes_court_suffix(tmp_path, monkeypatch, name):
  monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)
  launcher = tmp_path / "pipeline"
  launcher.write_text("CONFIG=${MULTICAL_PIPELINE_CONFIG:-configs/old.yaml}\n")
  settings = tmp_path / "init.yaml"
  output_root = tmp_path / "results"
  settings.write_text(yaml.safe_dump({
    "dataset": str(tmp_path / name),
    "court": "badminton",
    "output_root": str(output_root),
    "groups": {"01": ["cam0", "cam1"]},
  }))
  config_path = pipeline.initialize_from_settings(settings)
  assert config_path.name == "pipeline.worldgroups.20260924.badminton.yaml"
  config = pipeline.load_config(config_path)
  dataset = tmp_path / "20260924.badminton"
  assert config["variables"]["dataset"] == str(dataset)
  assert config["variables"]["output_root"] == str(output_root)
  assert config["stages"]["worldpoints_measured"]["args"]["observe"] == str(
    dataset / "observe/measured_observations.yaml")
  assert config_path.name in launcher.read_text()
  config_path.write_text("custom: true\n")
  assert pipeline.initialize_from_settings(settings) == config_path
  assert config_path.read_text() == "custom: true\n"


def test_initialize_experiment_keeps_config_unless_forced(tmp_path):
  template = tmp_path / "template.yaml"
  template.write_text("variables:\n  dataset: DATASET\nstages: {}\n")
  dataset = tmp_path / "dataset"
  config_dir = tmp_path / "configs"
  config_path = pipeline.initialize_experiment(
    dataset, template, config_dir=config_dir)
  config_path.write_text("custom: true\n", encoding="utf-8")

  pipeline.initialize_experiment(dataset, template, config_dir=config_dir)
  assert config_path.read_text(encoding="utf-8") == "custom: true\n"

  pipeline.initialize_experiment(
    dataset, template, force=True, config_dir=config_dir)
  assert "custom" not in config_path.read_text(encoding="utf-8")


def test_external_config_dir_does_not_modify_repository_launcher(
    tmp_path, monkeypatch):
  repository = tmp_path / "repo"
  repository.mkdir()
  launcher = repository / "pipeline"
  original = "CONFIG=${MULTICAL_PIPELINE_CONFIG:-configs/current.yaml}\n"
  launcher.write_text(original, encoding="utf-8")
  template = tmp_path / "template.yaml"
  template.write_text("variables:\n  dataset: DATASET\nstages: {}\n")
  monkeypatch.setattr(pipeline, "REPO_ROOT", repository)

  pipeline.initialize_experiment(
    tmp_path / "dataset", template,
    config_dir=tmp_path / "external-configs")

  assert launcher.read_text(encoding="utf-8") == original


def test_cli_arguments_support_flags_lists_and_false_values():
  assert pipeline.cli_arguments({
    "cameras": ["cam0", "cam1"],
    "fix_intrinsic": True,
    "refine": False,
    "optional": None
  }) == [
    "--cameras", "cam0", "cam1",
    "--fix_intrinsic",
    "--refine", "false"
  ]


def test_config_keyword_requires_unique_match(tmp_path, monkeypatch):
  config_dir = tmp_path / "configs"
  config_dir.mkdir()
  first = config_dir / "pipeline.worldgroups.20260901.yaml"
  first.write_text("stages: {}\n", encoding="utf-8")
  monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)

  assert pipeline.resolve_config_path("0901") == first.resolve()

  second = config_dir / "pipeline.backup.0901.yaml"
  second.write_text("stages: {}\n", encoding="utf-8")
  try:
    pipeline.resolve_config_path("0901")
  except pipeline.PipelineError as error:
    message = str(error)
    assert "pipeline.backup.0901.yaml" in message
    assert "pipeline.worldgroups.20260901.yaml" in message
  else:
    raise AssertionError("ambiguous config keyword should be rejected")


def test_set_default_config_accepts_keyword(tmp_path, monkeypatch):
  config_dir = tmp_path / "configs"
  config_dir.mkdir()
  config = config_dir / "pipeline.worldgroups.20260901.yaml"
  config.write_text("stages: {}\n", encoding="utf-8")
  launcher = tmp_path / "pipeline"
  launcher.write_text(
    "CONFIG=${MULTICAL_PIPELINE_CONFIG:-configs/old.yaml}\n",
    encoding="utf-8",
  )
  monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)

  assert pipeline.set_default_pipeline_config("0901") == config.resolve()
  assert launcher.read_text(encoding="utf-8") == (
    "CONFIG=${MULTICAL_PIPELINE_CONFIG:-"
    "configs/pipeline.worldgroups.20260901.yaml}\n"
  )


def test_world_manual_pipeline_is_opt_in_and_tracks_world_points():
  config = pipeline.load_config(
    SCRIPT.parents[1] / "configs" / "pipeline.worldgroups.20260811.yaml"
  )
  stages = {
    stage["name"]: stage for stage in pipeline.stage_items(config)
  }

  observe = stages["observe_world_45"]
  assert observe["enabled"] is False
  assert observe["args"]["cameras"] == ["cam4", "cam5"]
  assert observe["args"]["image_path"].endswith(
    "20260811/world/world_images"
  )
  assert observe["args"]["world_correspondences"].endswith(
    "20260811/world/world_markers_45.yaml"
  )
  assert "world_correspondences" in pipeline.INPUT_ARGUMENTS
  assert "--world_correspondences" in pipeline.command_for(observe)
  assert stages["worldgroupba_manual"]["args"]["output"].endswith(
    "world_extrinsic.manual.json"
  )
  assert stages["worldgroupba"]["args"]["output"].endswith(
    "world_extrinsic.json"
  )


def test_worldpoints_editors_are_configured_for_20260811_test():
  config = pipeline.load_config(
    SCRIPT.parents[1] / "configs" / "pipeline.worldgroups.20260811-test.yaml"
  )
  stages = {
    stage["name"]: stage for stage in pipeline.stage_items(config)
  }

  marker = stages["worldpoints_marker_01"]
  assert marker["enabled"] is False
  assert marker["interactive"] is True
  assert marker["args"]["image_path"] == (
    "20260811-test/world/world_images"
  )
  assert marker["args"]["output"].endswith("world_markers_01.yaml")
  assert pipeline.output_paths(marker) == [Path(marker["args"]["output"])]
  assert pipeline.command_for(marker)[3] == "worldpoints"

  measured = stages["worldpoints_measured_multi"]
  assert measured["args"]["observe"] == (
    "20260811-test/observe2/measured_observations.yaml"
  )
  assert "observe" in pipeline.INPUT_ARGUMENTS


def test_worldpoints_use_reusable_marker_board():
  config = pipeline.load_config(
    SCRIPT.parents[1] / "configs" / "pipeline.worldgroups.20260901.yaml"
  )
  stages = {
    stage["name"]: stage for stage in pipeline.stage_items(config)
  }

  for name, suffix in (
      ("worldpoints_marker_01", "01"),
      ("worldpoints_marker_23", "23")):
    stage = stages[name]
    assert stage["enabled"] is False
    assert stage["interactive"] is True
    assert stage["args"]["world_board"] == (
      "boards/world_boards/标定板图案设计.yaml"
    )
    assert "marker_ids" not in stage["args"]
    assert "marker_heights" not in stage["args"]
    assert "marker_occurrences" not in stage["args"]
    assert stage["args"]["output"].endswith(
      "world_markers_{}.yaml".format(suffix)
    )


def test_worldpoints_existing_marker_yaml_allows_missing_images(tmp_path):
  output = tmp_path / "world_markers_01.yaml"
  output.write_text("captures: []\n", encoding="utf-8")
  stage = {
    "name": "worldpoints_marker_01",
    "command": "worldpoints",
    "args": {
      "mode": "markers",
      "image_path": str(tmp_path / "missing_images"),
      "output": str(output),
    },
  }

  assert pipeline.missing_inputs(stage) == []
  output.unlink()
  assert pipeline.missing_inputs(stage) == [
    "image_path={}".format(tmp_path / "missing_images")
  ]


def test_worldpoints_existing_measured_yaml_allows_missing_observe(tmp_path):
  output = tmp_path / "measured_world_points.yaml"
  output.write_text("points: {}\n", encoding="utf-8")
  stage = {
    "name": "worldpoints_measured",
    "command": "worldpoints",
    "args": {
      "mode": "measured",
      "observe": str(tmp_path / "missing_observe.yaml"),
      "output": str(output),
    },
  }

  assert pipeline.missing_inputs(stage) == []
  output.unlink()
  assert pipeline.missing_inputs(stage) == [
    "observe={}".format(tmp_path / "missing_observe.yaml")
  ]


def test_grouped_stereo_config_enables_warmup_only_for_local_pairs():
  config = pipeline.load_config(
    SCRIPT.parents[1] / "configs" / "pipeline.worldgroups.20260804.yaml"
  )
  stages = {
    stage["name"]: stage for stage in pipeline.stage_items(config)
  }

  assert stages["extrinsic_01"]["args"][
    "warmup_before_outlier_rejection"
  ] is True
  assert stages["extrinsic_23"]["args"][
    "warmup_before_outlier_rejection"
  ] is True
  assert "warmup_before_outlier_rejection" not in stages["intrinsic"]["args"]
  assert stages["worldgroupba"]["args"]["relative_prior_weights"] == [
    2.0, 5.0
  ]


def test_validation_camera_selector_uses_group_and_camera():
  stages = [
    {
      "name": "intrinsic", "group": "calibration", "command": "intrinsic",
      "enabled": True, "args": {"cameras": ["cam0", "cam3"]}
    },
    {
      "name": "validate_cam3", "group": "validation",
      "command": "calibrate", "enabled": True,
      "args": {"cameras": ["cam3"], "master": "cam0"}
    }
  ]
  selected = pipeline.select_stages(
    stages, ["validation"], "cam3", None, None)
  assert [stage["name"] for stage in selected] == ["validate_cam3"]
  assert selected[0]["args"]["master"] == "cam3"


def test_disabled_stage_can_be_selected_explicitly_but_not_by_all():
  stages = [{
    "name": "observe", "group": "reconstruction", "command": "observe",
    "enabled": False, "args": {}
  }]
  assert pipeline.select_stages(
    stages, ["all"], None, None, None) == []
  selected = pipeline.select_stages(
    stages, ["observe"], None, None, None)
  assert [stage["name"] for stage in selected] == ["observe"]


def test_subset_calibration_keeps_only_requested_camera(tmp_path):
  calibration = tmp_path / "intrinsic.json"
  calibration.write_text(json.dumps({
    "cameras": {
      "cam0": {"K": [[1]], "dist": [[]]},
      "cam3": {"K": [[3]], "dist": [[]]}
    },
    "image_sets": {"rgb": []}
  }), encoding="utf-8")
  stage = {
    "name": "validate_cam3",
    "command": "calibrate",
    "args": {
      "image_path": str(tmp_path),
      "output_path": str(tmp_path / "output"),
      "cameras": ["cam3"],
      "calibration": str(calibration)
    }
  }

  pipeline.subset_calibration(stage, dry_run=False)

  subset_path = Path(stage["args"]["calibration"])
  subset = json.loads(subset_path.read_text(encoding="utf-8"))
  assert list(subset["cameras"]) == ["cam3"]
  first_mtime = subset_path.stat().st_mtime_ns
  pipeline.subset_calibration(stage, dry_run=False)
  assert subset_path.stat().st_mtime_ns == first_mtime


def test_resume_requires_matching_state_and_outputs(tmp_path):
  output = tmp_path / "result.json"
  output.write_text("{}", encoding="utf-8")
  record = {"status": "success", "fingerprint": "same"}
  assert pipeline.can_resume(record, "same", [output])
  assert not pipeline.can_resume(record, "changed", [output])
  output.unlink()
  assert not pipeline.can_resume(record, "same", [output])


def test_dependencies_are_added_in_config_order():
  stages = [
    {"name": "intrinsic", "enabled": True, "needs": []},
    {"name": "extrinsic", "enabled": True, "needs": ["intrinsic"]},
    {"name": "world", "enabled": True, "needs": ["extrinsic"]},
    {"name": "report", "enabled": True, "needs": ["world"]}
  ]
  selected = pipeline.include_dependencies(stages, [stages[-1]])
  assert [stage["name"] for stage in selected] == [
    "intrinsic", "extrinsic", "world", "report"
  ]


def test_followup_analysis_is_added_even_when_disabled():
  stages = [
    {
      "name": "intrinsic", "enabled": True,
      "then": ["analyze_intrinsic"]
    },
    {
      "name": "analyze_intrinsic", "enabled": False,
      "needs": ["intrinsic"]
    },
    {"name": "extrinsic_01", "enabled": True},
  ]

  selected = pipeline.include_followups(stages, [stages[0]])

  assert [stage["name"] for stage in selected] == [
    "intrinsic", "analyze_intrinsic"
  ]


def test_image_deletion_changes_stage_fingerprint(tmp_path):
  camera = tmp_path / "cam3"
  camera.mkdir()
  first = camera / "000000.jpg"
  second = camera / "000001.jpg"
  first.write_bytes(b"first")
  second.write_bytes(b"second")
  stage = {
    "args": {
      "image_path": str(tmp_path),
      "cameras": ["cam3"]
    }
  }
  before = pipeline.fingerprint(stage, ["multical", "intrinsic"])
  second.unlink()
  after = pipeline.fingerprint(stage, ["multical", "intrinsic"])
  assert before != after


def test_non_image_outputs_do_not_change_image_fingerprint(tmp_path):
  camera = tmp_path / "cam3"
  camera.mkdir()
  (camera / "000000.jpg").write_bytes(b"image")
  stage = {
    "args": {
      "image_path": str(tmp_path),
      "cameras": "cam3"
    }
  }
  before = pipeline.fingerprint(stage, ["multical", "intrinsic"])
  (tmp_path / "intrinsic.json").write_text("{}", encoding="utf-8")
  after = pipeline.fingerprint(stage, ["multical", "intrinsic"])
  assert before == after


def test_worldgroups_command_tracks_all_inputs_and_outputs(tmp_path):
  intrinsic = tmp_path / "intrinsic.json"
  calibration01 = tmp_path / "calibration01.json"
  calibration23 = tmp_path / "calibration23.json"
  world01 = tmp_path / "world01.json"
  world23 = tmp_path / "world23.json"
  for path in (
      intrinsic, calibration01, calibration23, world01, world23):
    path.write_text("{}", encoding="utf-8")
  output = tmp_path / "world.json"
  combined = tmp_path / "calibration.json"
  stage = {
    "name": "worldgroups",
    "command": "worldgroups",
    "args": {
      "intrinsic": str(intrinsic),
      "calibrations": [str(calibration01), str(calibration23)],
      "world_extrinsics": [str(world01), str(world23)],
      "output": str(output),
      "calibration_output": str(combined)
    }
  }

  assert pipeline.missing_inputs(stage) == []
  assert pipeline.output_paths(stage) == [output, combined]
  command = pipeline.command_for(stage)
  assert command[3] == "worldgroups"
  assert "--calibrations" in command


def test_worldgroupba_tracks_workspaces_inputs_and_outputs(tmp_path):
  inputs = {
    name: tmp_path / name for name in (
      "intrinsic.json", "initial.json", "calibration01.json",
      "calibration23.json", "calibration01.pkl", "calibration23.pkl",
      "world01.yaml", "world23.yaml"
    )
  }
  for path in inputs.values():
    path.write_text("{}", encoding="utf-8")
  output = tmp_path / "world.json"
  combined = tmp_path / "calibration.json"
  stage = {
    "name": "worldgroupba",
    "command": "worldgroupba",
    "args": {
      "intrinsic": str(inputs["intrinsic.json"]),
      "initial_world_extrinsics": str(inputs["initial.json"]),
      "calibrations": [
        str(inputs["calibration01.json"]),
        str(inputs["calibration23.json"])
      ],
      "workspaces": [
        str(inputs["calibration01.pkl"]),
        str(inputs["calibration23.pkl"])
      ],
      "correspondences": [
        str(inputs["world01.yaml"]), str(inputs["world23.yaml"])
      ],
      "output": str(output),
      "calibration_output": str(combined)
    }
  }

  assert pipeline.missing_inputs(stage) == []
  assert pipeline.output_paths(stage) == [output, combined]
  command = pipeline.command_for(stage)
  assert command[3] == "worldgroupba"
  assert "--workspaces" in command


def test_analyze_worldgroups_tracks_json_and_xlsx_outputs(tmp_path):
  world = tmp_path / "world.json"
  evaluation = tmp_path / "evaluation.json"
  world.write_text("{}", encoding="utf-8")
  evaluation.write_text("{}", encoding="utf-8")
  output = tmp_path / "worldgroups_analysis.json"
  stage = {
    "name": "analyze_worldgroups",
    "command": "analyze_worldgroups",
    "args": {
      "world_extrinsics": str(world),
      "evaluation": str(evaluation),
      "output": str(output)
    }
  }

  assert pipeline.missing_inputs(stage) == []
  assert pipeline.output_paths(stage) == [output, output.with_suffix(".xlsx")]
  command = pipeline.command_for(stage)
  assert command[1].endswith("scripts/analyze_worldgroups.py")


def test_calibration_commands_copy_after_success_and_on_resume(tmp_path, monkeypatch):
  from types import SimpleNamespace

  monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)
  for command, filename in [("intrinsic", "intrinsic.json"),
                            ("worldgroupba", "world_extrinsic.json")]:
    source = tmp_path / "custom_results" / command / filename
    args = ({"output_path": str(source.parent)} if command == "intrinsic"
            else {"output": str(source)})
    config_path = tmp_path / (command + ".yaml")
    config_path.write_text(pipeline.yaml.safe_dump({
      "variables": {"dataset": "data/20260901"},
      "stages": {command: {"command": command, "args": args}},
    }))
    options = pipeline.make_parser().parse_args([
      "--config", str(config_path), "--resume"])
    destination = tmp_path / "outputs" / "20260901" / filename
    calls = []

    def generate(argv, env):
      calls.append(argv)
      source.write_text(json.dumps({"cameras": {
        "cam0": {"world_to_camera": {
          "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T": [-30, -5, -3]}
        }
      }}))
      return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", generate)
    options.dry_run = True
    assert pipeline.run_pipeline(options) == 0
    assert not destination.exists()
    assert not calls
    options.dry_run = False
    assert pipeline.run_pipeline(options) == 0
    assert destination.read_bytes() == source.read_bytes()
    assert destination.with_suffix(".png").exists() == (command == "worldgroupba")
    destination.write_text("stale")
    assert pipeline.run_pipeline(options) == 0
    assert destination.read_bytes() == source.read_bytes()
    assert len(calls) == 1
    destination.unlink()
    options.dry_run = True
    assert pipeline.run_pipeline(options) == 0
    assert not destination.exists()
    options.dry_run = False
    assert pipeline.run_pipeline(options) == 0
    assert destination.read_bytes() == source.read_bytes()
    assert len(calls) == 1

    options.force = True
    monkeypatch.setattr(pipeline.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1))
    destination.write_text("previous successful result")
    import pytest
    with pytest.raises(pipeline.PipelineError, match="exit code 1"):
      pipeline.run_pipeline(options)
    assert destination.read_text() == "previous successful result"


def test_launcher_accepts_config_before_and_after_stage(tmp_path):
  import os
  import subprocess

  shim = tmp_path / "uv"
  shim.write_text('#!/bin/sh\nshift 2\nprintf "%s\\n" "$@"\n')
  shim.chmod(0o755)
  env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
  launcher = str(SCRIPT.parents[1] / "pipeline")
  for args in [
      ["--config", "0811", "stage", "intrinsic"],
      ["0811", "--stage", "intrinsic"],
      ["--config", "0811", "--stage", "intrinsic"],
      ["stage", "--config", "0811", "intrinsic"],
      ["stage", "-c", "0811", "intrinsic"]]:
    result = subprocess.run(["sh", launcher, *args, "--dry-run"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    parsed = pipeline.make_parser().parse_args(result.stdout.splitlines()[1:])
    assert parsed.config == "0811"
    assert parsed.stage == ["intrinsic"]
    assert parsed.dry_run
  for args in [["stage"], ["stage", "--config"],
               ["stage", "--config", "0811", "--dry-run"]]:
    result = subprocess.run(["sh", launcher, *args], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_initialize_rejects_unknown_court_before_creating_dataset(tmp_path):
  settings = tmp_path / "init.yaml"
  dataset = tmp_path / "capture"
  settings.write_text(yaml.safe_dump({
    "dataset": str(dataset), "groups": {"01": ["cam0", "cam1"]},
    "court": "unknown",
  }))
  with pytest.raises(pipeline.PipelineError, match="court"):
    pipeline.initialize_from_settings(settings)
  assert not dataset.exists()


def test_copy_calibration_passes_court_to_layout(tmp_path, monkeypatch):
  import multical.image.camera_layout as layout

  source, destination = tmp_path / "world.json", tmp_path / "outputs" / "world.json"
  source.write_text('{}')
  calls = []
  monkeypatch.setattr(layout, "render_camera_layout",
                      lambda path, court=None: calls.append((path, court)))
  pipeline.copy_calibration((source, destination), render_layout=True, court="badminton")
  assert destination.read_text() == '{}'
  assert calls == [(destination, "badminton")]


@pytest.mark.parametrize("group_count", [2, 3, 4])
@pytest.mark.parametrize("auto_analyze", [True, False])
def test_default_template_supports_dynamic_groups(tmp_path, group_count, auto_analyze):
  cameras = ["cam{}".format(i) for i in range(group_count * 2)]
  groups = [("{}{}".format(i, i + 1), cameras[i:i + 2])
            for i in range(0, len(cameras), 2)]
  config_path = pipeline.initialize_experiment(
    tmp_path / "capture.tennis", config_dir=tmp_path / "configs",
    cameras=cameras, groups=groups, auto_analyze=auto_analyze,
  )
  config = pipeline.load_config(config_path)
  stages = config["stages"]
  assert ("then" in stages["intrinsic"]) == auto_analyze
  assert stages["worldgroups"]["needs"] == ["world_" + label for label, _ in groups]
  assert stages["worldgroupba"]["args"]["group_names"] == [
    "group" + label for label, _ in groups]
  assert config["variables"]["cameras"] == cameras
  for label, members in groups:
    extrinsic = stages["extrinsic_" + label]
    assert extrinsic["args"]["cameras"] == members
    assert ("then" in extrinsic) == auto_analyze
    if auto_analyze:
      assert extrinsic["then"] == ["analyze_extrinsic_" + label]
    assert stages["analyze_extrinsic_" + label]["needs"] == ["extrinsic_" + label]
    assert "worldpoints_marker_" + label in stages
  assert "worldpoints_measured" in stages
  for stage in stages.values():
    for key in ("needs", "then"):
      assert set(stage.get(key, [])) <= stages.keys()
