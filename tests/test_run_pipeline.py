import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


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
