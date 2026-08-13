from pathlib import Path

from scripts.random_frame_dataset import build_dataset, main


def write_frames(root, cameras, frames):
  for camera in cameras:
    for frame in frames:
      path = root / camera / "{:06d}.jpg".format(frame)
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes("{}:{}".format(camera, frame).encode("ascii"))


def output_frames(root, camera):
  return sorted(int(path.stem) for path in (root / camera).glob("*.jpg"))


def test_sample_copies_same_reproducible_frames_for_every_camera(tmp_path):
  source = tmp_path / "source"
  first = tmp_path / "first"
  second = tmp_path / "second"
  write_frames(source, ["cam0", "cam1"], range(10))

  result = build_dataset(source, first, 2, 8, "sample", 3, seed=17)
  build_dataset(source, second, 2, 8, "sample", 3, seed=17)

  assert len(result["output_frames"]) == 3
  assert output_frames(first, "cam0") == result["output_frames"]
  assert output_frames(first, "cam1") == result["output_frames"]
  assert output_frames(second, "cam0") == result["output_frames"]
  assert (first / "random_frame_dataset.json").is_file()


def test_drop_outputs_only_range_minus_random_frames(tmp_path):
  source = tmp_path / "source"
  output = tmp_path / "output"
  write_frames(source, ["cam0", "cam1"], range(10))

  result = build_dataset(source, output, 3, 7, "drop", 2, seed=4)

  assert len(result["output_frames"]) == 3
  assert set(result["output_frames"]) | set(result["excluded_frames"]) == set(
    range(3, 8)
  )
  assert output_frames(output, "cam0") == result["output_frames"]
  assert 0 not in output_frames(output, "cam0")


def test_incomplete_multicamera_frame_is_not_a_candidate(tmp_path):
  source = tmp_path / "source"
  write_frames(source, ["cam0", "cam1"], range(5))
  (source / "cam1" / "000003.jpg").unlink()

  result = build_dataset(
    source, tmp_path / "output", 0, 4, "sample", 4, seed=2
  )

  assert result["incomplete_frames"] == [3]
  assert 3 not in result["candidate_frames"]


def test_dry_run_and_existing_output_are_safe(tmp_path):
  source = tmp_path / "source"
  output = tmp_path / "output"
  write_frames(source, ["cam0"], range(4))

  assert main([
    str(source), str(output), "--range", "0", "3", "--mode", "sample",
    "--count", "2", "--dry-run"
  ]) == 0
  assert not output.exists()

  output.mkdir()
  assert main([
    str(source), str(output), "--range", "0", "3", "--mode", "sample",
    "--count", "2"
  ]) == 2
