import csv
from pathlib import Path

from scripts.merge_camera_sessions import main


def write_file(path: Path, content: bytes = b"image") -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(content)


def test_merges_sessions_with_continuous_synchronized_names(tmp_path):
  source = tmp_path / "camera data"
  write_file(source / "capture2" / "cam0" / "frame1.png", b"a")
  write_file(source / "capture2" / "cam1" / "frame1.png", b"b")
  write_file(source / "capture10" / "cm0" / "nested" / "frame2.JPG", b"c")
  write_file(source / "capture10" / "cm1" / "nested" / "frame2.JPG", b"d")
  write_file(source / "capture10" / "cm1" / "notes.txt", b"ignored")

  assert main([str(source)]) == 0

  output = source / "merged"
  assert (output / "cam0" / "000000.png").read_bytes() == b"a"
  assert (output / "cam1" / "000000.png").read_bytes() == b"b"
  assert (output / "cam0" / "000001.jpg").read_bytes() == b"c"
  assert (output / "cam1" / "000001.jpg").read_bytes() == b"d"

  with (output / "merge_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
  assert len(rows) == 4
  assert rows[-1]["source_session"] == "capture10"
  assert rows[-1]["source_relative_path"] == "nested/frame2.JPG"


def test_dry_run_does_not_create_output(tmp_path):
  source = tmp_path / "source"
  write_file(source / "session" / "cam0" / "1.png")

  assert main([str(source), "--dry-run"]) == 0
  assert not (source / "merged").exists()


def test_require_complete_rejects_missing_camera_frame(tmp_path):
  source = tmp_path / "source"
  write_file(source / "session" / "cam0" / "1.png")
  write_file(source / "session" / "cam1" / "2.png")

  assert main([str(source), "--require-complete"]) == 2
  assert not (source / "merged").exists()


def test_rejects_nonempty_output(tmp_path):
  source = tmp_path / "source"
  write_file(source / "session" / "cam0" / "1.png")
  write_file(source / "merged" / "old.png")

  assert main([str(source)]) == 2
  assert (source / "merged" / "old.png").is_file()
