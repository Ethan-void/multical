import pickle

from structs.struct import struct

from multical.io.detections import (
  check_dataset_similarity, try_load_detections
)


def cache_key(filenames, boards="board-a", image_sizes=((100, 80),)):
  return struct(
    filenames=filenames,
    boards=boards,
    image_sizes=image_sizes
  )


def loaded_cache(key, detected_points="points"):
  return struct(cache_key=key, detected_points=detected_points)


def test_cache_camera_count_mismatch_is_not_reused():
  loaded = loaded_cache(cache_key([["old/cam0/000.jpg"]]))
  current = cache_key([
    ["new/cam0/000.jpg"],
    ["new/cam3/000.jpg"]
  ])
  assert not check_dataset_similarity(loaded, current)


def test_cache_image_count_mismatch_is_not_reused():
  loaded = loaded_cache(cache_key([["old/cam0/000.jpg"]]))
  current = cache_key([[
    "new/cam0/000.jpg", "new/cam0/001.jpg"
  ]])
  assert not check_dataset_similarity(loaded, current)


def test_moved_dataset_cache_is_reused_when_metadata_matches():
  loaded = loaded_cache(cache_key([["old/root/cam0/000.jpg"]]))
  current = cache_key([["new/root/cam0/000.jpg"]])
  assert check_dataset_similarity(loaded, current)


def test_changed_board_is_not_reused():
  loaded = loaded_cache(cache_key(
    [["old/root/cam0/000.jpg"]], boards="board-a"))
  current = cache_key(
    [["new/root/cam0/000.jpg"]], boards="board-b")
  assert not check_dataset_similarity(loaded, current)


def test_corrupt_cache_is_ignored(tmp_path):
  filename = tmp_path / "detections.pkl"
  filename.write_bytes(b"not a pickle")
  assert try_load_detections(
    filename, cache_key([["root/cam0/000.jpg"]])) is None
