import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
  Path(__file__).parents[1] / "scripts" / "stratified_intrinsic_split.py"
)
SPEC = importlib.util.spec_from_file_location(
  "stratified_intrinsic_split", SCRIPT_PATH
)
splitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splitter)


def synthetic_records(count=40):
  records = []
  for index in range(count):
    angle = 2.0 * np.pi * index / count
    records.append({
      "filename": "{:06d}.jpg".format(index),
      "path": "/images/{:06d}.jpg".format(index),
      "features": np.array([
        0.5 + 0.35 * np.cos(angle),
        0.5 + 0.35 * np.sin(angle),
        -2.0 + 0.5 * (index % 4),
        np.cos(angle),
        np.sin(angle),
        np.cos(2 * angle),
        np.sin(2 * angle)
      ])
    })
  return records


def test_stratified_split_has_requested_sizes_and_is_reproducible():
  first_calibration, first_validation = splitter.stratified_split(
    synthetic_records(), validation_count=10, seed=17
  )
  second_calibration, second_validation = splitter.stratified_split(
    synthetic_records(), validation_count=10, seed=17
  )

  assert len(first_calibration) == 30
  assert len(first_validation) == 10
  assert [item["filename"] for item in first_validation] == [
    item["filename"] for item in second_validation
  ]
  assert {item["filename"] for item in first_calibration}.isdisjoint(
    {item["filename"] for item in first_validation}
  )


def test_each_validation_stratum_remains_in_calibration():
  calibration, validation = splitter.stratified_split(
    synthetic_records(), validation_count=10, seed=23, cluster_count=5
  )
  calibration_strata = {item["stratum"] for item in calibration}
  validation_strata = {item["stratum"] for item in validation}

  assert validation_strata <= calibration_strata
  assert len(validation_strata) == 5
