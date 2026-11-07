"""Tests for the driver-wise split (run with: pytest -q)."""

import numpy as np
import pandas as pd
import pytest

from src.data.split import check_no_leakage, make_driver_split, summarize_split


@pytest.fixture
def fake_driver_list() -> pd.DataFrame:
    """26 drivers x 10 classes with uneven image counts, like State Farm."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(26):
        for c in range(10):
            for i in range(rng.integers(5, 15)):
                rows.append({"subject": f"p{d:03d}", "classname": f"c{c}",
                             "img": f"img_{d}_{c}_{i}.jpg"})
    df = pd.DataFrame(rows)
    df["path"] = "imgs/train/" + df["classname"] + "/" + df["img"]
    return df


def test_no_driver_in_two_splits(fake_driver_list):
    out = make_driver_split(fake_driver_list, trials=200)
    check_no_leakage(out)
    sets = {s: set(out.loc[out["split"] == s, "subject"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])


def test_every_image_assigned(fake_driver_list):
    out = make_driver_split(fake_driver_list, trials=200)
    assert out["split"].notna().all()
    assert len(out) == len(fake_driver_list)


def test_driver_counts(fake_driver_list):
    summary = summarize_split(make_driver_split(fake_driver_list, trials=200))
    assert summary.loc["val", "drivers"] == 4    # round(26 * 0.15)
    assert summary.loc["test", "drivers"] == 4
    assert summary.loc["train", "drivers"] == 18


def test_deterministic(fake_driver_list):
    a = make_driver_split(fake_driver_list, seed=7, trials=100)
    b = make_driver_split(fake_driver_list, seed=7, trials=100)
    pd.testing.assert_series_equal(a["split"], b["split"])


def test_leakage_is_detected(fake_driver_list):
    out = make_driver_split(fake_driver_list, trials=50)
    first = out.index[0]
    out.loc[first, "split"] = "val" if out.loc[first, "split"] != "val" else "train"
    with pytest.raises(AssertionError):
        check_no_leakage(out)
