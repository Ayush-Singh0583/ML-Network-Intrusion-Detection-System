"""
Tests for the identifier side-table, the capture reader and the new split
protocols -- TORCH-FREE (pandas, scikit-learn and pyarrow only).

Each test pins one way the study could go wrong without raising: a timestamp
read as morning that was afternoon, an address leaking into the feature
matrix, a "time-ordered" split that is not, a held-out class that was trained
on after all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import synth_cic  # noqa: E402
from config import (  # noqa: E402
    CACHE_VERSION, DAY_COL, LABEL_COL, META_CAPTURE, META_COLUMNS, META_DST_PORT,
    META_ROW, META_SRC_IP, META_TS,
)
from meta import (  # noqa: E402
    extract_meta, feature_columns, has_identifiers, meta_columns_present,
    parse_cic_timestamps, require_identifiers, time_order, time_position,
)
from preprocessing import (  # noqa: E402
    blocked_assignment, build_cache, build_splits, cache_state, load_clean, read_capture,
    sources_changed,
)

MORNING_WEB = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"


def _when(seconds: float) -> str:
    return str(pd.to_datetime(seconds, unit="s"))


# =====================================================================
# TIMESTAMPS
# =====================================================================


def test_bare_afternoon_hour_is_read_as_pm():
    """'3:30' on a capture that ran 09:00-17:00 is 15:30, not 03:30."""
    t = parse_cic_timestamps(["7/7/2017 3:30", "7/7/2017 9:05", "7/7/2017 12:59"])
    assert _when(t[0]) == "2017-07-07 15:30:00"
    assert _when(t[1]) == "2017-07-07 09:05:00"
    assert _when(t[2]) == "2017-07-07 12:59:00"


def test_afternoon_sorts_after_morning():
    """The failure this guards: read naively, 1:00 sorts before 9:00 and every
    time-ordered split is scrambled without an error."""
    t = parse_cic_timestamps(["4/7/2017 9:00", "4/7/2017 12:30", "4/7/2017 1:00", "4/7/2017 4:59"])
    assert list(np.argsort(t)) == [0, 1, 2, 3]


def test_seconds_and_zero_padding_are_parsed():
    t = parse_cic_timestamps(["03/07/2017 08:55:58", "3/7/2017 1:00:07"])
    assert _when(t[0]) == "2017-07-03 08:55:58"
    assert _when(t[1]) == "2017-07-03 13:00:07"


def test_iso_and_explicit_marker_are_not_shifted():
    """A 24-hour ISO stamp, or one carrying AM/PM, must not get the +12 rule."""
    t = parse_cic_timestamps(["2017-07-07 03:30:00", "07/07/2017 03:30:00 AM",
                              "07/07/2017 03:30:00 PM", "2017-07-07 15:30:00.250"])
    assert _when(t[0]) == "2017-07-07 03:30:00"
    assert _when(t[1]) == "2017-07-07 03:30:00"
    assert _when(t[2]) == "2017-07-07 15:30:00"
    assert _when(t[3]) == "2017-07-07 15:30:00"


def test_unparseable_timestamp_is_nan_never_a_guess():
    t = parse_cic_timestamps(["garbage", None, "", "32/13/2017 9:00"])
    assert np.isnan(t).all()


def test_month_first_file_is_detected_from_the_expected_date():
    day_first = parse_cic_timestamps(["3/7/2017 9:00"], expected_date="2017-07-03")
    month_first = parse_cic_timestamps(["7/3/2017 9:00"], expected_date="2017-07-03")
    assert _when(day_first[0]) == _when(month_first[0]) == "2017-07-03 09:00:00"


# =====================================================================
# SIDE-TABLE
# =====================================================================


def _raw(n=4, with_ids=True):
    d = {"Destination Port": [80, 443, 22, 53][:n], "f": np.arange(n, dtype=float),
         "Label": ["BENIGN"] * n}
    if with_ids:
        d.update({"Source IP": ["10.0.0.1"] * n, "Destination IP": ["10.0.0.2"] * n,
                  "Source Port": [40000 + i for i in range(n)], "Protocol": [6] * n,
                  "Timestamp": ["7/7/2017 3:30", "7/7/2017 9:00", "7/7/2017 9:01",
                                "7/7/2017 2:00"][:n]})
    return pd.DataFrame(d)


def test_missing_identifiers_are_explicit_not_invented():
    m = extract_meta(_raw(with_ids=False), capture_index=2)
    assert list(m.columns) == META_COLUMNS
    assert m[META_SRC_IP].isna().all() and np.isnan(m[META_TS]).all()
    assert (m[META_DST_PORT] >= 0).all()            # this one the file does carry
    assert not has_identifiers(m)
    with pytest.raises(RuntimeError, match="GeneratedLabelledFlows"):
        require_identifiers(m, "The behaviour layer")


def test_meta_columns_are_never_features():
    raw = _raw()
    frame = pd.concat([raw[["f", "Label"]], extract_meta(raw, 0, day="Friday")], axis=1)
    frame[DAY_COL] = "Friday"
    assert feature_columns(frame) == ["f"]
    assert set(meta_columns_present(frame)) == set(META_COLUMNS)


def test_time_order_uses_timestamps_and_breaks_ties_by_row():
    raw = _raw()
    frame = extract_meta(raw, 0, day="Friday")
    # rows: 15:30, 09:00, 09:01, 14:00  ->  09:00, 09:01, 14:00, 15:30
    assert list(time_order(frame)) == [1, 2, 3, 0]
    assert list(time_position(frame)) == [3, 0, 1, 2]


def test_time_order_falls_back_to_file_row_without_timestamps():
    m = extract_meta(_raw(with_ids=False), 0)
    assert list(time_order(m.iloc[::-1].reset_index(drop=True))) == [3, 2, 1, 0]


def test_captures_are_never_interleaved_by_time():
    """Order is capture first, then time: a later capture's morning must not
    sort ahead of an earlier capture's afternoon."""
    a = extract_meta(_raw(2), 0, day="Friday")
    b = extract_meta(_raw(2), 1, day="Friday")
    frame = pd.concat([b, a], ignore_index=True)
    assert list(frame[META_CAPTURE].to_numpy()[time_order(frame)]) == [0, 0, 1, 1]


# =====================================================================
# READING THE CAPTURES AS SHIPPED
# =====================================================================


def _frame(root: Path, **kw) -> pd.DataFrame:
    """The cached frame of a fixture, checked against THAT FIXTURE'S data folder.

    ``load_clean`` asks whether the CSVs a cache was built from are still the
    files in the data folder, and rebuilds if they are not.  Left to its
    default it asks about the project's own ``data/``.  On a machine that
    holds the real dataset there, the eight names match, the sizes do not,
    and the fixture's cache is replaced by 2.8 million real flows (seen on
    2026-10-05).  ``tests/conftest.py`` now seals the default off as well;
    naming the folder here says what the test means.
    """
    return load_clean(root / "cache" / "clean.parquet", folder_path=root / "data",
                      verbose=False, **kw)


def test_the_suite_cannot_see_the_projects_own_folders():
    """Every folder the code reads or writes by default must lie outside the
    project while the tests run: no real CSV, cache, run or paper table may
    be read, and none written.  ``tests/conftest.py`` arranges it."""
    import os

    import config

    root = Path(config.PROJECT_ROOT).resolve()
    folders = {
        "data": config.DATA_DIR, "cache": config.CACHE_DIR, "runs": config.RUNS_DIR,
        "saved models": config.ARTIFACT_DIR, "paper": os.environ.get("NIDS_PAPER_DIR", ""),
    }
    for what, folder in folders.items():
        assert str(folder), f"the {what} folder is not redirected"
        here = Path(folder).resolve()
        assert here != root and root not in here.parents, f"the {what} folder is {here}"
    assert not any(Path(config.DATA_DIR).glob("*.csv"))


@pytest.fixture(scope="module")
def glf(tmp_path_factory):
    d = tmp_path_factory.mktemp("glf")
    synth_cic.write_dataset(d / "data", n_benign=400, attack_scale=160, seed=1, distribution="glf")
    build_cache(d / "data", d / "cache" / "clean.parquet", verbose=False)
    return d


@pytest.fixture(scope="module")
def mlcsv(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlcsv")
    synth_cic.write_dataset(d / "data", n_benign=300, attack_scale=120, seed=1,
                            distribution="mlcsv")
    build_cache(d / "data", d / "cache" / "clean.parquet", verbose=False)
    return d


def test_cp1252_capture_is_read_and_blank_rows_are_dropped(glf):
    path = glf / "data" / MORNING_WEB
    with pytest.raises(UnicodeDecodeError):
        path.read_bytes().decode("utf-8")          # the file really is not UTF-8
    df = read_capture(path)
    assert df.attrs["encoding"] == "cp1252"
    assert df.attrs["raw_rows"] - len(df) == 40    # the trailing empty rows
    assert "nan" not in set(df[LABEL_COL].astype(str).str.lower())


def test_cache_keeps_identifiers_beside_the_features(glf):
    df = _frame(glf)
    assert has_identifiers(df)
    assert set(META_COLUMNS) <= set(df.columns)
    feats = feature_columns(df)
    assert not any(c.startswith("meta__") for c in feats)
    assert "Destination Port" not in feats and "Timestamp" not in feats
    labels = set(df[LABEL_COL].astype(str))
    assert {"WebAttack_BruteForce", "WebAttack_XSS", "WebAttack_SQLInjection"} <= labels
    assert "nan" not in labels
    state = cache_state(glf / "cache" / "clean.parquet")
    assert state["version"] == CACHE_VERSION and all(state["identifiers"].values())


def test_every_capture_parses_onto_its_own_day_in_working_hours(glf):
    df = _frame(glf)
    ts = pd.to_datetime(df[META_TS], unit="s")
    expected = {"Monday": 3, "Tuesday": 4, "Wednesday": 5, "Thursday": 6, "Friday": 7}
    for day, dom in expected.items():
        t = ts[df[DAY_COL].astype(str) == day]
        assert (t.dt.day == dom).all() and (t.dt.month == 7).all()
        assert t.dt.hour.min() >= 9 and t.dt.hour.max() <= 17


def test_mlcsv_cache_has_no_identifiers_and_says_so(mlcsv):
    df = _frame(mlcsv)
    assert not has_identifiers(df)
    assert (df[META_DST_PORT] >= 0).all()
    assert not any(cache_state(mlcsv / "cache" / "clean.parquet")["identifiers"].values())
    # the row index still gives a usable order
    b = build_splits("blocked", df=df, verbose=False)
    assert b.meta_test is not None and len(b.meta_test) == len(b.X_test)


def test_old_cache_layout_is_rebuilt_not_resumed(mlcsv, tmp_path):
    """A cache written before the side-table existed must not be read as
    'this dataset has no timestamps'."""
    cache = tmp_path / "cache" / "clean.parquet"
    build_cache(mlcsv / "data", cache, verbose=False)
    state_path = cache.parent / "clean_parts" / "_state.json"
    state = json.loads(state_path.read_text())
    state.pop("version")
    state_path.write_text(json.dumps(state))
    before = {p.name: p.stat().st_mtime_ns for p in state_path.parent.glob("*.parquet")}
    load_clean(cache, folder_path=mlcsv / "data", verbose=False)
    after = {p.name: p.stat().st_mtime_ns for p in state_path.parent.glob("*.parquet")}
    assert json.loads(state_path.read_text())["version"] == CACHE_VERSION
    assert all(after[k] != before[k] for k in before), "old parts were reused"


def test_replacing_the_csvs_rebuilds_the_cache_and_the_identifiers_appear(tmp_path):
    """The hand-off this project depends on: the data folder first holds the
    distribution WITHOUT addresses and timestamps, the cache is built from it,
    and then the eight files are replaced by the distribution that has them.
    The next read must come from the new files."""
    data, cache = tmp_path / "data", tmp_path / "cache" / "clean.parquet"
    synth_cic.write_dataset(data, n_benign=120, attack_scale=40, distribution="mlcsv")
    df = load_clean(cache, folder_path=data, verbose=False)
    assert not has_identifiers(df) and not any(cache_state(cache)["identifiers"].values())

    synth_cic.write_dataset(data, n_benign=120, attack_scale=40, distribution="glf")
    df = load_clean(cache, folder_path=data, verbose=False)
    assert has_identifiers(df) and all(cache_state(cache)["identifiers"].values())
    assert sources_changed(cache_state(cache), data) == []


def test_an_untouched_or_merely_copied_dataset_does_not_rebuild_the_cache(tmp_path):
    """Size decides, not modification time: moving the project to another
    machine must not cost a rebuild."""
    import os

    data, cache = tmp_path / "data", tmp_path / "cache" / "clean.parquet"
    synth_cic.write_dataset(data, n_benign=120, attack_scale=40, distribution="glf")
    load_clean(cache, folder_path=data, verbose=False)
    parts = cache.parent / "clean_parts"
    before = {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")}
    for f in data.glob("*.csv"):
        os.utime(f, (1_600_000_000, 1_600_000_000))            # new mtime, same bytes
    load_clean(cache, folder_path=data, verbose=False)
    assert {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")} == before
    # and a machine that holds the cache but not the CSVs can still read it
    load_clean(cache, folder_path=tmp_path / "nowhere", verbose=False)
    assert {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")} == before


def test_explicit_rebuild_discards_the_parts(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache" / "clean.parquet"
    synth_cic.write_dataset(data, n_benign=120, attack_scale=40, distribution="glf")
    build_cache(data, cache, verbose=False)
    parts = cache.parent / "clean_parts"
    before = {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")}
    build_cache(data, cache, verbose=False)                      # resume: nothing to do
    assert {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")} == before
    build_cache(data, cache, verbose=False, resume=False)        # what `cache --rebuild` does
    after = {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")}
    assert set(after) == set(before) and all(after[k] != before[k] for k in before)


def test_a_cache_that_needs_rebuilding_is_not_deleted_when_the_csvs_are_gone(tmp_path):
    """Look before deleting.  An outdated cache on a machine that no longer
    holds the CSVs cannot be rebuilt.  Throwing it away first would leave that
    machine with nothing; the build must stop and leave every file in place."""
    data, cache = tmp_path / "data", tmp_path / "cache" / "clean.parquet"
    synth_cic.write_dataset(data, n_benign=120, attack_scale=40, distribution="mlcsv")
    build_cache(data, cache, verbose=False)
    parts = cache.parent / "clean_parts"
    state_path = parts / "_state.json"
    state = json.loads(state_path.read_text())
    state.pop("version")                                      # now it reads as the old layout
    state_path.write_text(json.dumps(state))
    old_state = state_path.read_text()
    before = {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")}
    assert before and cache.exists()

    empty = tmp_path / "no_csvs_here"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Nothing was deleted"):
        load_clean(cache, folder_path=empty, verbose=False)
    with pytest.raises(FileNotFoundError, match="Nothing was deleted"):
        build_cache(empty, cache, verbose=False, resume=False)   # `cache --rebuild`, same rule
    assert {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")} == before
    assert cache.exists() and state_path.read_text() == old_state

    # one capture missing is enough to stop, and the message names it
    one = sorted(data.glob("*.csv"))[0]
    one.rename(tmp_path / one.name)
    with pytest.raises(FileNotFoundError, match=one.name.replace(".", r"\.")):
        load_clean(cache, folder_path=data, verbose=False)
    assert {p.name: p.stat().st_mtime_ns for p in parts.glob("*.parquet")} == before

    # put it back: now the rebuild goes ahead
    (tmp_path / one.name).rename(one)
    load_clean(cache, folder_path=data, verbose=False)
    assert json.loads(state_path.read_text())["version"] == CACHE_VERSION


def test_run_py_cache_accepts_rebuild():
    import run

    assert run.build_parser().parse_args(["cache", "--rebuild"]).rebuild is True
    assert run.build_parser().parse_args(["cache"]).rebuild is False


# =====================================================================
# SPLIT PROTOCOLS
# =====================================================================


def test_no_meta_column_reaches_a_model(glf):
    df = _frame(glf)
    for protocol, kw in (("crossday", {}), ("blocked", {}), ("random", {}),
                         ("loco", {"holdout_class": "PortScan"})):
        b = build_splits(protocol, df=df, verbose=False, **kw)
        assert not meta_columns_present(b.feature_names)
        assert b.X_train.shape[1] == len(b.feature_names)
        for m, X in ((b.meta_train, b.X_train), (b.meta_val, b.X_val), (b.meta_test, b.X_test)):
            assert m is not None and len(m) == len(X)


def test_blocked_split_is_time_ordered_inside_every_capture_and_class(glf):
    df = _frame(glf).reset_index(drop=True)
    a = blocked_assignment(df)
    pos = time_position(df)
    g = pd.DataFrame({"cap": df[META_CAPTURE], "lab": df[LABEL_COL].astype(str), "a": a, "pos": pos})
    checked = 0
    for (_cap, _lab), grp in g.groupby(["cap", "lab"]):
        if len(grp) < 3:
            assert (grp["a"] == 0).all()           # too small to split: trains whole
            continue
        tr, va, te = (grp.loc[grp["a"] == i, "pos"] for i in (0, 1, 2))
        assert len(tr) and len(va) and len(te)
        assert tr.max() < va.min() and va.max() < te.min()
        checked += 1
    assert checked >= 15


def test_blocked_keeps_every_class_in_training(glf):
    df = _frame(glf)
    b = build_splits("blocked", df=df, verbose=False)
    assert set(b.y_test_str) <= set(b.known_classes)
    assert (b.y_test >= 0).all()


def test_loco_holds_the_class_out_of_train_and_val_entirely(glf):
    df = _frame(glf)
    n_scan = int((df[LABEL_COL].astype(str) == "PortScan").sum())
    b = build_splits("loco", df=df, holdout_class="PortScan", verbose=False)
    assert "PortScan" not in set(b.y_train_str) | set(b.y_val_str)
    assert "PortScan" not in b.known_classes
    assert int((b.y_test_str == "PortScan").sum()) == n_scan          # ALL of it is test
    assert (b.y_test[b.y_test_str == "PortScan"] == -1).all()
    assert b.info["holdout_class"] == "PortScan" and b.info["holdout_rows"] == n_scan


def test_loco_rejects_nonsense():
    from test_preprocessing import _toy_frame
    with pytest.raises(ValueError, match="holdout_class"):
        build_splits("loco", df=_toy_frame(), verbose=False)
    with pytest.raises(ValueError, match="not in the data"):
        build_splits("loco", df=_toy_frame(), holdout_class="Nope", verbose=False)
    with pytest.raises(ValueError, match="BENIGN"):
        build_splits("loco", df=_toy_frame(), holdout_class="BENIGN", verbose=False)
    with pytest.raises(ValueError, match="only meaningful"):
        build_splits("crossday", df=_toy_frame(), holdout_class="DDoS", verbose=False)


def test_random_protocol_sees_friday_classes_and_crossday_does_not(glf):
    df = _frame(glf)
    rnd = build_splits("random", df=df, verbose=False)
    cross = build_splits("crossday", df=df, verbose=False)
    assert {"PortScan", "DDoS", "Bot"} <= set(rnd.known_classes)
    assert not ({"PortScan", "DDoS", "Bot"} & set(cross.known_classes))
    assert {"PortScan", "DDoS", "Bot"} <= set(cross.y_test_str[cross.y_test < 0])


def test_tail_validation_is_later_than_training_for_every_thursday_class(glf):
    df = _frame(glf)
    b = build_splits("crossday", df=df, val_mode="tail", verbose=False)
    thu_caps = sorted(set(b.meta_val[META_CAPTURE]))
    tr = b.meta_train[b.meta_train[META_CAPTURE].isin(thu_caps)].assign(lab=np.asarray(
        b.y_train_str)[b.meta_train[META_CAPTURE].isin(thu_caps).to_numpy()])
    va = b.meta_val.assign(lab=b.y_val_str)
    key = lambda m: list(zip(m[META_CAPTURE], m[META_TS], m[META_ROW]))  # noqa: E731
    for lab in set(va["lab"]):
        t, v = tr[tr["lab"] == lab], va[va["lab"] == lab]
        if len(t) and len(v):
            assert max(key(t)) < min(key(v)), f"{lab}: a validation flow precedes a training flow"
    assert b.info["val_mode"] == "tail"


def test_random_validation_is_interleaved_which_is_the_point(glf):
    """The contrast that makes the tail mode worth having: a random Thursday
    split does put validation flows before training flows."""
    df = _frame(glf)
    b = build_splits("crossday", df=df, val_mode="random", verbose=False)
    thu = b.meta_train[META_CAPTURE].isin(sorted(set(b.meta_val[META_CAPTURE])))
    assert b.meta_val[META_TS].min() < b.meta_train.loc[thu, META_TS].max()


def test_dedup_count_is_recorded_with_the_split(glf):
    df = _frame(glf)
    dup = pd.concat([df, df[df[DAY_COL].astype(str) == "Monday"].head(50)], ignore_index=True)
    b = build_splits("crossday", df=dup, verbose=False)
    assert b.info["rows_removed_by_dedup"] >= 50
    assert b.info["dedup_split"] == "train"


def test_random_protocol_leaves_validation_and_test_as_recorded(glf):
    """The leaky baseline must leak: a flow recorded four hundred times stays
    four hundred flows, spread over all three splits.  Only the training split
    is de-duplicated, as in every other study protocol -- otherwise its test
    set would be unique vectors while the others test raw flows, and the
    per-class rates of the protocols would not be the same quantity."""
    df = _frame(glf)
    one = df[df[LABEL_COL].astype(str) == "BENIGN"].head(1)
    dup = pd.concat([df] + [one] * 400, ignore_index=True)
    b = build_splits("random", df=dup, verbose=False)
    # classes too small to stratify three ways are dropped from this protocol, and said so
    rare = dup[LABEL_COL].astype(str).isin(b.info.get("dropped_rare_classes", [])).sum()
    assert (len(b.X_train) + len(b.X_val) + len(b.X_test) + b.info["rows_removed_by_dedup"]
            == len(dup) - rare)
    assert b.info["dedup_split"] == "train" and b.info["rows_removed_by_dedup"] >= 250

    def copies(X):
        return int(pd.DataFrame(X).duplicated(keep=False).sum())

    assert copies(b.X_train) == 0
    assert copies(b.X_val) >= 40 and copies(b.X_test) >= 40      # about 60 of the 400 each


def test_dedup_train_false_keeps_the_training_days_as_recorded(glf):
    df = _frame(glf)
    one = df[df[DAY_COL].astype(str) == "Monday"].head(1)
    dup = pd.concat([df] + [one] * 30, ignore_index=True)
    raw = build_splits("crossday", df=dup, verbose=False, dedup_train=False)
    ded = build_splits("crossday", df=dup, verbose=False)
    assert len(raw.X_train) - len(ded.X_train) == ded.info["rows_removed_by_dedup"] >= 30
    assert "rows_removed_by_dedup" not in raw.info
    assert len(raw.X_val) == len(ded.X_val) and len(raw.X_test) == len(ded.X_test)
