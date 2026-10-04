"""
End-to-end test of the study driver -- TORCH-FREE.

Runs the real command line (``python src/study.py ...``) against small
synthetic captures in a temporary folder.  ``NIDS_DATA_DIR``, ``NIDS_CACHE_DIR``,
``NIDS_RUNS_DIR``, ``NIDS_ARTIFACT_DIR`` and ``NIDS_PAPER_DIR`` point every path
at that folder, so the project's own ``data/``, ``cache/``, ``runs/`` and
``paper/`` are never read or written.

What this protects: that every experiment starts, finishes, and writes the
tables and figures the paper is built from -- on all three shapes of dataset a
user can have (identifiers in every capture, in none, in some).

What it does not show: anything about detection.  The data is synthetic (see
``synth_cic.py``) and no number produced here is evidence of anything.

Takes about two and a half minutes.  Skip with ``pytest -k "not study_cli"``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from synth_cic import write_dataset  # noqa: E402

DDOS_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
FAST = ["--quick", "--quick-rows", "4000", "--seeds", "1", "--n-boot", "20", "--models", "dt",
        "--quiet"]


class Sandbox:
    """One synthetic dataset with its own cache, runs and paper folders."""

    def __init__(self, root: Path, distribution: str):
        self.root = root
        self.data = root / "data"
        if distribution == "mixed":
            # seven MachineLearningCSV files and one GeneratedLabelledFlows file:
            # what you get by downloading one zip and keeping one older file
            write_dataset(self.data, n_benign=600, attack_scale=150, distribution="mlcsv")
            write_dataset(self.data, n_benign=600, attack_scale=150, distribution="glf",
                          only=[DDOS_FILE])
        else:
            write_dataset(self.data, n_benign=600, attack_scale=150, distribution=distribution)
        self.env = {
            **os.environ,
            "NIDS_DATA_DIR": str(self.data),
            "NIDS_CACHE_DIR": str(root / "cache"),
            "NIDS_RUNS_DIR": str(root / "runs"),
            "NIDS_ARTIFACT_DIR": str(root / "saved_models"),
            "NIDS_PAPER_DIR": str(root / "paper"),
            "MPLBACKEND": "Agg",
        }

    def study(self, *argv: str, check: bool = True) -> subprocess.CompletedProcess:
        p = subprocess.run([sys.executable, str(ROOT / "src" / "study.py"), *argv],
                           cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=540)
        if check and p.returncode != 0:
            raise AssertionError(f"study.py {' '.join(argv)} exited {p.returncode}\n"
                                 f"--- stdout ---\n{p.stdout[-3000:]}\n"
                                 f"--- stderr ---\n{p.stderr[-3000:]}")
        return p

    def latest(self, name: str) -> Path:
        runs = sorted((self.root / "runs").glob(f"*-study-{name}*"))
        assert runs, f"no run directory for {name}"
        return runs[-1]

    def table(self, name: str, table: str) -> pd.DataFrame:
        return pd.read_csv(self.latest(name) / f"{table}.csv")

    def results(self, name: str) -> dict:
        return json.loads((self.latest(name) / "results.json").read_text())


@pytest.fixture(scope="module")
def glf(tmp_path_factory) -> Sandbox:
    sb = Sandbox(tmp_path_factory.mktemp("study_glf"), "glf")
    sb.study("doctor", "--quiet")
    return sb


@pytest.fixture(scope="module")
def mlcsv(tmp_path_factory) -> Sandbox:
    sb = Sandbox(tmp_path_factory.mktemp("study_mlcsv"), "mlcsv")
    sb.study("doctor", "--quiet")
    return sb


# =====================================================================
# DOCTOR
# =====================================================================


def test_study_cli_doctor_reports_a_complete_dataset(glf):
    v = glf.results("doctor")["verdict"]
    assert v["all_files_present"] and v["identifiers_in_every_capture"]
    assert v["problems"] == [] and v["cannot_run"] == []
    caps = json.loads((glf.latest("doctor") / "doctor.json").read_text())["captures"]
    assert len(caps) == 8
    thu = next(c for c in caps if "Morning-WebAttacks" in c["file"])
    assert thu["encoding"].startswith("cp1252") and thu["blank_rows_dropped"] == 40
    assert all(c["timestamp_parse_rate"] == 1.0 and c["timestamp_on_expected_date"] == 1.0
               for c in caps)


def test_study_cli_doctor_prints_the_hours_each_capture_parses_onto(glf):
    """The timestamps have no AM/PM marker and an hour below 8 is read as
    afternoon.  The only way to see that this reading is right for a file is
    to look at the hours it produced, so `doctor` prints them."""
    out = glf.study("doctor", "--quiet").stdout
    ranges = re.findall(r"parsed times (\d\d):\d\d-(\d\d):\d\d", out)
    assert len(ranges) == 8
    assert all(9 <= int(first) and int(last) <= 17 for first, last in ranges)
    assert out.count("of timestamps parse") == 8
    caps = json.loads((glf.latest("doctor") / "doctor.json").read_text())["captures"]
    assert all(re.fullmatch(r"\d\d:\d\d-\d\d:\d\d", c["time_range"]) for c in caps)


def test_study_cli_doctor_raises_a_capture_whose_hours_fall_outside_the_working_day(tmp_path):
    """A capture stamped 7:xx is read as 19:xx.  The capture is documented as
    09:00-17:00, so that is either a wrong reading of the clock or a file that
    is not what it claims to be.  Either way it must be said out loud."""
    sb = Sandbox(tmp_path, "glf")
    f = sb.data / DDOS_FILE
    text = f.read_text(encoding="utf-8")
    changed, n = re.subn(r"(0?7/0?7/2017 )0?3:", r"\g<1>7:", text)
    assert n > 0
    f.write_text(changed, encoding="utf-8")
    out = sb.study("doctor", "--quiet").stdout
    problems = sb.results("doctor")["verdict"]["problems"]
    hit = [p for p in problems if DDOS_FILE in p and "12-hour clock" in p]
    assert len(hit) == 1 and "[problem]" in out


def test_study_cli_doctor_says_what_a_dataset_without_identifiers_cannot_do(mlcsv):
    v = mlcsv.results("doctor")["verdict"]
    assert v["all_files_present"] and not v["identifiers_in_every_capture"]
    assert "e4" in v["cannot_run"] and "e1" in v["can_run"]


def test_study_cli_doctor_flags_a_mixture_of_the_two_downloads(tmp_path):
    sb = Sandbox(tmp_path, "mixed")
    out = sb.study("doctor", "--quiet").stdout
    v = sb.results("doctor")["verdict"]
    assert not v["identifiers_in_every_capture"]
    assert any("MIXED DISTRIBUTIONS" in p and "1 of the 8" in p for p in v["problems"])
    assert "GeneratedLabelledFlows.zip" in out and "TrafficLabelling" in out
    assert "REPLACE the ones in data/" in out


def test_study_cli_doctor_names_a_missing_file(tmp_path):
    sb = Sandbox(tmp_path, "glf")
    (sb.data / DDOS_FILE).unlink()
    sb.study("doctor", "--quiet")
    v = sb.results("doctor")["verdict"]
    assert not v["all_files_present"] and f"missing: {DDOS_FILE}" in v["problems"]


def _resave_like_a_spreadsheet(path: Path) -> int:
    """Rewrite every timestamp of a capture the way the Friday DDoS file on
    the project's own laptop looked on 2026-10-04: ``07-07-2017 03:30``
    (dashes, zero-padded), where the dataset prints ``7/7/2017 3:30``."""
    text = path.read_text(encoding="utf-8")

    def dashed(m):
        return (f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{m.group(3)} "
                f"{int(m.group(4)):02d}:{m.group(5)}")

    changed, n = re.subn(r"\b(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2})(?::\d{2})?",
                         dashed, text)
    path.write_text(changed, encoding="utf-8")
    return n


def test_study_cli_doctor_says_a_timestamp_column_it_cannot_read_is_unreadable(tmp_path):
    """Seen on real files.  The capture HAS a Timestamp column; none of it
    parses.  `doctor` used to print `time=NO`, which reads as "the column is
    missing" and sends the reader looking for the wrong fault.  It must say
    the column is there and unreadable, show a value, and say once what is
    wrong -- not also that 100% of the timestamps are on the wrong day."""
    sb = Sandbox(tmp_path, "glf")
    assert _resave_like_a_spreadsheet(sb.data / DDOS_FILE) > 0
    out = sb.study("doctor", "--quiet").stdout
    v = sb.results("doctor")["verdict"]

    line = next(ln for ln in out.splitlines() if "Afternoon-DDos" in ln and "cols" in ln)
    assert "src-ip=yes" in line and "time=UNREADABLE" in line
    mine = [p for p in v["problems"] if DDOS_FILE in p]
    assert len(mine) == 1, mine
    assert "0.0% of timestamps can be read" in mine[0] and "'07-07-2017 " in mine[0]
    assert "spreadsheet" in mine[0] and "not the original" in mine[0]
    assert not any("are not on 2017-07-07" in p for p in v["problems"])

    # every capture HAS the columns, so this is not a mixture of two downloads
    assert not any("MIXED DISTRIBUTIONS" in p for p in v["problems"])
    assert "every capture has the two columns" in out
    # ... but they cannot be used, and the verdict must not promise E4
    assert not v["identifiers_in_every_capture"] and "e4" in v["cannot_run"]
    cap = next(c for c in json.loads((sb.latest("doctor") / "doctor.json").read_text())["captures"]
               if c["file"] == DDOS_FILE)
    assert cap["has_timestamp"] and not cap["time_usable"] and not cap["identifiers_usable"]
    assert cap["timestamp_unparsed_examples"][0].startswith("07-07-2017 ")


def test_study_cli_all_stops_when_doctor_lists_a_problem(tmp_path):
    """Experiments on files with a known fault are hours that cannot be used,
    and their tables would land in paper/ looking like results.  `all` stops
    after `doctor`, before the first experiment."""
    sb = Sandbox(tmp_path, "glf")
    _resave_like_a_spreadsheet(sb.data / DDOS_FILE)
    p = sb.study("all", *FAST, "--only", "e6", check=False)
    assert p.returncode == 2, p.stdout[-1500:]
    assert "`all` stopped before any experiment" in p.stdout
    assert "--despite-problems" in p.stdout
    assert not list((sb.root / "runs").glob("*-study-e6*"))
    assert not (sb.root / "paper").exists()


def test_study_cli_all_can_be_told_to_run_despite_a_problem_and_the_report_says_so(tmp_path):
    sb = Sandbox(tmp_path, "glf")
    _resave_like_a_spreadsheet(sb.data / DDOS_FILE)
    p = sb.study("all", *FAST, "--only", "e6", "--despite-problems")
    assert "--despite-problems: `doctor` listed 1 problem(s)" in p.stdout
    assert list((sb.root / "runs").glob("*-study-e6*"))
    sources = (sb.root / "paper" / "quick" / "SOURCES.md").read_text(encoding="utf-8")
    assert "## What `doctor` said about the CSVs" in sources
    assert "- problems listed: 1" in sources and "can be read" in sources
    assert "`doctor` listed a problem with the files" in sources
    assert "DATASET: the newest `doctor` run listed 1 problem(s)" in p.stdout


# =====================================================================
# THE SIX EXPERIMENTS, IDENTIFIERS PRESENT
# =====================================================================


def test_study_cli_e1_compares_the_split_protocols(glf):
    glf.study("e1", *FAST)
    det, summ = glf.table("e1", "e1_detection"), glf.table("e1", "e1_summary")
    assert set(summ["protocol"]) == {"random", "blocked", "crossday"}
    assert set(det["score_rule"]) == {"max attack probability", "1 - P(benign)"}
    assert set(det["budget"]) == {0.001, 0.005, 0.01}
    # crossday tests on Friday, whose attack classes were never trained on;
    # the random split has seen every class it is tested on
    by = summ.set_index("protocol")["unseen_classes_in_test"]
    assert by["crossday"] >= 2 and by["random"] == 0 and by["blocked"] == 0
    # On the cross-day test split BENIGN is the only known class present, so
    # macro-F1 over known classes is a one-class number there.  The table says
    # so, and carries an attack-versus-benign F1 that is comparable everywhere.
    known = summ.set_index("protocol")["known_classes_in_test"]
    assert known["crossday"] == 1 and known["random"] > 5
    assert {"attack_f1_argmax", "attack_precision_argmax", "attack_recall_argmax",
            "macro_f1_known_rows"} <= set(summ.columns)
    assert summ["attack_f1_argmax"].between(0, 1).all()
    assert "n_blocks" in det.columns and (det["n_blocks"] >= 1).all()
    # the reading rule travels with every interval: fewer than 20 blocks, not interpreted
    assert (det["interval_counts"] == (det["n_blocks"] >= 20)).all()
    assert (glf.latest("e1") / "e1_protocol_effect.png").exists()
    assert glf.results("e1")["quick"] is True and glf.results("e1")["seeds_used"] == [42]


def test_study_cli_e2_holds_each_class_out_in_turn(glf):
    glf.study("e2", *FAST, "--classes", "PortScan", "DDoS")
    facts, cmp_ = glf.table("e2", "e2_class_facts"), glf.table("e2", "e2_seen_unseen")
    assert {"unique_vectors", "benign_collision_share", "seen_before_share",
            "benign_cost_flows", "benign_cost_share"} <= set(facts.columns)
    assert facts["benign_cost_share"].between(0, 1).all()
    assert any(c.startswith("collides_with_benign_") for c in facts.columns)
    unseen = glf.table("e2", "e2_unseen_detection")
    assert set(unseen["class"]) == {"PortScan", "DDoS"} and set(unseen["condition"]) == {"unseen"}
    held = cmp_[cmp_["class"].isin(["PortScan", "DDoS"])]
    assert held["det_seen"].notna().all() and held["det_unseen"].notna().all()
    assert {"gap", "pred_benign_share", "top_predicted"} <= set(cmp_.columns)
    # each of the two rates has its own interval, so each carries its own block count
    assert {"n_blocks_seen", "n_blocks_unseen", "interval_counts_seen",
            "interval_counts_unseen"} <= set(cmp_.columns)
    assert held["n_blocks_seen"].notna().all() and held["n_blocks_unseen"].notna().all()
    # one seed, and the tables say so themselves
    assert set(cmp_["seed"]) == {42} and set(unseen["seed"]) == {42}
    assert set(glf.table("e2", "e2_seen_detection")["seed"]) == {42}
    assert glf.results("e2")["seeds_used"] == [42]
    assert (glf.latest("e2") / "e2_seen_unseen.png").exists()


def test_study_cli_e3_scores_benign_only_detectors(glf):
    glf.study("e3", *FAST, "--detectors", "pca", "iforest", "mahalanobis")
    det, auc = glf.table("e3", "e3_detection"), glf.table("e3", "e3_auc")
    assert set(det["protocol"]) == {"crossday", "blocked"}
    assert set(det["detector"]) == {"dt (supervised)", "pca", "iforest", "mahalanobis"}
    assert {"auroc", "pauc@0.01"} <= set(auc.columns)
    assert auc["auroc"].between(0, 1).all() and "fit_seconds" in auc.columns
    # no deep detector was asked for, so there is nothing to compare with the classical ones
    assert not (glf.latest("e3") / "e3_deep_vs_classical.csv").exists()
    assert not (glf.latest("e3") / "e3_rq3_verdict.csv").exists()
    assert glf.results("e3")["rq3_supported"] == "not evaluated"
    for protocol in ("crossday", "blocked"):
        assert (glf.latest("e3") / f"e3_auroc_{protocol}.png").exists()


def test_study_cli_e4_measures_the_behaviour_layer(glf):
    glf.study("e4", *FAST)
    for name in ("e4_deployed_coverage", "e4_flagged_source_windows", "e4_alerts",
                 "e4_alert_summary", "e4_online_coverage", "e4_calibrated",
                 "e4_threshold_sweep", "e4_window_length", "e4_top_source_windows"):
        assert len(glf.table("e4", name)), name
    cal = glf.table("e4", "e4_calibrated")
    assert set(cal["calibration_set"]) == {"monday (benign-only day)", "thursday benign as labelled",
                                           "all earlier days, benign as labelled"}
    # the direction repair is measured, not assumed: both settings are in the table
    assert set(glf.table("e4", "e4_deployed_coverage")["canonical"]) == {True, False}
    # the synthetic Thursday scan is labelled BENIGN, and must surface as such
    alerts = glf.table("e4", "e4_alerts")
    assert ((alerts["day"] == "Thursday") & (alerts["attack_share"] == 0.0)).any()
    assert (glf.latest("e4") / "e4_threshold_sweep.png").exists()


def test_study_cli_e5_fuses_three_layers_under_one_budget(glf):
    glf.study("e5", *FAST, "--novelty", "pca")
    res = glf.results("e5")
    assert res["behaviour_layer"] is True
    assert res["conditions"] == ["as labelled", "scan-like benign flows removed"]
    fus = glf.table("e5", "e5_fusion")
    assert set(fus["rule"]) == {"min-p", "bonferroni"}
    assert fus["layers"].nunique() == 7 and set(fus["n_layers"]) == {1, 2, 3}
    attr = glf.table("e5", "e5_attribution")
    share = attr.groupby(["condition", "seed", "budget", "class"])["share"].sum()
    assert ((share - 1.0).abs() < 1e-9).all()
    gain = glf.table("e5", "e5_gain")
    assert set(gain["system"]) == {"classifier+novelty+behaviour"}
    assert gain["compared_with"].nunique() == 6                 # every smaller system
    assert "BENIGN" in set(gain["class"])                       # false alarms are compared too
    assert (gain["diff"] - (gain["rate_system"] - gain["rate_compared"])).abs().max() < 1e-12
    assert set(gain["score_rule"]) == {"max attack probability"}
    assert (gain["differs"] <= gain["interval_counts"]).all()      # never "differs" on few blocks

    # The same comparison with the classifier read through the other score
    # rule: same shape, and identical wherever the classifier is not involved.
    alt = glf.table("e5", "e5_gain_other_score_rule")
    assert set(alt["score_rule"]) == {"1 - P(benign)"} and len(alt) == len(gain)
    key = ["condition", "seed", "budget", "class", "compared_with"]
    both = gain.merge(alt, on=key, suffixes=("", "_alt"))
    assert len(both) == len(gain)
    no_clf = ~both["compared_with"].str.contains("classifier")
    assert (both.loc[no_clf, "rate_compared"] == both.loc[no_clf, "rate_compared_alt"]).all()
    alt_fus = glf.table("e5", "e5_fusion_other_score_rule")
    assert set(alt_fus["rule"]) == {"min-p", "bonferroni"} and alt_fus["layers"].nunique() == 4
    assert alt_fus["layers"].str.contains("classifier").all()
    # "which layer caught what" is answered under both rules too
    assert set(attr["score_rule"]) == {"max attack probability"}
    alt_attr = glf.table("e5", "e5_attribution_other_score_rule")
    # (the two tables need not have the same number of rows: a combination of
    # layers that fires on nothing under one rule can fire under the other)
    assert set(alt_attr["score_rule"]) == {"1 - P(benign)"} and len(alt_attr) > 0
    alt_share = alt_attr.groupby(["condition", "seed", "budget", "class"])["share"].sum()
    assert ((alt_share - 1.0).abs() < 1e-9).all()

    # the protocol's rule for RQ5, applied by the code, under both score rules
    verdict = glf.table("e5", "e5_rq5_verdict")
    assert set(verdict["score_rule"]) == {"max attack probability", "1 - P(benign)"}
    assert set(verdict["single_layer"]) == {"classifier", "novelty", "behaviour"}
    assert verdict["complete"].all() and verdict["layers_complete"].all()
    assert (verdict["false_alarms_ok"]
            == (verdict["within_budget"] | verdict["excess_not_shown"])).all()
    assert (verdict["passes"] == (verdict["more_detection"] & verdict["false_alarms_ok"])).all()
    assert (verdict["passes_strict"] == (verdict["more_detection"] & verdict["no_excess"])).all()
    per = verdict.groupby(["score_rule", "condition", "seed"])
    assert (per["fusion_helps"].nunique() == 1).all()
    assert (per["passes"].all() == per["fusion_helps"].first()).all()
    assert res["rq5_supported"] in (True, False)
    assert res["rq5_supported_other_score_rule"] in (True, False)
    assert res["rq5_supported_strict"] in (True, False)
    assert res["seeds_used"] == [42] and res["split_seed"] == 42
    summ = glf.table("e5", "e5_summary")
    assert {"precision_at_0.1pct_prevalence", "precision_at_1pct_prevalence"} <= set(summ.columns)
    cond = glf.table("e5", "e5_conditions").set_index("condition")
    assert cond.loc["scan-like benign flows removed", "sanitise_removed_train"] > 0
    assert cond.loc["scan-like benign flows removed", "n_train"] < cond.loc["as labelled", "n_train"]
    assert (glf.latest("e5") / "e5_layer_attribution.png").exists()


def test_study_cli_e6_compares_ways_of_fitting_the_threshold(glf):
    glf.study("e6", *FAST)
    t = glf.table("e6", "e6_threshold_transfer")
    assert set(t["strategy"]) == {
        "A random half of Thursday", "A2 half of the validation flows, uncalibrated",
        "E the same flows, per-class isotonic recalibration", "B later part of Thursday",
        "C leave-one-day-out", "D oracle (test-day benign)"}
    assert set(t["score_rule"]) == {"max attack probability", "1 - P(benign)"}
    # the oracle is fitted on the test day's own benign scores, so by the
    # threshold's guarantee it cannot exceed the budget there
    oracle = t[t["strategy"].str.startswith("D oracle")]
    assert (oracle["observed_fpr"] <= oracle["budget"] + 1e-12).all()
    assert "fpr_ratio" in t.columns
    # recalibration is compared with no recalibration on the same model and the
    # same reference flows, false alarms included
    cal = glf.table("e6", "e6_calibration_effect")
    assert {"BENIGN", "__ANY_ATTACK__"} <= set(cal["class"])
    assert (cal["diff"] - (cal["rate_recalibrated"] - cal["rate_uncalibrated"])).abs().max() < 1e-12
    assert {"interval_counts", "differs"} <= set(cal.columns)
    assert set(cal["seed"]) == {42} and set(t["seed"]) == {42}
    # the protocol's rule for the recalibration claim, applied by the code
    verdict = glf.table("e6", "e6_rq6_verdict")
    assert set(verdict["model"]) == {"dt"} and set(verdict["budget"]) == {0.01}
    assert set(verdict["score_rule"]) == {"max attack probability", "1 - P(benign)"}
    assert {"lowers_false_alarms", "over_budget_before", "within_budget_after", "repairs",
            "improves", "detection_diff"} <= set(verdict.columns)
    # a repair needs something broken first; an improvement is lower but still over budget
    assert (verdict["repairs"] <= (verdict["over_budget_before"] & verdict["lowers_false_alarms"]
                                   & verdict["within_budget_after"])).all()
    assert not (verdict["repairs"] & verdict["improves"]).any()
    r6 = glf.results("e6")
    assert r6["rq6_recalibration_repairs"] in (True, False)
    assert r6["rq6_recalibration_improves"] in (True, False)
    assert r6["rq6_recalibration_lowers_false_alarms"] in (True, False)
    assert glf.results("e6")["seeds_used"] == [42]
    assert len(glf.table("e6", "e6_benign_score_quantiles")) >= 8
    assert (glf.latest("e6") / "e6_threshold_transfer.png").exists()
    assert "held out" in (glf.latest("e6") / "log.txt").read_text()   # leave-one-day-out ran


def _study_files(folder: Path):
    """Tables and figures of the study in ``folder`` (they are named eN_...)."""
    return sorted(f.name for f in folder.glob("e[1-6]_*")) if folder.is_dir() else []


def test_study_cli_report_ignores_quick_runs_unless_asked(glf):
    """Runs after the experiments above (same module, same sandbox).  Every run
    so far was a quick one, so the results folders must stay empty."""
    paper = glf.root / "paper"
    glf.study("report")
    sources = (paper / "SOURCES.md").read_text()
    assert "*not run*" in sources and "study-e1" not in sources     # quick runs are not results
    assert _study_files(paper / "tables") == [] and _study_files(paper / "figures") == []

    out = glf.study("report", "--allow-quick").stdout
    assert "QUICK RUNS" in out and "were not touched" in out
    quick = paper / "quick"
    qsrc = (quick / "SOURCES.md").read_text()
    assert qsrc.startswith("# QUICK RUNS") and "study-e1-quick" in qsrc and "| YES |" in qsrc
    assert "| `tables/e1_detection.csv` | `runs/" in qsrc           # every file names its run
    assert (quick / "tables" / "e1_detection.csv").exists()
    assert (quick / "tables" / "e1_summary.md").exists()
    assert (quick / "tables" / "e5_rq5_verdict.csv").exists()
    assert list((quick / "figures").glob("*.pdf"))
    # ... and the results folders are exactly as the plain report left them
    assert _study_files(paper / "tables") == [] and _study_files(paper / "figures") == []
    assert (paper / "SOURCES.md").read_text() == sources


def test_study_cli_all_runs_each_experiment_in_its_own_process_and_reports(glf):
    out = glf.study("all", *FAST, "--only", "e4", "e6").stdout
    assert ">>> python src/study.py e4 --quick" in out
    assert ">>> python src/study.py e6 --quick" in out and "study.py e1" not in out
    assert "e4: ok" in out and "e6: ok" in out
    # a quick `all` reports into paper/quick/ and nowhere else
    sources = (glf.root / "paper" / "quick" / "SOURCES.md").read_text()
    assert "## Last `study.py all`" in sources and "| e4 | ok |" in sources
    # the report repeats what `doctor` said about the files the runs read
    assert "## What `doctor` said about the CSVs" in sources
    assert "- problems listed: 0" in sources
    assert "- source IP and timestamp usable in every capture: True" in sources
    assert "`doctor` listed a problem with the files" not in sources
    assert _study_files(glf.root / "paper" / "tables") == []


def test_study_cli_all_survives_an_experiment_that_fails(glf):
    """An experiment that dies must not take the queue behind it down: the
    summary names it, the others still run, and the exit code says so."""
    p = glf.study("all", *FAST[:-3], "--models", "no_such_model", "--quiet",
                  "--only", "e1", "e4", check=False)
    assert p.returncode == 1
    assert "e1: FAILED" in p.stdout and "e4: ok" in p.stdout
    assert "python src/study.py all --only e1" in p.stdout
    sources = (glf.root / "paper" / "quick" / "SOURCES.md").read_text()
    assert "| e1 | FAILED" in sources


# =====================================================================
# THE DEFAULT MODELS  (needs torch and xgboost; skipped without them)
# =====================================================================


def test_study_cli_default_models_run_and_the_table_that_decides_rq3_is_written(glf):
    """Everything above uses a decision tree and PCA so that it runs without
    torch.  The paper's experiments use XGBoost, an autoencoder and Deep SVDD.
    This runs those paths once -- few epochs, quick mode -- and checks the
    tables the decision rules of RQ3 and RQ5 are read from."""
    pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    real = ["--quick", "--quick-rows", "4000", "--seeds", "2", "--n-boot", "20",
            "--epochs", "2", "--models", "xgb", "--quiet"]

    glf.study("e3", *real, "--detectors", "autoencoder", "deep_svdd", "pca", "--protocols",
              "crossday")
    pairs = glf.table("e3", "e3_deep_vs_classical")
    assert set(pairs["deep"]) == {"autoencoder", "deep_svdd"} and set(pairs["baseline"]) == {"pca"}
    assert {"diff", "diff_lo", "diff_hi", "n_blocks", "interval_counts", "differs"} <= set(pairs.columns)
    # RQ3's rule is applied to the autoencoder only, one row per test-day attack class
    verdict = glf.table("e3", "e3_rq3_verdict")
    assert set(verdict["deep"]) == {"autoencoder"} and set(verdict["baselines"]) == {1}
    assert "BENIGN" not in set(verdict["class"]) and "__ANY_ATTACK__" not in set(verdict["class"])
    assert verdict["deep_false_alarm_rate"].between(0, 1).all()
    # Only PCA was run here.  "Every classical baseline" means the three the
    # protocol names, so no row may pass and the question is not evaluated.
    assert not verdict["baselines_complete"].any() and not verdict["passes"].any()
    assert glf.results("e3")["rq3_supported"] == "not evaluated"
    assert "RQ3 is not evaluated" in (glf.latest("e3") / "log.txt").read_text()
    summ = glf.table("e3", "e3_summary")
    # every detector, the supervised reference and PCA included, once per seed
    assert set(summ["n_seeds"]) == {2}

    glf.study("e5", *real)                                   # default novelty layer: autoencoder
    res = glf.results("e5")
    assert res["novelty"] == "autoencoder" and res["classifier"] == "xgb"
    s5 = glf.table("e5", "e5_summary")
    assert set(s5["n_seeds"]) == {2}
    gain = glf.table("e5", "e5_gain")
    assert set(gain["seed"]) == {42, 43} and "n_blocks" in gain.columns


# =====================================================================
# NO IDENTIFIERS: WHAT STILL RUNS, AND WHAT SAYS WHY IT CANNOT
# =====================================================================


def test_study_cli_e4_refuses_without_identifiers_and_says_what_to_download(mlcsv):
    p = mlcsv.study("e4", *FAST, check=False)
    assert p.returncode != 0
    msg = p.stdout + p.stderr
    assert "GeneratedLabelledFlows" in msg and "cache --rebuild" in msg


def test_study_cli_all_skips_the_behaviour_experiment_without_identifiers(mlcsv):
    out = mlcsv.study("all", *FAST, "--only", "e4").stdout
    assert "e4: skipped: no source IP / timestamp" in out and "study.py e4" not in out


def test_study_cli_e5_runs_with_two_layers_when_there_are_no_identifiers(mlcsv):
    out = mlcsv.study("e5", *FAST, "--novelty", "pca").stdout
    res = mlcsv.results("e5")
    assert res["behaviour_layer"] is False and res["conditions"] == ["as labelled"]
    assert "behaviour layer is left out" in out
    assert mlcsv.table("e5", "e5_fusion")["layers"].nunique() == 3
    # RQ5 is a question about the three-layer system: with two layers it is not answered
    assert res["rq5_supported"] == "not evaluated" and "RQ5" in out and "not evaluated" in out
    assert set(mlcsv.table("e5", "e5_rq5_verdict")["single_layer"]) == {"classifier", "novelty"}
