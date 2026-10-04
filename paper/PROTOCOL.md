# Study protocol

**Written 2026-10-04: after one baseline run on the real data, and before any of
the six experiments E1-E6 was run. Revised 2026-10-05, still before any of them
was run: the changes are listed under "Revisions before the first run" at the
bottom.**

This file fixes what the six experiments measure and how, so that their results
cannot shape the method. If anything below has to change after one of them has
run, do not edit the text: add a dated line to **Deviations** at the bottom
saying what changed and why. Commit this file before running
`python src/study.py all`, and push the commit: the commit is then evidence of
the order (a local commit date can be set by hand; a pushed one cannot).

**What was already known when this was written.** One cross-day XGBoost run
existed (`runs/f853071-20260912-034653-xgb-crossday`, 12 September 2026). Its
results are the reason for RQ2, RQ5 and RQ6, so this plan is not blind to the
test day: it was written knowing that DDoS is largely detected, that PortScan
and Bot are not, and that the 1% budget landed at 4.38%. What is fixed in
advance is how the six experiments are analysed, not the questions.

Nothing in this file is a result. Every number in the paper must come from a run
directory listed in `paper/SOURCES.md`.

The three yes/no rules of section 5 (RQ3, RQ5, RQ6) are also written as code,
in `src/rules.py`. Each experiment applies its rule itself and writes the
verdict as a table, so an answer is computed and not judged from a table by
eye.

---

## 1. What the paper is

An **evaluation paper**, not a new architecture. Stacking a classifier, an
anomaly detector and a window counter is not new (see Related work in the
draft). What is missing in the literature is a measurement of *which layer
catches which attack, at one shared false-alarm budget, on a day the system has
never seen*. That measurement is the contribution.

The system under test has three layers:

| layer | what it is | what it has seen |
|---|---|---|
| L1 classifier | XGBoost over per-flow features, score = max P(attack class) | labelled flows of the training days |
| L2 novelty | autoencoder reconstruction error | BENIGN flows of the training days only |
| L3 behaviour | distinct destination ports / hosts per source per minute | nothing: it has no training |

## 2. Data

- **CIC-IDS2017**, the `GeneratedLabelledFlows` CSVs (they carry source IP and
  timestamp; the `MachineLearningCSV` set does not). `python src/study.py
  doctor` records which one is on disk.
- The eight files are used as they come out of the zip. `doctor` lists as a
  problem a missing capture, a mixture of the two downloads, and timestamps
  that cannot be read, that fall on another date than the capture's, or that
  parse outside 08:00-18:00. `python src/study.py all` runs `doctor` first and
  stops before the first experiment if it lists a problem (section 4,
  "problems with the files").
- Labels are used **as shipped**. The corrected labels of Engelen et al. (2021)
  and Liu et al. (2022) are regenerated flows and cannot be joined to these
  rows (for Liu et al. this is taken from their abstract and code repository;
  their documentation page could not be opened). Label noise is studied
  through E4 and the E5 ablation instead.
- Features: the flow statistics only. `Destination Port`, `Flow ID`, both
  addresses, `Source Port`, `Protocol` and `Timestamp` are never features
  (shortcut learning). Seven exactly collinear columns are dropped. Columns
  that are constant, or more than half missing, **in the training split** are
  dropped.
- Rows removed from the whole week before any split, test days included: a
  `Flow Duration` that is negative or not a number, and a blank label.
  Three more things remove rows, and only these. Before it splits, the
  `random` protocol drops a class with fewer than 10 flows, so that class is in
  none of its three splits (10 is a round floor taken over from the existing
  pipeline: a judgement call). After the split, a validation split does not
  keep attack rows of a class that is absent from its training split
  (validation only; benign rows and the test split are never affected). And
  the E5 ablation edits training and validation (section 5, RQ5).
- De-duplication: **training split only**, on feature columns, count reported,
  in all four protocols below. Validation and test are raw flows.

## 3. Split protocols

| name | train | validation | test | used in |
|---|---|---|---|---|
| `random` | 70% of all five days, pooled and split at random | 15% | 15% | E1 (the leaky baseline) |
| `blocked` | earliest 60% of each (capture, class) | next 20% | last 20% | E1, E2 seen, E3 |
| `loco` | `blocked` with one attack class removed | same | + every flow of that class | E2 unseen |
| `crossday` | Mon, Tue, Wed + half of Thursday | other half of Thursday | Friday | E1, E3, E5, E6 |

Details that matter when reading the tables:

- `random` drops any class with fewer than 10 flows before it splits (a round
  floor, so that each of the three parts gets a few).
- `blocked` sends a (capture, class) group with fewer than 3 flows wholly to
  training. Within a class no test flow precedes a training flow of the same
  capture; flows of *different* classes are still interleaved in time.
- `crossday` is the primary protocol. Nothing in E1-E6 is chosen with Friday:
  no threshold, model, hyper-parameter, layer or score rule. Friday is scored
  once per fitted model, by several experiments, so it is reused; and it was
  looked at before this plan existed (see section 7).

## 4. Fixed in advance

| choice | value |
|---|---|
| false-alarm budgets | 0.1%, 0.5%, 1% of benign flows |
| **primary budget** | **1%** |
| threshold | lowest validation-benign score leaving at most the budget above it (`stats.budget_threshold`); a flow fires if strictly above |
| primary classifier | XGBoost, project settings (`config.XGB_PARAMS`), no tuning |
| second classifier | Random Forest (`config.RF_PARAMS`), no tuning |
| primary classifier score | max P(attack class); `1 - P(benign)` reported beside it in E1, E2 and E6, and E5 repeats its fusion comparison under it |
| primary novelty layer | autoencoder: `TrainConfig` defaults, except up to 30 epochs and patience 10; all benign training rows |
| novelty baselines | Deep SVDD (all benign training rows); Isolation Forest and PCA (random subsample of at most 500,000 benign rows); Mahalanobis (covariance from a random 200,000) |
| behaviour window | 60 s, tumbling, aligned to the timestamp |
| behaviour thresholds | 100 distinct ports or 50 distinct hosts in 60 s: the live tracker's defaults (a test pins the study's copy of the three numbers to the tracker's). A source at or above either is flagged; that is the "deployed" operating point of E4. In every budget-fitted table (E4's, E5) the behaviour score is thresholded on validation like any other layer, and 100 and 50 only set the scale between ports and hosts |
| direction repair | swap endpoints when source port < 1024 <= destination port. E4's deployed-coverage tables report the unrepaired result beside it; everything else (the replay, the budget-fitted tables, E5) uses the repaired endpoints only |
| primary fusion rule | min-p (smallest empirical p-value across layers, one threshold) |
| second fusion rule | Bonferroni (each of k layers at budget / k) |
| seeds | E1, E3, E5: every model refitted with seeds 42, 43, 44 (the split in E3 and E5 stays that of seed 42). E2 and E6: seed 42 only, because each already needs a dozen or more fits |
| intervals | block bootstrap, 1,000 replicates, block = one minute of one capture, 95% percentile |
| when an interval counts | only for a class spread over at least 20 blocks; below that it is printed and not interpreted. Every detection table and every paired table carries `n_blocks` and a true/false column `interval_counts`, written by the code (`stats.interval_counts`). In `e2_seen_unseen` the two columns are suffixed `_seen` and `_unseen`; the false-alarm interval has its own count, `n_blocks_benign`. The verdict tables fold the rule into their pass columns |
| "A beats B" | paired block bootstrap of the difference on the same test rows. Supported when (1) the 95% interval lies above zero and the class spans at least 20 blocks, **and** (2) A's false alarms on the test day are acceptable: A is at or under the budget there (`within_budget`), **or** A is not shown to raise more false alarms than B: the paired interval of the BENIGN difference does not lie above zero (`excess_not_shown`). Applied by `src/rules.py` |
| why two ways to pass (2) | "not more than B" alone punishes A for using a budget that B leaves unused: a layer whose score is a count ties heavily and can fire far below its budget. "Within the budget" alone can never be met on a day when every threshold drifts above its budget, which the baseline run showed. A gain that needs more false alarms than the budget allows **and** clearly more than the detector it is compared with is refused |
| what rule (2) does not do | it is asymmetric: detection must be *shown* to be higher, false alarms only *not shown* to be higher. An excess that is large but noisy passes it. The two verdicts that use this rule (`e3_rq3_verdict`, `e5_rq5_verdict`) therefore also carry the strict reading, `passes_strict`: more detection, and A's observed false-alarm rate is not above B's at all, whatever the budget. Both are reported; where they disagree the paper says so. (RQ6's verdict compares false-alarm rates directly and has no such column) |
| an incomplete table | never passes. A comparison whose rows are missing (a baseline that was not run, a layer without its false-alarm row) is a failed row or "not evaluated", never skipped |
| quick runs | never results. Plain `report` ignores them; `all --quick` and `report --allow-quick` write to `paper/quick/` and never to `paper/tables` or `paper/figures` |
| what `report` puts in `paper/` | it works experiment by experiment. For an experiment it has a run for, it removes the files its previous report wrote for that experiment (it keeps the list in `.report_files.json`) and copies the new ones. For an experiment it has no run folder for, it touches nothing and lists the files as "kept from an earlier report". A file it did not write is never deleted; such files are named at the end of `SOURCES.md` as not covered. `SOURCES.md` lists every file the report vouches for with the run it came from |
| runs with other settings | the command line's defaults are this plan's values. For each experiment `report` takes the newest full run **that follows the plan**; a newer run with other settings does not replace it and is named as passed over. Only when no run of an experiment follows the plan is the newest one taken, and `SOURCES.md` then names it "not run as planned" and lists the settings that differ. "Settings" means the ones that experiment reads, resolved to the values the code uses (`study.plan_settings`): `--models` cannot change E4, and `--epochs 30` is the same setting as no `--epochs`. A run whose command line was not recorded counts as not following the plan. A deviation line below is required before an off-plan run's tables are quoted |
| problems with the files | `all` stops before the first experiment when `doctor` lists a problem (exit code 2). `--despite-problems` runs anyway. Either way `SOURCES.md` repeats what the newest `doctor` run said about the CSVs, next to the list of tables. Running despite a problem needs a deviation line below |
| when a capture "has identifiers" | at least 99% of its rows carry a source IP and a timestamp that parses (`config.IDENTIFIER_COVERAGE`; a round floor, a judgement call). The same floor decides whether a split is ordered by time and whether the bootstrap blocks are minutes. A dataset with identifiers in no capture is not a "problem": E4 is skipped and E5 runs with two layers, which leaves RQ4 unanswered and RQ5 "not evaluated" |
| precision at realistic prevalence | projected from the measured rates at 0.1% and 1% attack prevalence, any-attack only, labelled as a projection |
| hyper-parameter search | none |

## 5. Research questions and what would count as an answer

**RQ1. How much of the detection reported under a random split survives a
split without interleaving, and a split onto a new day?** (E1)
Expectation: random >= blocked >= crossday. This repeats earlier findings
(Engelen 2021; D'hooge 2022; Buriboyev 2026) on this code base. It is a
baseline for the other questions, not a claim of novelty. The column to compare
across protocols is the attack-versus-benign F1 of the arg-max and the per-class
detection at the budgets. Macro-F1 over known classes is a one-class number on
the cross-day test split and is not compared.

**RQ2. When a per-flow classifier misses an attack class on a new day, is that
because the class cannot be told from benign traffic, or because it was never
in training?** (E2)
Three explanations are in play for PortScan, the largest miss in the existing
run (0.23% detected at a 0.1% budget). Each predicts something different:

| | says | predicts |
|---|---|---|
| H-a | scan flows carry no per-flow signature | low detection even when PortScan is in training; scan flows share their feature vectors with a large share of benign flows, on every day |
| H-b | the class was simply unseen | high detection when seen (`blocked`), low when held out (`loco`) |
| H-c | scan traffic labelled BENIGN taught the model that scans are benign | benign collisions concentrated on one day; behaviour layer flags BENIGN-only sources on that day (E4); removing them raises unseen detection (E5 ablation) |

The answer is whichever pattern the tables show. More than one can hold. The
share of a class's flows that collide with a benign vector is read together
with the share of benign flows that carry those vectors: the first alone is
not a ceiling on detection.

**RQ3. Does a deep benign-only detector find unseen attacks that simple
benign-only detectors do not?** (E3)
Deep (autoencoder, Deep SVDD) against classical (Isolation Forest, PCA,
Mahalanobis), same budgets. "The deep model helps" is supported only if, under
the cross-day protocol, at the 1% budget and for at least one Friday class
that spans at least 20 blocks, the autoencoder detects more than **every**
classical baseline, with a paired interval of the difference that lies above
zero, **and** its false alarms on the test day are acceptable against every
one of them (section 4, "A beats B"). "Every classical baseline" is the three
named above; if one of them was not run the question is "not evaluated". Read
from `e3_deep_vs_classical` at the first seed; the spread over seeds is
reported beside it. The verdict is computed by `rules.rq3_verdict` and written
as `e3_rq3_verdict`, one row per class, with the strict reading beside it.
Otherwise the paper says it does not.

**RQ4. What does the label-free behaviour layer cover, and what does it
cost?** (E4)
Reported: share of each class's flows covered at the deployed thresholds;
alerts per day from replaying the week through the live tracker class; how
many of those alerts sit on traffic labelled BENIGN, on which day and from
which source; the same without the direction repair.

**RQ5. At one total false-alarm budget, does the three-layer system detect
more than its best single layer, and which layer accounts for which class?**
(E5)
Every subset of the layers is scored at the same total budget. "Fusion helps"
is supported only if, at the 1% budget under min-p fusion, with the labels as
shipped, at the first seed and under the primary score rule, the full system
detects more attack flows (any-attack row) than **each** single layer, with a
paired interval of the difference that lies above zero on at least 20 blocks,
**and** its false alarms on the test day are acceptable against that layer
(section 4, "A beats B"; the BENIGN row of `e5_gain`). The verdict is computed
by `rules.rq5_verdict` and written as `e5_rq5_verdict`.

Reported beside that verdict, never in place of it: the same verdict at seeds
43 and 44; under the ablation condition; under the second score rule; and the
strict reading (`passes_strict`: the system's observed false-alarm rate is not
above the layer's, whatever the budget). If the verdict under the second score
rule differs from the first, the paper reports RQ5 as not robust to the score
rule. RQ5 is a question about the three-layer system: if the behaviour layer
cannot run, it is "not evaluated".

Everything in E5 that involves the classifier is computed under both score
rules, with the same fitted models and nothing refitted: both fusion rules,
the gains and the layer attribution (tables `e5_*_other_score_rule`). The
attribution table (Bonferroni rule, full system) says, per class, which layers
fired. `e5_summary` and the figure are the primary rule's.

**RQ6. Does a threshold fitted before the test day deliver its budget on the
test day, and does it matter how the validation flows were drawn?** (E6)
Six rows per model and score rule:

| | calibration sample | same fitted model as A? |
|---|---|---|
| A | random half of Thursday | yes |
| A2 | a random half of A's flows (the reference for E) | yes |
| E | A2's flows after per-class isotonic recalibration fitted on the other half | yes |
| B | later part of each Thursday class | no: the training split differs |
| C | leave-one-day-out over the training days, scored as recorded | no: one model per held-out day |
| D | the test day's own benign flows (oracle, used for no claim) | yes |

Reported: observed false-alarm rate divided by the budget. "Recalibration
repairs the threshold" is supported only if, at the 1% budget, for XGBoost
under the primary score rule, all three hold on the test day: (a) the
uncalibrated threshold (A2) was over the budget; (b) the recalibrated one (E)
has a lower false-alarm rate, with a paired interval that lies below zero
(table `e6_calibration_effect`); (c) that lower rate is at or under the
budget. With (a) and (b) but not (c) recalibration *improves* the threshold
and does not repair it. Without (a) there was nothing to repair: a lower rate
is then a stricter threshold under another name. The verdict is computed by
`rules.rq6_verdict` and written as `e6_rq6_verdict`, with one column per
condition and the change in detection beside them. B and C differ from A in
the model as well as in the sample, so a difference there is suggestive, not
attributable.

## 6. What the paper will not claim

- Not "a novel hybrid architecture".
- Not a deployable false-alarm rate: the test day is about 41% attack, a real
  link is far below 1%.
- Not generalisation beyond CIC-IDS2017 unless a second dataset is added.
- No number from `--quick` runs or from the synthetic test data.
- No comparison in E1-E6 that used the test day to choose anything.

## 7. Known weaknesses, stated up front

- One dataset, one week, one test day.
- **The test day is not untouched.** The baseline run and earlier development
  looked at Friday. In particular the primary score rule, max P(attack class),
  was adopted because it detected more DDoS on Friday than `1 - P(benign)` did
  (recorded in `src/run.py`). Both rules are therefore reported wherever the
  classifier is scored on its own (E1, E2, E6), and E5 repeats its fusion
  comparison under the second rule, so the answer to RQ5 can be checked
  against the choice. E3's supervised reference row uses the primary rule
  only; no rule of section 5 reads that row.
- The false-alarm condition of "A beats B" (section 4) first had one way to
  pass, "not more false alarms than B". The other way, "within the budget on
  the test day", was added after a run on **synthetic** captures showed a
  count-valued layer firing far below its budget, which made the first way
  impossible to meet for a reason that has nothing to do with fusion. No
  real-data result was seen. The strict reading is still reported.
- The rules of section 5 are applied by code written on the same day as this
  plan and before any of E1-E6 ran. A rule that is ambiguous in words is
  whatever `src/rules.py` does; `tests/test_rules.py` pins that.
- The `loco` unseen condition tests on all flows of a class; the `blocked` seen
  condition tests on its last 20%. The two test sets differ; `n` is reported.
- E2 and E6 use one seed.
- "Scan-like benign flows removed" (E5) removes exactly the flows the behaviour
  layer flags, then calibrates that layer on what is left. Its validation
  false-alarm rate is optimistic by construction; only the test-day rate, on
  untouched labels, is evidence.
- Leave-one-day-out (E6, C) uses fold models with a day less of training data
  and, for some folds, fewer attack classes than the deployed model.
- The offline replay of the behaviour layer uses the live tracker class with a
  half-open window, direction-repaired endpoints and no source eviction. It is
  not the live configuration.
- The identifier CSVs are expected to be stamped to the minute on most days, so
  the behaviour window cannot be finer than a minute and within-minute order is
  file order. `doctor` records the resolution actually found.
- Bootstrap intervals describe the test day's traffic given one fitted model.
  A class confined to a few minutes has too few blocks for an interval to mean
  anything (`n_blocks`).

## 8. Commands

```bash
python src/study.py doctor          # read this output before anything else
python src/study.py all --quick     # smoke test on a subsample; writes paper/quick/ only
python src/study.py all             # doctor, E1-E6, report; each experiment in its own
                                    # process; stops after doctor if it lists a problem
python src/study.py all --only e3   # re-run one
python src/study.py report          # refresh paper/tables, paper/figures, paper/SOURCES.md
```

## Revisions before the first run

Changes to this text made after 2026-10-04 and **before** any of E1-E6 was run
on the real data. They are listed so that the plan first handed over and the
plan in force can be told apart. None was made with a result of E1-E6 in view:
there was none. The only real-data output seen in between was one `doctor` run
(`runs/96e8769-20261004-231545-study-doctor`), which describes the files and
scores nothing.

- **2026-10-05.** Section 4, "what rule (2) does not do": "every verdict
  carries `passes_strict`" was wider than the code. The RQ3 and RQ5 verdicts
  carry it; RQ6's does not need it.
- **2026-10-05.** Section 4, "quick runs" and "runs with other settings":
  rewritten, and "what `report` puts in `paper/`" added, to say what `report`
  does after the third review of the code. Before: it removed every file of
  its previous report, including those of an experiment whose run folder was
  gone; it took the newest run even when an older one followed the plan; it
  listed every argument that differed from the defaults, including ones the
  experiment never reads.
- **2026-10-05.** Sections 2, 4 and 8: `all` stops when `doctor` lists a
  problem with the files; `SOURCES.md` repeats the `doctor` verdict. Added
  after the first `doctor` run on the laptop showed that `data/` held seven
  `MachineLearningCSV` files and one 85-column file whose timestamps could not
  be read.
- **2026-10-05.** Section 4, "when a capture has identifiers": the 99% floor
  was in the code on 2026-10-04 and was not written down here.
- **2026-10-05.** Section 5, RQ5: unchanged in words. The code behind "if the
  behaviour layer cannot run, it is not evaluated" was incomplete (a two-layer
  system could still be marked as helping in the verdict table) and now does
  what the sentence says.

Seen after these revisions were written and before the first full run: a second
`doctor` run, on the replaced files (`runs/96e8769-20261005-011910-study-doctor`),
and the output of `--quick` smoke runs of all six experiments on them (the
`*-quick` run folders of 2026-10-05). A quick run trains on a subsample and
scores the whole test day, so it prints detection shares; none is a result
(section 6). This paragraph and the two points under it are the only text added
to this file after they were seen.

Two things the smoke runs showed are written down here, so that the plan is not
read as blind to them. No rule above was changed because of either.

- The recalibration of E (section 5, RQ6) gives a probability of zero to every
  class that is absent from the Thursday flows it is fitted on. The RQ6 rule
  looks at false alarms only. It was left as written: the change in detection
  is reported beside the verdict, and the paper must read the two together.
- The behaviour layer's budget-fitted threshold is set on validation flows
  labelled BENIGN, and many of those are flows the layer itself flags at its
  deployed level. The rule for RQ5 and the ablation of section 5 were left as
  written; this is the case the ablation was planned for.

## Deviations

*(none yet)*
