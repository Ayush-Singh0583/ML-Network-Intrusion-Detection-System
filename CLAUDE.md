# Knowledge Base Schema & Operational Guide

This repository contains an LLM Knowledge Base for the **ML Network Intrusion Detection System (ML-NIDS)**. This document specifies the schema, naming conventions, wiki page formats, wikilink syntax, and automatic ingestion rules for raw sources and session learnings.

> **Looking for why something was done?** The **Decision Log** at the end of this file records each decision taken on this project since 2026-10-04 and the reason for it. It is updated in the same session a decision is made. Earlier decisions are in `learnings.md` and the wiki.

---

## 📁 Knowledge Base Directory Architecture

```
ML-Network-Intrusion-Detection-System/
├── raw/                      # Raw, unedited source files (reports, specs, notes, papers)
│   └── .state.json           # Content hashes of processed sources — do not edit by hand
├── wiki/                     # Structured, atomic, interlinked knowledge graph pages
│   ├── Index.md              # Master knowledge index & navigation graph
│   └── *.md                  # Topic pages cross-referenced via [[Wikilinks]]
├── learnings.md              # Running log of what works, what fails, and engineering insights
├── CLAUDE.md                 # This schema, conventions, and processing instructions; the Decision Log is its last section
├── scripts/
│   └── process_raw_hook.py   # Change detector for raw/ — see Ingestion below
└── .claude/
    └── settings.json         # Hook registration (SessionStart / UserPromptSubmit / PostToolUse)
```

---

## 📑 Wiki Page Schema & Formatting Conventions

Every page in `wiki/` must adhere to this standard layout:

```markdown
---
title: "<Page Title>"
category: "<Architecture | ML | Security | Data | Backend | Frontend | Process>"
sources:
  - "raw/<source_file_1>.md"
code_refs:
  - "src/<file>.py"
status: current          # current | stale | superseded
updated: YYYY-MM-DD
tags: ["<tag1>", "<tag2>"]
---

# <Page Title>

## 📌 Executive Summary
Brief 2-4 sentence summary of what this topic covers and its role in the NIDS.

## 🧠 Core Concepts & Mechanics
In-depth technical breakdown of the algorithms, data structures, or security principles.

## 📐 Architecture, Equations & Code Patterns
Mermaid diagrams, mathematical formulations, or python/javascript code snippets.

## ⚠️ Gotchas & Security Pitfalls
Known bugs, security risks (e.g. shortcut learning, scale mismatch), or edge cases.

## 🔗 Related Topics (Wikilinks)
- [[Related-Page-1]] - Description of relationship
- [[Related-Page-2]] - Description of relationship
```

### Frontmatter Fields

| Field | Required | Purpose |
| :--- | :--- | :--- |
| `title` | yes | Human-readable page name |
| `category` | yes | One of the values listed above |
| `sources` | yes | The `raw/` files this page was derived from |
| `code_refs` | **yes, if the page describes code** | The source files whose behaviour this page asserts. This is what makes staleness detectable — see below. |
| `status` | yes | `current`, `stale` (code changed, page not yet reviewed), or `superseded` (replaced by another page, which must be named) |
| `updated` | yes | ISO date of last substantive edit |
| `tags` | yes | Lowercase, hyphenated |

### Wikilink Rules
1. **Syntax**: Use `[[Page-Name]]` (e.g., `[[System-Architecture]]`, `[[Feature-Engineering-Timing]]`).
2. **Precision Anchor Links**: Use `[[Page-Name#Section-Heading|Display Text]]` when referencing specific subsections.
3. **Bidirectional Linking**: Whenever a concept is introduced on page A that relates to page B, both pages should reference each other via wikilinks.
4. **Index Registry**: Every new page must be indexed with a one-line summary in `[[Index]]`.
5. **No orphans**: a page reachable from nothing is a page nobody will read. If it does not belong in the Index's topic map, it does not belong in `wiki/`.

---

## 🕰️ Staleness: the failure this schema exists to prevent

**A knowledge base that describes code which no longer exists is worse than no knowledge base**, because it is confidently wrong and looks authoritative.

This happened here on 2026-08-26. The wiki was written at 13:12 against the pre-refactor codebase. At 17:56 the pipeline was rebuilt and at 20:38 patched again. Pages asserting `θ = 0.55` max-softmax rejection, a `n_estimators=100` unbounded Random Forest, and a 4-file artifact bundle were describing functions that had been deleted or that now raise on call.

**Rules that follow from that:**

1. **`code_refs` is mandatory** on any page that asserts how code behaves. It is the join key between the wiki and the repository.
2. **Before trusting a page, check its `code_refs` against the repo.** If a referenced file has changed materially since `updated`, the page is `stale` until reviewed.
3. **Never silently overwrite a claim.** When new evidence contradicts an existing page, add a correction block above the affected section:

   ```markdown
   > **⚠️ CORRECTION (2026-08-26)** — This section described `<old claim>`.
   > `<what changed and the evidence>`. See [[New-Page]].
   > The original text is retained below because the reasoning is still instructive.
   ```

   The old text stays. A KB that shows *why* a belief was wrong teaches more than one that pretends it never held the belief.
4. **Measured numbers beat asserted ones.** A claim like "energy scoring detects unknown attacks" is worth less than "energy AUROC 0.120 under BatchNorm, 0.630 under LayerNorm, measured on a 12-class ablation at CIC-IDS2017 imbalance". Record the number, the conditions, and the date.
5. **Record refuted hypotheses.** "Weight decay on BatchNorm gains was suspected as the collapse cause; measured effect was 0.870 → 0.870 macro-F1" is a durable, expensive-to-rediscover fact.

---

## 🔄 Raw Source Ingestion Protocol

Triggered automatically by `scripts/process_raw_hook.py` (see Hook Mechanics), or run by hand:

```bash
python scripts/process_raw_hook.py --report      # what changed in raw/
```

When a new source file appears in `raw/`:

1. **Read it in full first.** Do not begin writing pages from a partial read. Sources are usually reports whose conclusions sit at the end.
2. **Extract entities**: architectural decisions, mathematical formulations, measured results, breaking changes, refuted hypotheses.
3. **Decide update vs. create**, per concept. **Prefer updating.** A new page requires a concept that genuinely does not fit an existing one. Ten well-linked pages beat forty stubs.
4. **Enforce wikilinks** throughout the text for every referenced concept, dataset, model, metric or pipeline stage — and add the reciprocal link on the target page.
5. **Update the master graph**: add or revise the entry in `wiki/Index.md`.
6. **Reconcile contradictions** with a `> **⚠️ CORRECTION**` block, never a silent edit.
7. **Record session learnings** in `learnings.md` using the template below.

---

## 🪝 Hook Mechanics

`scripts/process_raw_hook.py` hashes every file in `raw/` and compares against `raw/.state.json`. It is registered in `.claude/settings.json` on three events, because no single event catches every way a file can arrive:

| Event | Catches |
| :--- | :--- |
| `SessionStart` | Files dropped between sessions |
| `UserPromptSubmit` | Files dropped mid-session — fires each turn |
| `PostToolUse` (`Write\|Edit`) | Files the agent itself writes |

**A file dragged into `raw/` from Explorer is not a tool call.** `PostToolUse` alone would never see it. `UserPromptSubmit` closes that gap with at most one turn of latency — the next thing you type triggers ingestion.

The hook **commits state when it fires**. If ingestion is interrupted, the file will not be re-announced. Recover with:

```bash
python scripts/process_raw_hook.py --reset       # treat all of raw/ as new
python scripts/process_raw_hook.py --mark-clean  # accept current raw/ as done
```

State keys are stored with forward slashes (`raw/FILE.md`) so the same state file works from Windows and from the Linux side of the device bridge.

---

## 📝 `learnings.md` Session Logging Convention

After every development or analysis session, append an entry using the following template:

```markdown
## [YYYY-MM-DD] - <Session Topic / Feature>
### ✅ What Worked
- High-level decision, optimal hyperparameter, or architectural pattern that succeeded.
### ❌ What Failed / Gotchas
- Bugs encountered, invalid assumptions, or security pitfalls discovered.
### 🔬 Measured, Not Assumed
- Numbers with their conditions. Include results that came out negative or negligible.
### 💡 Actionable Rule for Next Sessions
- Concrete rule to prevent regressions (e.g., "Always convert flow IAT to microseconds before scaling").
```

---

## 🛡️ Core Project Domain Invariants

These are the rules that survive refactors. Anything not on this list is subject to the staleness process above.

- **Shortcut Learning**: Always omit `Destination Port` and network identifiers (`Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Timestamp`) from ML features (`[[Shortcut-Learning-Port-Bias]]`).
- **Timing Synchronization**: Live network packet statistics must be scaled in **microseconds ($\mu s$)**, not seconds, to match `CICFlowMeter` training distributions (`[[Feature-Engineering-Timing]]`).
- **Evaluation Integrity**: Every metric in a report must be computed from the **same ground-truth array**. Always benchmark using unweighted **Macro Precision, Macro Recall, and Macro F1**, and state whether the macro average is over the union of true-and-predicted labels or over the classes present in the ground truth — the two differ by an order of magnitude when one class dominates (`[[Metric-Dilution-Traps]]`, `[[Model-Evaluation-Metrics]]`).
- **Temporal Splitting**: The primary protocol is day-based (train Mon–Wed + half of Thursday, validate on the rest of Thursday, test on Friday **once**). Random splits of this dataset leak, because flows arrive in near-identical bursts (`[[Data-Leakage-Audit]]`, `[[Training-Protocol]]`).
- **Deduplication**: Deduplicate on **feature columns only**, never including `Day` or `Label`, and report the count removed (`[[Data-Leakage-Audit]]`).
- **Open-Set Rejection**: Use a **distance-from-manifold** score (`mahalanobis_embed`), not maximum softmax posterior. Calibrate the threshold on the validation day at a stated FPR, never on the test day (`[[Rejection-Scoring]]`, `[[Open-Set-Recognition]]`).
- **One-Class Encoders**: Any Deep SVDD encoder must have `bias=False` on every layer and `affine=False` on every norm, and training must assert a non-degenerate distance spread (`[[Hypersphere-Collapse]]`).
- **Artifact Integrity**: A model is served as a single verified bundle in `saved_models/<name>/` whose scaler width, feature list, cleaner and encoder are cross-checked on load (`[[Machine-Learning-Models]]`).
- **Reproducibility**: Seed `random`, `numpy`, `torch`, `torch.cuda`, `cudnn.deterministic` and `CUBLAS_WORKSPACE_CONFIG` before constructing any model (`[[Training-Protocol]]`).

---

## 🧭 Decision Log

> Added 2026-10-04 at Ayush's request: *"Also create a Claude.md In Which You keep Updating every Decision u Are Taking and Why"*.

### The rule

Whenever a session makes a choice that changes **what the code does, what a document claims, or how something is measured**, it adds a row here **in the same session**, with the reason. A choice *not* to do something counts too.

1. **One row per decision**, under the date it was made. Give it the next free number after the last row (`D-nn`). Numbers are never reused.
2. **Say why in plain words**: what was picked, what it was picked over, and what goes wrong with the other option.
3. **If it is a judgement call, say "judgement call".** A round number or a convention must not be dressed up as something derived. These are the rows a reviewer is most likely to question.
4. **Once a row has been handed over (written into the project folder), never delete it and never change what it says.** If a decision changes, add a new row that says "replaces D-xx", and add "→ replaced by D-yy (date)" to the old one. Same idea as the `⚠️ CORRECTION` rule above. Until the hand-over a row may be corrected in place.
5. **A decision is not a result.** A measured number may appear in a row only with the run or the test it came from.
6. **"Where"** names the file that carries the decision, so the row can be checked against the code.
7. A choice or a step that belongs to Ayush goes under **Open — waiting on Ayush** until he answers. Then it becomes a row.
8. **Keep this file readable.** When the log passes about 400 lines (a round number, not a measured limit), move the oldest dated block, unchanged, to `wiki/Decision-Log-Archive.md` (indexed in [[Index]]) and leave one line here saying where it went.

"The draft" below means the paper draft, the Claude Doc titled *Which Layer Catches What? A Leakage-Aware Cross-Day Evaluation of a Three-Layer Hybrid Intrusion Detector*.

### The brief (Ayush's own decisions, 2026-10-04)

- The paper is a college requirement. It should be fit for **at least a Q3 journal**, with research depth.
- **Make the changes in the project folder** while the paper is written.
- Keep this log.
- Asked whether the deep models had ever been run on the real data, the answer was "not sure". Checked on the laptop the same day: the project folder holds outputs for **one** real-data run (XGBoost, cross-day, 12 Sep 2026). The Random Forest run folder holds only its `config.json`. There is no run folder and no saved model for any deep model.

### 2026-10-04 — the study and the paper

#### A. What the paper is

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-01 | The paper is an **evaluation paper**: which layer catches which attack, at one shared false-alarm budget, on a day never seen. It is not sold as a new hybrid architecture. | Combining a classifier with an anomaly model is an established design, and counting ports per source is decades old (the draft §2.1–2.2). Putting known parts together is unlikely to be accepted as a contribution on its own (judgement call). What was not found in the literature is the measurement. | the draft §1–2; `paper/PROTOCOL.md` §1 |
| D-02 | The system under test is put together from parts already in this repo: L1 the XGBoost classifier of the baseline run, L2 the autoencoder (a training option here, never run on the real data), L3 the per-source port/host counter that runs live. Fusing the three is new code. | The paper then tests parts that exist, and each layer uses different information: labels, benign traffic only, no training at all. Only L3 is deployed in this form; the live product serves a Random Forest with a Mahalanobis rejector beside the counter. | `src/layers.py`; `src/fusion.py` |
| D-03 | **No number is written unless a real-data run produced it.** The sections for E1–E6 are empty shells that name the table each will be filled from. | The dataset is not in Claude's workspace and `data/` on the laptop holds no CSVs, so E1–E6 could not be run. A "typical" or estimated number would be fabrication. | the draft "Draft status", §7 |
| D-04 | The only result reported as measured is the baseline run `runs/f853071-20260912-034653-xgb-crossday`, labelled as made by an earlier version of the code. | It is the only real-data run with saved outputs. Its commit is in the repo history (checked 2026-10-04), but the code has changed since, so it cannot be re-made exactly. | the draft §7.1, §9.3 |
| D-05 | The paper says plainly what it does **not** claim: no new architecture, no deployable false-alarm rate, nothing beyond CIC-IDS2017. | Friday is about 41% attack traffic (baseline run, D-04) and there is one dataset. Claiming more is unlikely to survive review (judgement call). | `paper/PROTOCOL.md` §6 |
| D-06 | The draft is a Claude Doc with a "Draft status" block on top: what is measured, what is pending, what must happen before submission. | Ayush and his guide can edit it and export it to Word or PDF. The status block keeps pending work from being read as a result. | the draft |

#### B. Data

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-07 | Use the **`GeneratedLabelledFlows`** CSVs (85 columns), not `MachineLearningCSV` (79). | Only that set has source address and timestamp. The behaviour layer cannot run without them (E4, and the third layer of E5). Time-ordered splits and minute blocks use the timestamp when it is there; without it they fall back to file-row order and fixed 2,000-row blocks. | `data/Readme.md`; [[Dataset-CICIDS2017]] |
| D-08 | Labels are used **as shipped**. | The corrected releases (Engelen 2021; Liu 2022, known here only from its abstract and code repository) rebuild the flows, so their rows cannot be matched to these files. Label noise is studied inside the data instead (E4, the E5 ablation). | `paper/PROTOCOL.md` §2 |
| D-09 | Addresses, ports and timestamp travel in a **side-table** (`meta__*` columns) and never enter the features. | The behaviour layer and time ordering need them; the Shortcut Learning invariant forbids them as features. One function decides what a model may see and a check asserts it. | `src/meta.py` |
| D-10 | A **`doctor`** command describes the CSVs before any experiment trusts them. | Every assumption about the files is measured instead of guessed. Printed per file: rows, columns, source address, timestamp resolution, the hours the timestamps parse onto, the share that parse, whether file order is time order. The full record is in `doctor.json`. | `python src/study.py doctor` |
| D-11 | Timestamps have no AM/PM: an hour below 8 is read as afternoon. | The capture ran 09:00–17:00, so 1 to 7 o'clock can only be p.m. Read naively, afternoon flows sort before morning ones and every time-ordered split is scrambled with no error. `doctor` prints the hours each capture parses onto and lists a capture that parses outside 08:00–18:00 as a problem (an hour of slack on each side of the documented hours: a judgement call). That shows one kind of wrong guess: a file with hours 6 or 7. It cannot show evening traffic printed as 8–11 o'clock, which reads as morning; nothing in the files could. (The first version only wrote the range to `doctor.json`; the second review caught it.) | `src/meta.py`; `study.run_doctor` |
| D-12 | Two filters hit every flow of the week before any split: a `Flow Duration` that is negative or not a number, and a blank label. Only three more things remove rows: before it splits, `random` drops a class with fewer than 10 flows, so that class is in none of its splits; after the split, validation does not keep attack rows of a class missing from its training split (validation only); the E5 ablation edits train and validation (D-45). | The first two are broken records, not traffic. The floor of 10 is a round number kept from the existing pipeline (judgement call). Dropping unseen classes from validation keeps model selection on known classes. The ablation is the experiment itself. Nothing else is removed from a test split. | `src/preprocessing.py`; `paper/PROTOCOL.md` §2 |
| D-13 | The cache remembers each CSV's size and its own layout version. It rebuilds when a size changes or when it was written by older code, and only after checking that all eight CSVs are there. If one is missing the command stops; the old cache stays on disk, unused, until the CSVs are back. | Otherwise replacing the CSVs leaves the old cache in place, still saying "no identifiers". The check comes first because the first version deleted the old cache and only then found the CSVs missing, which leaves a machine with nothing (second review). | `preprocessing.sources_changed`; `preprocessing.build_cache` |

#### C. Splits

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-14 | **Cross-day stays the primary protocol** (train Mon–Wed + half of Thursday, validate on the other half, test on Friday). Kept from the existing project. | It is the only split that tests a day the models have not seen. (`loco` also tests an unseen class, but inside days the model has seen.) | [[Training-Protocol]] |
| D-15 | Three protocols added: `random`, `blocked`, `loco`. | `random` is the kind of split the hybrid-detector papers cited in the draft use (§2.1); it is there to measure what the leak is worth, never as a result. `blocked` shows every class to the model but, inside each capture, never tests a class on flows earlier than its training flows. `loco` holds one class out completely, which is what separates "unseen" from "undetectable". | `preprocessing.build_splits` |
| D-16 | `blocked` cuts each class inside each capture (earliest 60% / next 20% / last 20%), not each capture as a whole. | An attack fills a small part of its capture; cutting the whole capture by time would put most attacks entirely in one split. The 60/20/20 shares are a judgement call. | same |
| D-17 | **Only the training split is de-duplicated**, in all four study protocols. Validation and test are raw flows. | A deployed detector sees duplicates, and a rate means the same thing under every protocol only if every protocol tests raw flows. (The first version, written this session, de-duplicated the whole `random` pool; the independent review caught it.) | same; `tests/test_meta.py` |
| D-18 | The `closedset` protocol and the `run.py train` commands were not redesigned. The study has its own entry point, `src/study.py`. | They belong to the working product, and the study does not use `closedset` unless asked (E1 accepts it as an option). The only change to `run.py` is a `cache --rebuild` flag. | `src/run.py` |

#### D. Scoring, thresholds, statistics

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-19 | Every detector is reduced to **one score per flow**. Every budget-fitted table of a single score (min-p fusion included) goes through one function. Tables that are not budget-fitted single scores say so: Bonferroni fusion (one threshold per layer) and E4's deployed-level tables (coverage, alerts, sweeps), which use the tracker's fixed alert level. | A classifier, a novelty model and a counter can only be compared if they are thresholded and counted the same way. | `src/layers.py`; `stats.detection_table`; `fusion.bonferroni_fired` |
| D-20 | Results are reported **at a fixed false-alarm budget**, per attack class, with the false-alarm rate the threshold really gave on the test day. Accuracy is not a headline number. | An analyst can only absorb so many alerts. Friday is 58.9% benign (baseline run, D-04), so "call everything benign" already scores 58.9% accuracy. | the draft §3.2, §5.5 |
| D-21 | Budgets are 0.1%, 0.5% and 1%; **1% is primary**. | Judgement call, fixed before the runs so it cannot be picked afterwards. In its favour: 1% of Friday's 414,275 benign flows is already about 4,100 false alerts, so a looser budget has little practical use; and all three of the baseline's budgets drifted on Friday (0.1% → 0.14%, 1% → 4.38%, 5% → 9.48%), the 1% one the most, which RQ6 studies. | `config.STUDY_BUDGETS`; `study.PRIMARY_BUDGET` |
| D-22 | The threshold is found **by counting** (`budget_threshold`): the lowest validation-benign score that leaves at most the budget above it. A flow fires only if strictly above. Not `np.quantile`. | `np.quantile` interpolates and can let one extra flow through. Then "at most the budget" is false, and a budget shared between layers no longer adds up. Test: 40,000 flows, 0.1% budget shared by 3 layers → quantile fires 14 per layer, 42 in all; the budget allows 40; counting fires 39. | `src/stats.py`; `tests/test_stats_fusion.py` |
| D-23 | The product's own path keeps `np.quantile`: `evaluation.detection_by_class` (the table `run.py train` writes) and `openset.threshold_at_fpr`. | They made the saved baseline; changing them changes the product. The interpolated threshold is a slightly different number, usually a little lower: on validation it lets through at most one flow more than the budget, and where scores tie the two rules can differ by a whole group of tied flows. Written down so nobody expects a `run.py` table and a study table to match. | [[Three-Layer-Study]] |
| D-24 | Classifier score = largest attack-class probability (primary). `1 − P(benign)` is reported beside it in E1, E2 and E6. | The primary rule was adopted in earlier development **because it found more DDoS on Friday**, so it is not a blind choice. Reporting both in E1, E2 and E6 shows what the choice is worth for the classifier alone; E5 computes everything that involves the classifier under the second rule as well (D-62). E3's reference row uses the primary rule only, and no decision rule reads that row. | `src/run.py` records it; `paper/PROTOCOL.md` §7 |
| D-25 | Intervals come from a **block bootstrap**: resample whole minutes of a capture, 1,000 times, 95%. | Flows in one burst are near-copies. Resampling single flows treats Friday's 158,930 PortScan flows (baseline run, D-04) as independent and gives intervals far too narrow (constructed test in `tests/test_stats_fusion.py`: under 2 points wide by flow, over 30 points wide by minute). The one-minute block is a judgement call: it is the resolution of the timestamps. 1,000 and 95% are the usual conventions. | `stats.bootstrap_rates` |
| D-26 | An interval is **interpreted only if the class spans at least 20 one-minute blocks**, and the code applies that: every detection and paired table prints `n_blocks` and a true/false `interval_counts`; paired tables add `differs` (excludes zero **and** counts). In `e2_seen_unseen` the two columns are suffixed `_seen` / `_unseen`, the false-alarm interval has `n_blocks_benign`, and the verdict tables fold the rule into their pass columns. | A class inside one minute has nothing to resample and gets a zero-width interval, which "excludes zero" by accident. The number 20 is a judgement call (a round floor), not derived. In the first version the rule lived only in the plan and no code applied it (second review). | `stats.interval_counts`; `stats.difference_supported` |
| D-27 | "A beats B" is decided by a **paired** bootstrap of the difference, on the same test flows. | Two detectors scored on the same bursty day have wide intervals that overlap even when one is always better, because the traffic moves both together. | `stats.paired_difference` |
| D-28 | Precision is given as a **projection** at 0.1% and 1% attack prevalence, and called a projection. | Friday is about 41% attack (baseline run, D-04). Precision measured there flatters every detector. The two prevalence values are a judgement call: round numbers for "rare" and "very rare". | `stats.projected_precision` |
| D-29 | **No hyper-parameter tuning.** XGBoost and Random Forest use `config.XGB_PARAMS` / `RF_PARAMS`. The deep models use the `TrainConfig` defaults, except that the study caps them at 30 epochs with patience 10 (the defaults are 60 and 12). The Isolation Forest settings were written with the study and never adjusted. | Judgement call. Tuning on Thursday's validation half was possible; it was left out so that no setting can have been picked for its test-day score, and to keep laptop hours down. Untuned settings may understate a deep model; the draft says so (§9.3). | `study.train_config`; `paper/PROTOCOL.md` §4 |
| D-30 | Seeds: E1, E3 and E5 refit every model with 42, 43, 44. E2 and E6 use 42 only. | Judgement call on cost. Three seeds give a small picture of the spread. E2 and E6 already need a dozen or more fits each; three seeds would triple that on a laptop. Their detection and comparison tables carry the seed, and `results.json` records `seeds_used` (`config.json` shows the command line as given, where `seeds` is still the default 3). | `paper/PROTOCOL.md` §4 |
| D-31 | Macro-F1 over known classes is **not** compared across protocols; an attack-versus-benign F1 is. | On the cross-day test day the only known class present is BENIGN, so that macro-F1 is a one-class number. | `study.run_e1` |

#### E. The three layers

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-32 | The novelty layer is the **autoencoder**. Deep SVDD, Isolation Forest, PCA and Mahalanobis are baselines. | Judgement call. The layer has to be named before the runs, or it would be picked by its Friday score. The autoencoder is the usual benign-only deep model in this literature, and this repo's Deep SVDD once collapsed ([[Hypersphere-Collapse]]). E3 tests whether a deep model earns the place at all. | `src/layers.py` |
| D-33 | Isolation Forest and PCA are fitted on at most 500,000 benign rows, the Mahalanobis covariance on 200,000. The two deep models use all benign rows. | Judgement call: memory and time on a laptop; the two sizes are round numbers. Disclosed in the draft (§4.2, §9.3), because deep and classical models are then not fitted on identical rows. | `src/layers.py` |
| D-34 | At its "deployed" operating point the behaviour layer uses the **live tracker's own numbers**: 100 distinct ports or 50 distinct hosts per source in 60 s. They are copied into `config.py`, and a test pins the copy to the tracker's defaults. In budget-fitted tables (E4's, E5) the layer's threshold is fitted on validation like any other; 100 and 50 then only set the scale between ports and hosts. | The point is to measure the rule that is deployed, not to tune a new one on the test day. Threshold and window sweeps are reported as description only. | `src/config.py`; `backend/live/scan_tracker.py`; `tests/test_behaviour_layer.py` |
| D-35 | Windows are **one-minute tumbling** windows. | The identifier CSVs are expected to be stamped to the minute on most days, so nothing finer can be computed for the week. `doctor` records the resolution really found. | `src/behaviour.py` |
| D-36 | Port and host counts are computed over the **whole week**, then looked up per split. | Counting inside a random half of Thursday sees about half of each source's flows. A scan's port count then roughly halves and the threshold is set too low. | `behaviour.WeekBehaviour` |
| D-37 | **Direction repair**: swap the endpoints when source port < 1024 ≤ destination port. E4's deployed-coverage tables report the result with and without it; the replay, the budget-fitted tables and E5 use the repaired endpoints only. | The flow exporter sometimes records the server as the source; a web server answering 150 clients then looks like a scan of 150 ports. It is a heuristic, so its effect is measured, not assumed. | `behaviour.canonical_endpoints` |
| D-38 | The offline replay hands the live tracker a window **one millisecond shorter**. The live tracker's own behaviour is not changed. | The tracker keeps everything "at most 60 s old". On minute stamps the previous minute is exactly 60 s old, so a closed window holds two minutes. A real clock is unaffected. | `behaviour.replay_scan_tracker` |

#### F. Fusion

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-39 | All layers share **one total false-alarm budget**. | Three detectors at 1% each can cost 3%. An OR of detectors always detects more; without one budget the comparison proves nothing. | `src/fusion.py` |
| D-40 | Primary fusion rule: **min-p** (turn each score into a p-value against validation benign flows, take the smallest, one threshold). Second rule: **Bonferroni** (each of k layers gets budget/k). | Bonferroni wastes budget when layers fire on the same benign flows (two identical layers at a 1% budget use only 0.5%: `tests/test_stats_fusion.py`); min-p uses all of it. Bonferroni is kept because each layer then has its own threshold, which is what lets a detection be credited to a layer. | same |
| D-41 | **No learned combiner** (no stacked model on top of the layers). | It would have to be trained on the attack classes that happen to be in validation: the opposite of a study of unseen attacks. | the draft §4.4 |
| D-42 | Every subset of the layers is scored, not just the full system. | "Fusion helps" means the full system beats each single layer at the same budget. That needs the subsets. | `study.run_e5` |

#### G. The six experiments

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-43 | Six experiments, one per research question, with the **rule for reading each result written down before the runs**. Later changes go into the file's "Deviations" list, never into its text. | So the results cannot shape the analysis. The plan says honestly that it was written *after* the baseline run, with Friday already seen. | `paper/PROTOCOL.md` |
| D-44 | Three explanations for the PortScan miss are tested instead of one being assumed: no per-flow signature, class never seen, scan traffic labelled benign. E2 separates the first two; the third is followed up in E4 and the E5 ablation. | The README said port scans are "not detectable from a single flow, by construction". But PortScan is Friday-only: no model had ever seen one. Unseen and undetectable had never been told apart. | `study.run_e2` |
| D-45 | E5 has an ablation: remove from train and validation the benign-labelled flows the behaviour layer flags. The test day is never touched. | It tests the label-noise explanation. It is marked as an ablation because that layer's validation false-alarm rate becomes optimistic by construction. | `study.sanitise_bundle` |
| D-46 | E6 compares six calibration samples (A, A2, E, B, C, D) and **measures** per-class isotonic recalibration instead of asserting what it does. | The README advised wrapping XGBoost in isotonic calibration without ever running it. The first correction written this session ("it cannot help") was wrong too: that is proved only for a monotone map of the score, and per-class recalibration is not one. The honest position was "never tested", so it is tested. | `study.run_e6` |
| D-47 | `study.py all` runs each experiment in **its own process**. | A process killed for running out of memory cannot be caught by the code. This way it takes down only that one experiment. | `study.run_all` |
| D-48 | `--quick` runs are never results. Plain `report` ignores them. `all --quick` and `report --allow-quick` write to `paper/quick/` and never to `paper/tables` or `paper/figures`. Numbers from the synthetic test data are never evidence. | Both exist only to prove the code runs. In the first version `all --quick` copied its tables into `paper/tables`, where a later failed experiment would have left them looking like results (second review). | `study.run_report` |
| D-49 | `report` takes the newest complete run by the **time stamp** in the folder name, removes the files its previous run wrote (it keeps the list in `.report_files.json`), copies, and lists every file with its run in `paper/SOURCES.md`. | Folder names start with the commit hash, so sorting by name is not sorting by time. Removing the previous report's files means a table from an older run cannot stay behind unlisted. Going by the list, not by file name, means a file `report` did not write is never deleted, whatever it is called. | `study.latest_runs`; `study.run_report` → replaced by D-69 (2026-10-05) |

#### H. Tests

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-50 | The whole study is tested on **synthetic captures** that copy the quirks reported for the real files (odd label byte, blank rows, duplicate header, 12-hour clock, minute stamps, server-as-source flows). On this project's own files only the 85-column layout and the 12-hour clock have been seen so far. | The real dataset is not available where the tests run. This proves the code paths execute. It does not prove the results are right. | `tests/synth_cic.py`; `tests/test_study_cli.py` |
| D-51 | The data, cache, runs and paper folders can be redirected with `NIDS_*` environment variables. | So the tests run the full study in a temporary folder and cannot touch real data or real runs. | `src/config.py` (data, cache, runs, saved models); `src/study.py` (paper) → see D-82 (2026-10-05) |

#### I. Documents and process

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-52 | Wrong statements in the README and wiki were fixed with dated `⚠️ CORRECTION` blocks. The old text stays. | It is this repo's own rule (Staleness, rule 3). | `README.md`; `wiki/` |
| D-53 | A separate reviewer with no stake in the text fact-checked the draft against the code. Its findings were fixed in code, tests and text. | The author's own read-through had passed seven wrong statements and fourteen misleading ones. | `learnings.md`, entry of 2026-10-04 |
| D-54 | A source is cited as read only if its page was really opened. References written from memory carry a ‡. Entry 33 says "not read". | Two pages would not load while writing. Citing them as if read would be the kind of unchecked claim this paper criticises. | the draft, References |
| D-55 | **Nothing was committed or pushed to git.** All changes sit in the working folder. | Ayush did not ask for a commit. Also, `paper/PROTOCOL.md` should be committed by him *before* the first real run, so the commit is evidence that the plan came first. A local commit date can be set by hand; pushing it before the run gives a time stamp that cannot. | — |
| D-56 | A file is written to the laptop only if it has not changed since Claude last read it. | So a newer edit by Ayush is never overwritten without being seen. | — |
| D-57 | The frontend, the API and the live capture code were left alone, apart from one read-only helper (`ScanTracker.source_counts`) and a corrected docstring. | The job was the study and the paper. The working product should keep working. | `backend/live/scan_tracker.py` |
| D-58 | This log is a section of the existing `CLAUDE.md`, not a new file named `Claude.md`. Nothing that was in `CLAUDE.md` was removed. | `Claude.md` and `CLAUDE.md` differ only in letter case. On Windows, where this repo has also been used, they are one and the same file, so a checkout there would silently lose one of them. Keeping the log here also means every new session reads it. | this file; linked from `README.md`, `PIPELINE.md`, `wiki/Index.md` and `wiki/Three-Layer-Study.md` |
| D-59 | "Declare the AI help" was added to the draft's before-submission checklist. | The draft was written with an AI assistant. Publishers such as Elsevier, IEEE and Nature ask authors to declare it and do not accept an AI tool as an author; Elsevier's policy says the authors stay responsible for the content (Elsevier's policy page and a Purdue University library summary, both read 2026-10-04). The college may have its own rule. | the draft "Draft status" |

#### J. After the second review (2026-10-04, later the same day)

This log was itself fact-checked, row by row against the code, before it was handed over (D-65). About thirty of the rows above were corrected in place at that stage, as rule 4 allows before a hand-over; from the hand-over on, rule 4 applies to every row. Where a row was wrong because the code fell short of it, the code was changed and the row now says what the code does. The rows below are the decisions that check led to.

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-60 | The study's headline number is detection at a false-alarm budget, not macro precision / recall / F1. This is a recorded **exception to the "Evaluation Integrity" invariant** above, for `src/study.py` only. Macro-F1 over known classes is still printed in E1, with the number of known classes present beside it. | The invariant reads as written for a classifier that names a class for every flow. A novelty model and a window counter only say "suspicious or not", so there are no classes to average over. And on the cross-day test day the macro-F1 over known classes is a BENIGN-only number (D-31). The product pipeline (`run.py`) still follows the invariant. | `study.run_e1`; the draft §5.5 |
| D-61 | Friday is scored by E1, E3, E4, E5 and E6, once per fitted model, not once in all. This is a recorded **exception to "test on Friday once"** in the "Temporal Splitting" invariant above. | Five of the six research questions need the unseen day, and there is only one. Inside the six experiments nothing is chosen with it: no threshold, model, setting or layer (the primary score rule was chosen with it earlier: D-24). The reuse is stated in the plan and the draft. The real cure is a second dataset. | `paper/PROTOCOL.md` §3, §7; the draft §5.2, §9.3 |
| D-62 | E5 computes everything that involves the classifier a second time, with the same fitted classifier read through the second score rule, `1 − P(benign)`: both fusion rules, the gains, and the "which layer caught what" table. If the answer to RQ5 differs between the two rules, the paper must say RQ5 is not robust to the score rule. `e5_summary` and the figure stay with the primary rule. | The primary rule was picked earlier with Friday in view (D-24). In the first version E5 used that rule only, so "does fusion help?" and the attribution table rested on a choice made on the test day. Nothing is refitted, so the check is cheap. | `study.run_e5`; tables `e5_*_other_score_rule` |
| D-63 | The three yes/no rules of the plan (RQ3, RQ5, RQ6) are **code**. Each experiment applies its own rule and writes a verdict table with one column per condition. | A rule applied by eye after the run still leaves room to read a table kindly. As code, the rule runs the same way whatever the result, and a "no" shows which condition failed. The rules are tested on hand-built tables. | `src/rules.py`; `tests/test_rules.py`; tables `e3_rq3_verdict`, `e5_rq5_verdict`, `e6_rq6_verdict` |
| D-64 | "A detects more than B" counts only if A's false alarms on the test day are **acceptable**: A is at or under the budget there, **or** A is not shown to raise more than B (the paired BENIGN interval does not lie above zero). Applies to RQ3 and RQ5. Both verdict tables also carry the strict reading, `passes_strict`: A's observed false-alarm rate is not above B's at all. | Judgement call. A detector that fires more often detects more for free, so a gain must not be bought with false alarms. "Not more than B" alone punishes A for using a budget that B leaves unused: the behaviour layer's score is a count, ties heavily, and can fire far below its budget. "Within the budget" alone can never be met on a day when every threshold drifts over budget, as the baseline showed. The rule is lenient on purpose and says so: a large but noisy excess passes it, which is why the strict reading sits beside it. The quiet-layer effect was seen on a run on **synthetic** captures, not on real data. | `rules.false_alarm_check`; `paper/PROTOCOL.md` §4 |
| D-65 | The Decision Log is fact-checked against the code by a separate reviewer before it is delivered, like the paper draft (D-53). | It is a list of claims about what the code does. The first check found a false statement in 8 of 59 rows and loose wording in 18 more, and behind several of them real gaps in the code (D-11, D-13, D-26, D-48, D-49, D-62). A second check of the fixes found 2 more false statements and the points behind D-66 to D-68. | `learnings.md`, entry of 2026-10-04 |
| D-66 | "Recalibration **repairs** the threshold" needs three things on the test day: the uncalibrated threshold was over the budget, the recalibrated one has a lower false-alarm rate (paired interval below zero), and that lower rate is within the budget. Lower but still over is reported as "improves". Lower when nothing was over budget is neither. | A repair needs something broken. A lower rate from a threshold that already met its budget is just a stricter threshold, with the detection it costs. The first version of the rule counted any lower rate as support. | `rules.rq6_verdict`; `paper/PROTOCOL.md` §5 |
| D-67 | **An incomplete table never passes a rule.** RQ3 is "not evaluated" unless all three classical baselines the plan names were run. RQ5 is "not evaluated" unless the system has its three layers. A comparison with a missing row is a failed row, not a skipped one. | Skipping a condition is the same as granting it. In the first version of the rules code a single layer without its false-alarm row was silently left out, and beating PCA alone could count as beating "every classical baseline". | `src/rules.py`; `tests/test_rules.py` → see D-71 (2026-10-05) |
| D-68 | `report` lists, per run, every argument that differs from the defaults and names such a run "not run as planned" in `SOURCES.md`. | Only `--quick` marked a run as "not a result". A run with `--seeds 1` or `--models dt` was collected into `paper/tables` like any other. The command line's defaults are the plan's values, so a difference from them is a deviation and has to be visible next to the tables. | `study.non_default_args`; `study.run_report` → replaced by D-70 (2026-10-05) |

#### K. Not done, or could not be checked (2026-10-04)

- **E1–E6 have not been run on the real data.** There are no CSVs in `data/` on the laptop and none in Claude's workspace.
- **The dataset was not downloaded.** Downloading a file needs Ayush's go-ahead, and it belongs on his laptop.
- **arXiv:2609.36039 could not be opened**, and the corrected-labels documentation of Liu et al. would not load. Both are listed in the draft as not read.
- **References marked ‡** were written from memory. Their details must be checked before submission.
- **No second dataset.** Judged the biggest gap a reviewer will point at.
- **Nothing in `raw/` was changed.** As read at the start of the session, `raw/.state.json` lists a file that is no longer in `raw/`; left as found.

### 2026-10-05 — real files, and a third review

#### L. What happened

- **A third check of the 2026-10-04 fixes** (same independent reviewer) found eight more faults in `report` and in the verdict code, and sentences in the plan and the wiki that were wider than the code. They are behind D-69 to D-73.
- **`doctor` was run on real files for the first time** (`runs/96e8769-20261004-231545-study-doctor`). `data/` held seven `MachineLearningCSV` files and one 85-column Friday DDoS file with timestamps like `07-07-2017 03:30`, none of which could be read. That is behind D-74 to D-78.
- **Ayush replaced all eight files from `GeneratedLabelledFlows.zip`.** `doctor` then listed no problem (`runs/96e8769-20261005-011910-study-doctor`): 2,830,743 labelled rows, 85 columns in every capture, every timestamp readable. This closes item 1 of the old open list up to "run `all`".
- **He then started `all --quick` with the 2026-10-04 code**, before the fixes of this section were in his folder. E1 finished; he stopped the run during E2, and only then were the new files written into the folder. See D-80.
- **The rows of section M have not had the separate reviewer's check that D-65 asks for.** Ayush stopped that step on 2026-10-05. They were checked by their author against the code and the records only. A fault found in one later is corrected by a new row, as rule 4 says.

#### M. Decisions

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-69 | `report` works **experiment by experiment**. Where it has a run, it removes the files its previous report wrote for that experiment and copies the new ones. Where it has no run folder, it touches nothing and lists those files as kept. A file it did not write is never deleted and is named in `SOURCES.md` as not covered. Runs are still ordered by the time stamp in the folder name. Replaces D-49. | D-49's version removed everything the previous report had written and then copied what it found. A run folder deleted to free disk space then cost `paper/` its only copy of those tables. And a file nobody had listed could sit in `paper/tables` without a word about it (third review). | `study.run_report`; `tests/test_stats_fusion.py` |
| D-70 | For each experiment `report` takes the newest full run **that follows the plan**. A newer run with other settings does not replace it and is named as passed over. If no run follows the plan, the newest is taken and named "not run as planned", with what differs. "Settings" are the ones that experiment reads, resolved to the values the code uses. A run with no recorded command line counts as off-plan. Replaces D-68. | D-68 listed every argument that differed from the defaults. That flagged arguments an experiment never reads (`--models` for E4), showed a run without a recorded command line as "none", and let a later look with `--seeds 1` replace the planned run's tables just by being newer (third review). | `study.plan_settings`; `study.settings_changed`; `study.runs_for_report` |
| D-71 | D-67 stands, and the code now does all of it: the RQ5 verdict table carries `layers_complete`, and `fusion_helps` is false unless the three layers the plan names are all there. | The summary already said "not evaluated" for a two-layer system, but `fusion_helps` in the table itself could still be true (third review). A table and its summary must not disagree. | `rules.rq5_verdict`; `study.E5_LAYERS`; `tests/test_rules.py` |
| D-72 | `all` checks the folder the experiments read. `--data` is for `doctor` alone; given to `all` it is ignored, with a note. | Otherwise `doctor` could describe one folder while the experiments read another, and both the "skip E4" decision and the `doctor.json` copied into `paper/` would be about files nothing was run on (third review). | `study.run_all` |
| D-73 | `all` prints its summary before it writes the report. A report that cannot be written is said so in one line and the command ends with exit code 1, not with a traceback. | After hours of runs the summary is what says which experiment failed. A table left open in another program must not take it away (third review). | `study.run_all`; `study._remove_report_files` |
| D-74 | `doctor` tells "the timestamp column is missing" (`time=NO`) from "it is there and cannot be read" (`time=UNREADABLE`, with a value it could not read), and counts the second as no usable identifiers. The parser was **not** taught the shape `07-07-2017 03:30`. | On the first real run the line said `time=NO` for a file that has the column, which points at the wrong fault. Reading the dashed shape would have let through a file that had been edited by hand (D-78); nothing in the file says what else was changed in it. The fix for such a file is the original. | `study.run_doctor`; `tests/test_study_cli.py` |
| D-75 | A capture "has identifiers" when at least **99%** of its rows carry a source IP and a timestamp that parses. One constant holds the number. Judgement call: a round floor that lets a handful of broken rows through. | The floor was in the code on 2026-10-04, as a bare `0.99` in four places, and was never logged. It decides whether E4 runs, whether a split is ordered by time and whether bootstrap blocks are minutes. On the real files 100% of timestamps parse in every capture (`runs/96e8769-20261005-011910-study-doctor`), so the choice does not bite here. | `config.IDENTIFIER_COVERAGE` |
| D-76 | `all` **stops before the first experiment when `doctor` lists a problem** (exit code 2). `--despite-problems` runs anyway and needs a Deviations line in the plan. A dataset with identifiers in *no* capture is not a "problem": `all` runs, skips E4 and says so. | On 2026-10-04 `data/` held a mixture and an edited file. Started with `all` instead of `doctor`, the study would have skipped E4, fused two layers and filled `paper/` with hours of tables. A check that precedes a long run has to be able to stop it. The override exists because a problem line can be a false alarm on files that are fine. Where the line sits is a judgement call. | `study.run_all`; `tests/test_study_cli.py` |
| D-77 | `SOURCES.md` repeats what the newest `doctor` run said about the CSVs: files present, identifiers usable, each problem. | The tables are only as good as the files they were computed from, and that verdict was in a run folder nobody opens. | `study.run_report` |
| D-78 | The eight CSVs are used **exactly as they come out of `GeneratedLabelledFlows.zip`**. Nothing is cleaned by hand. All cleaning is code (D-12): a negative or unreadable `Flow Duration` and a blank label drop the row; an empty or infinite value is filled with the training split's median and the row stays. **Flows with a zero duration are kept.** | Ayush had cleaned the old Friday DDoS file by hand. He recalls removing empty values, negative times and zero durations; the records show two negative-duration rows gone and nothing else (the notebook: 225,745 rows, then 225,743, four empty values still there; the cleaner drops exactly 2 rows of the downloaded file). Saving the file rewrote its timestamps, which is what `doctor` tripped on. Hand cleaning is written down nowhere, differs from file to file and cannot be repeated by a reader. A zero duration is normally a one-packet flow, which a deployed detector sees too; how many attack flows have one here is not measured, so deleting them could remove part of an attack class from the test day. | `data/Readme.md`; `preprocessing._clean_frame`; [[Dataset-CICIDS2017]] |
| D-79 | `paper/PROTOCOL.md` was corrected in place, and it lists the corrections itself under "Revisions before the first run". | Its own rule forbids editing the text once one of E1–E6 has run. None has. But the file had been handed over, so a silent edit would hide which plan was given first. The file is still uncommitted (checked on the laptop: nothing committed since `96e8769`). The revisions were written having seen the first `doctor` run only. After the second `doctor` run and the quick run were seen, one paragraph was added to the file, and it says exactly that. | `paper/PROTOCOL.md` |
| D-80 | The `all --quick` run Ayush started on the real files is used for one thing: whether each experiment starts and finishes there. No number from it is quoted, and no rule of the plan is changed because of one. No file in the project folder was replaced while it ran. | PROTOCOL §6. Recorded because Claude read the log of that run (`runs/96e8769-20261005-012138-study-e1-quick/log.txt`) and the terminal output Ayush pasted (all of E1, half of E2), which print detection shares, and a reader should know what had been seen before the full run. `all` starts each experiment as a new process from the files on disk, so new code under a running job would mix two versions in one run. | — |
| D-81 | The draft says that the baseline run of 12 September 2026 most likely read the hand-edited Friday DDoS file. | The file was modified on 4 July 2026, before the run. The hand edit removed exactly the two negative-duration flows that the cleaner removes too: from the downloaded file the cleaner drops 2 rows and keeps 225,743, the edited copy's count (`cache/clean_parts/_state.json`, built 2026-10-05). So the number of flows tested is the same either way: the cross-day split of the downloaded files tests 703,198 Friday flows, 288,923 of them attacks, the baseline's numbers exactly (split line of `runs/96e8769-20261005-012138-study-e1-quick/log.txt`; the test split is not subsampled). Whether feature values differed was not checked. Once E1 has run on the downloaded files, its cross-day XGBoost row is the number to quote, and the baseline is history. | the draft §7.1, "Draft status" |

#### M2. After the tests were run on the laptop (2026-10-05, later)

Sections L and M were written into the project folder before this. Run there for the first time, the test suite gave 239 passed and 3 failed.

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-82 | **The tests are sealed off from the project's own folders.** `tests/conftest.py` points `NIDS_DATA_DIR`, `NIDS_CACHE_DIR`, `NIDS_RUNS_DIR`, `NIDS_ARTIFACT_DIR` and `NIDS_PAPER_DIR` at empty temporary folders before any test imports the code. The tests that load a fixture's cache also name that fixture's data folder, and one test fails if a default folder lies inside the project. This narrows D-51: its "cannot touch real data" was true only of the tests that start the study as a separate process. | The other tests call the library directly and left the data folder at its default. On the laptop that folder held the real dataset under the same eight file names, so `load_clean` found the sizes changed and rebuilt the fixtures' caches from 2.8 million real flows: three tests failed and the rest of that file ran on real data. Nothing in the project folder was written. In the sandbox `data/` holds no CSVs, so the hole could not show there. Checked after the fix with CSVs, a cache and runs placed in the project's own folders: 243 passed, and those folders were unchanged. | `tests/conftest.py`; `tests/test_meta.py` |
| D-83 | `load_clean` itself was **not** changed: given a cache and no data folder, it still compares the cache with the default folder and rebuilds from it when the sizes differ. | That is the behaviour D-13 asks for when the CSVs in `data/` are replaced, and the study always passes both defaults together. Recording the source folder inside the cache was considered and left out: it changes the cache layout hours before the first full run, to guard against a call only the tests made. Judgement call. | `preprocessing.load_clean` |

#### M3. After the smoke run of all six experiments (2026-10-05, later still)

D-82 and D-83 were written into the project folder before this. On the laptop the test suite then gave 243 passed, and `all --quick --seeds 1 --only e5 e6` finished both experiments. With the stopped run before it, every experiment has now started and finished on the real files in `--quick` mode.

| # | Decision | Why | Where |
| :--- | :--- | :--- | :--- |
| D-84 | **Nothing in the plan's rules and nothing in the experiment code was changed after the smoke runs**, although two things in their output invite a change. (1) E6's recalibration gives a probability of zero to every class absent from the Thursday flows it is fitted on, and the RQ6 rule looks at false alarms only, so a recalibration that lowers false alarms by removing detection can still pass it. (2) The behaviour layer's budget-fitted threshold is set on BENIGN-labelled validation flows, many of which the layer itself flags at its deployed level. Both are reported as found. The plan now says, in its disclosure paragraph, that these two things were seen. | A smoke run scores the whole test day, so its output is a preview of the results. Changing a rule after reading it is the result-driven choice the plan exists to prevent (D-43, D-80). For (1) the honest outcome is to report the verdict with the change in detection beside it and to say the rule had a blind spot. For (2) the E5 ablation (D-45) is the planned look at exactly this. A recalibration fitted on the training days, where every known class is present, was not added: it would be a new experiment designed after seeing the outcome. It belongs in the paper's limits, or in a later run labelled as added afterwards. | `paper/PROTOCOL.md` "Revisions before the first run" (two points, added to the one paragraph D-79 mentions) |
| D-85 | The behaviour layer's counting was checked by hand on one real capture (Friday DDoS, as cleaned by the cache) before the full run. Nothing was changed. | The smoke run of E5 removed a very large share of the validation flows as "scan-like", which is either a fact about the data or a fault in the counting, and a night of runs should not rest on a guess. On that capture the sources past the deployed thresholds are ordinary workstations reaching more than 50 distinct hosts in a minute, and without the direction repair the web server shows as a source "scanning" its clients' ports: the artefact D-37 describes. No attack flow of that capture is flagged, as expected for a flood from one address to one port. So the counting does what it says. The numbers themselves are E4's to report. | `src/behaviour.py`; `learnings.md` |

#### N. Still not done, or could not be checked (2026-10-05)

- **E1–E6 have not been run in full on the real data.** No result number exists beyond the baseline of D-04.
- **D-82 to D-85 have not had the separate reviewer's check either** (see section L).
- **The dashed-date hint in `doctor`** ("a spreadsheet writes this") is an inference from one file. The claim it rests on, that this dataset prints `7/7/2017 3:30`, is measured (the table in [[Dataset-CICIDS2017]]).
- **Whether the edited Friday DDoS copy differed from the original in its feature values** was not checked. The copy has been overwritten on the laptop by the downloaded file.
- **Feature values of the two downloads were not compared**, only row and label counts.
- Unchanged from section K: arXiv:2609.36039 and the Liu et al. documentation not read; references marked ‡ unchecked; no second dataset; `raw/` untouched.

#### Open — waiting on Ayush

*(This list is kept current. Items leave it when they are answered and become rows. Closed on 2026-10-05: the download and the `doctor` check, see section L.)*

1. Commit `paper/PROTOCOL.md` and the new code, and **push**, before the first full run.
2. Then run `python src/study.py all`.
3. Add a second dataset or not.
4. Which journal to aim at. Its template, length limit and AI-declaration rule follow from that.
5. Whether the paper should say that the baseline run may have used uncommitted changes on top of commit `f853071`. Since D-81: whether the baseline run stays in the paper at all once E1 exists.
6. The two invariants touched by D-60 and D-61: keep them as written for the product pipeline with the study as a recorded exception (the state now), or reword them.
