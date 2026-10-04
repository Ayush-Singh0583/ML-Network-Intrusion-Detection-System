# Project Learnings & Retrospectives

This log tracks architectural decisions, optimization experiments, what works, what fails, and recurring security gotchas across development and research sessions.

---

## [2026-08-26] - NIDS Security Refactor & Microsecond Feature Synchronization

### ✅ What Worked
- **Removing `Destination Port`**: Successfully broke the shortcut learning artifact where tree models memorized port-to-attack mappings (e.g. Port 80 = DoS, Port 21 = FTP-Patator). Forces classifiers to learn invariant traffic geometry (packet lengths, inter-arrival time variance, TCP flag sequences).
- **Balanced Sample Weighting for XGBoost & LightGBM**: Using `compute_sample_weight("balanced", y_train)` directly eliminated zero-recall failure modes on rare minority attacks (Infiltration, Heartbleed, SQL Injection).
- **Posterior Probability Thresholding for Open-Set Zero-Day Detection**: Establishing a confidence cutoff ($\theta = 0.55$) routes novel Friday attack distributions and out-of-distribution traffic to `Unknown_Attack` instead of falsely misclassifying them as `BENIGN`.
- **Microsecond Timing Conversion in Live Sniffer**: Multiplying `Flow Duration`, `Flow IAT`, `Active`, and `Idle` deltas by $10^6$ resolved the $1,000,000\times$ feature scale divergence between Scapy live captures and `CICFlowMeter` training features.
- **Two-Stage Mutex in Flow Manager / Garbage Collector**: Acquiring `flows_lock` only during the key-eviction phase and running model inference outside the lock prevents Scapy's packet capture thread from dropping incoming packets.

### ❌ What Failed / Gotchas
- **Using raw `time.time()` differences**: Passing seconds ($s$) to `StandardScaler` trained on microseconds ($\mu s$) generated massive negative z-score distortion, driving prediction confidence to near zero or defaulting to random classes.
- **Weighted F1 as a Primary Success Metric**: In CICIDS2017 (>80% Benign traffic), a trivial classifier predicting `BENIGN` for all packets yields >99% Accuracy and ~0.99 Weighted F1 while having 0.00 Recall on minority cyber attacks.
- **Closed-World Evaluation on Cross-Day Splits**: Standard `LabelEncoder.inverse_transform()` threw index errors when encountering Friday attacks unseen during Monday-Thursday training. Ground-truth mapping must remap unseen labels to `Unknown_Attack` prior to metric calculation.
- **Sorting by non-existent `F1 Score` column in `compare_models.py`**: Model comparison script crashed when sorting by `"F1 Score"` after renaming output metrics to `"Macro F1"` and `"Weighted F1"`.

### 💡 Actionable Rules for Future Sessions
1. **Always Synchronize Feature Dimensions & Scales**: Any timing feature added to live capture must strictly output microseconds ($\mu s$).
2. **Never Re-introduce Port Identifiers into the Feature Set**: Keep `remove_identifier_columns()` clean of `Destination Port`, `Source Port`, `Source IP`, `Destination IP`, and `Flow ID`.
3. **Always Save the 4-File Artifact Bundle**: Model updates must serialize `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `feature_names.pkl` into `saved_models/<model_name>/` in the same transaction.
4. **Mandatory Unweighted Macro Reporting**: Every evaluation must print and export Macro Precision, Macro Recall, Macro F1, and confusion matrix heatmaps.

---

## [2026-08-26] - Knowledge Base Repair & Staleness Convention

### ✅ What Worked
- **`code_refs` frontmatter as a join key**: pinning each wiki page to the source files whose behaviour it asserts makes staleness *detectable* instead of discovered by accident. The wiki had been silently describing a deleted codebase for seven hours.
- **Correction banners instead of overwrites**: retaining the refuted text under a dated `> **⚠️ CORRECTION**` block preserves *why* a belief was held. The θ=0.55 max-softmax reasoning is still instructive precisely because it was measured wrong.
- **Three hook events instead of one**: `SessionStart` + `UserPromptSubmit` + `PostToolUse` covers every route a file takes into `raw/`, with at most one turn of latency.

### ❌ What Failed / Gotchas
- **The existing hook could never fire.** `.agents/hooks.json` was written for a different agent runtime (Windsurf/Cascade): matchers `write_to_file|replace_file_content`, response schema `{"injectSteps":[{"ephemeralMessage":...}]}`. This runtime reads `.claude/settings.json`, matches `Write|Edit`, and expects `{"hookSpecificOutput":{"additionalContext":...}}`.
- **`PostToolUse` alone cannot see a file dragged in from Explorer** — that is not a tool call. Any "watch a folder" hook built only on tool events is watching the wrong thing.
- **`raw/.state.json` used `os.path.join` keys.** A state file written on Windows (`raw\FILE.md`) never matches Linux keys (`raw/FILE.md`), so every file looks new from the other side of the device bridge. Keys are now normalised to forward slashes, with legacy migration on read.
- **The hook commits state when it fires**, so an interrupted ingestion silently loses the notification. `--reset` / `--mark-clean` / `--report` exist for that.

### 🔬 Measured, Not Assumed
- Wiki written 13:12; pipeline rebuilt 17:56; patched again 20:38. Four of ten pages were materially wrong by the end of the day — 40% decay in under eight hours during active development.
- `Open-Set-Recognition` asserted a method measured at AUROC 0.112–0.120 (inverted) against 0.914 for the alternative on identical models.

### 💡 Actionable Rules for Next Sessions
1. **Any page asserting code behaviour carries `code_refs`.** No exceptions — it is the only mechanism that makes decay visible.
2. **Never silently overwrite a claim.** Add a dated correction block naming the old claim and the evidence against it.
3. **Record refuted hypotheses and negligible effects.** "Weight decay on norm gains was suspected as the collapse cause; measured 0.870 → 0.870" is expensive to rediscover and cheap to write down.
4. **Prefer a measured number with its conditions** over a qualitative claim. "Energy AUROC 0.120 under BatchNorm, 0.630 under LayerNorm, 12-class ablation at CIC-IDS2017 imbalance" beats "energy detects unknowns".
5. **Verify a hook by firing it**, not by reading its config. The previous one looked correct and was inert.

---

## [2026-09-29] - Error sweep: lint, lockfile, live capture, flow direction

### ✅ What Worked
- **Running the whole product on synthetic CIC-shaped CSVs** (built through `backend/live/extractor.py`, real header quirks included): `cache → train → bundle → API → dashboard` end to end without the real dataset. Found three bugs the 66 unit tests could not see.
- **AsyncSniffer + `started_callback`** for capture: Stop really stops, and a start that cannot open the interface returns 503 with the reason.

### ❌ What Failed / Gotchas
- **Flow direction came from the sorted flow key**, not the first packet. "Forward" meant "whichever IP sorts first as a string", so Fwd/Bwd features were swapped for many connections and the history showed servers as sources. The wiki (`Live-Packet-Flow-Pipeline`) described the correct design; the code did not follow it.
- **`sniff(stop_filter=...)` only checks the flag when a packet arrives.** Stop then Start left two sniffers running.
- **A dead sniffer thread still reported `running: true`**: the Event flag outlived the thread.
- **Report footer and dashboard hardcoded 58.47% / 69.68% / 58.91%** from one RF run, for every model. Now computed per run and stored in the bundle as `accuracy_readings`.
- `package-lock.json` was missing `@emnapi/*` entries: `npm ci` failed on npm 10 and 11.

### 🔬 Measured, Not Assumed
- Stop→Start race: **2.00** `process_packet` calls per packet on the wire before the fix, **0.98** after (reference sniffer on the same interface, live traffic).
- Frontend: 9 ESLint errors + 1 warning → 0. `npm audit`: 5 vulnerabilities (4 high, all build tooling) → 0 via `npm audit fix`, no `package.json` change.
- Tests: 66 → 73 passing (`tests/test_live_fixes.py`).

### 💡 Actionable Rule for Next Sessions
- Anything the UI or a report states about a model must come from the bundle's manifest, never from a literal.
- A thread-backed "running" status must ask the thread (`is_alive()`), not a flag.

---

## [2026-10-04] - Study pipeline for the paper (three layers, one false-alarm budget)

### ✅ What Worked
- **Reducing every detector to "a score per flow"** (`src/layers.py`). Classifier, benign-only novelty model, window counter and every min-p fusion then go through one function (`stats.detection_table`): threshold on validation benign scores, per-class share above it on test, observed false-alarm rate beside the budget. Numbers from different kinds of detector become comparable. (Two tables are built differently, and say so: Bonferroni fusion has one threshold per layer, and E4's deployed coverage uses the tracker's fixed alert level.)
- **An identifier side-table instead of identifier features** (`src/meta.py`). Address, ports, timestamp, file row and capture index travel as `meta__*` columns; one function (`feature_columns`) keeps them out of `X` and `SplitBundle.validate()` asserts it. The behaviour layer and the time-ordered splits read the side-table; the shortcut-learning invariant is untouched.
- **A `doctor` command that measures the files before anything trusts them.** Which download is on disk, the hours the timestamps parse onto, their resolution, whether file order is time order. Each is printed, the full record is in `doctor.json`, and a capture that parses off its date or outside working hours is listed as a problem.
- **The plan's yes/no rules as code** (`src/rules.py`). RQ3, RQ5 and RQ6 each have a rule in `paper/PROTOCOL.md`; each experiment applies its own and writes a verdict table with one column per condition. A rule that runs the same way whatever the result cannot be read kindly afterwards.
- **Synthetic captures that reproduce the file quirks** (`tests/synth_cic.py`): cp1252 label byte, blank trailing rows, duplicated header, `Infinity` strings, 12-hour clock without AM/PM, minute stamps, server-as-source flows. The whole study runs end to end in the tests without the dataset.
- **One process per experiment in `study.py all`.** A process killed for memory raises nothing a `try` can catch; isolating experiments means it takes only itself down.

### ❌ What Failed / Gotchas
- **Three README claims did not follow from the runs behind them** (corrected in place, 2026-10-04): "port scans are not detectable from a single flow, by construction" (every model was trained without a PortScan flow — PortScan is Friday-only); "wrap XGBoost in isotonic calibration to fix the threshold" (never run; see the next item); the Random Forest table (its run directory holds only `config.json`).
- **My own first correction of the calibration claim was wrong too.** I wrote that a monotone re-mapping cannot move a quantile threshold, so calibration "cannot help". True for a strictly increasing map of the scalar score. Not true as a statement about the remedy the README named: per-class recalibration plus renormalisation is not a monotone map of the largest attack probability. An independent review caught it. The honest position is "never tested", so E6 now tests it.
- **An independent fact-check of the first draft found seven wrong statements and fourteen misleading ones** — in text I had already checked myself. The ones that needed code: the `random` protocol de-duplicated its whole pool (test set of unique vectors, not comparable with the other protocols); "three seeds for every random fit" held for E1 only; leave-one-day-out scored de-duplicated rows and let the Thursday fold early-stop on Thursday; the benign-collision share was called a ceiling on detection; a class inside one time block gets a zero-width interval that "excludes zero" trivially; `report` picked the "newest" run by folder name, which starts with the commit hash; macro-F1 over known classes is a one-class number on the cross-day test split. The ones that needed words: the plan was written after the baseline run, not before any real-data run; the primary score rule had been chosen with Friday in view (`run.py` records it).
- **A second independent check, this time of the Decision Log, found that 8 of its 59 rows said something the code did not do** — and behind several of them the code, not the wording, was at fault: `all --quick` copied its smoke-test tables into `paper/tables`, and `report` never removed a table an earlier run had left there; "intervals on fewer than 20 blocks are not interpreted" was a sentence in the plan that no code applied, and `e2_seen_unseen` did not even carry `n_blocks`; E5 scored the classifier by the one rule that had been picked with Friday in view, so "does fusion help?" rested on it; `doctor` wrote each capture's parsed time range to a JSON file and printed nothing; a cache rebuild deleted the old cache before checking that the CSVs to rebuild it from were there; E2 and E6 are single-seed but their tables did not say so. All fixed in code, with tests.
- **The same reviewer's second pass, over the fixes, still found things**: the new false-alarm rule was weaker than its wording ("no more false alarms than B" was coded as "an excess is not *shown*", so a large noisy excess passed); a verdict could pass on an incomplete table (a layer with no false-alarm row was skipped, beating PCA alone counted as beating "every classical baseline"); "recalibration repairs the threshold" counted any lower rate, even when nothing had been over budget; `report` deleted any file named like a study table, including one a person had made; and a run with `--seeds 1` was collected into `paper/tables` as if it had followed the plan. Fixed: a strict reading beside every verdict, "not evaluated" for incomplete tables, a three-part definition of "repair", a list of the files `report` wrote, and off-plan arguments named in `SOURCES.md`.
- **A decision rule can be impossible to meet for a reason that has nothing to do with the question.** "The full system must raise no more false alarms than each single layer" fails whenever one layer is very quiet: the behaviour layer's score is a count, ties heavily, and fires far below its budget, so a system that uses the budget it was given "loses" to it. Seen on a synthetic run, before any real one. The rule now accepts a system that is within its budget on the test day, or no worse than the layer; the harsher reading is still reported (`passes_strict`).
- **`np.quantile` breaks a budget guarantee by one flow.** Interpolation can leave one more reference flow above the threshold than the budget allows; with the budget split across layers the union bound no longer holds. Replaced in the study by an order statistic (`stats.budget_threshold`).
- **A closed 60-second window on minute stamps is two minutes.** The live tracker keeps `ts >= now - window`; on a capture stamped to the minute the previous minute is exactly 60 s old and stays in. The offline replay now hands it a window one millisecond shorter.
- **Counting ports inside a split shrinks the counts.** A random half of Thursday holds about half of each (source, minute) group, so a scan's port count roughly halves. Window counts are computed over the whole week and looked up per split.
- **A cache that outlives its CSVs.** Replacing the files in `data/` left the old cache in place, still reporting "no identifiers". The cache now records each CSV's size and rebuilds when one changes.
- **The CSVs the baseline was built on were a mixture of the two CIC downloads** (seven `MachineLearningCSV` files, one `GeneratedLabelledFlows` file — the schema-guard comment in `preprocessing.py` and `notebooks/01_dataset_exploration.ipynb` both show it). It went unnoticed because the six extra columns are all identifiers that the cleaner drops. *(Reworded the same day: this line first said "`data/` holds a mixture". `data/` on the laptop holds no CSVs at all today, only `.gitkeep` and `Readme.md`.)*
- **`raw/.state.json` lists `raw/PROJECT_DOCUMENTATION.md`, which is no longer in `raw/`** (it is git-ignored). The ingestion hook will report it as REMOVED on its next firing. Left as found.
- First version of the synthetic scans was thinned out to fit a small flow count, so a "scan" was 74 ports a minute and nothing flagged it. Scans are now generated at a fixed rate, starting on a minute boundary.

### 🔬 Measured, Not Assumed
- **No experiment was run on the real dataset this session.** The only real-data numbers in the project are still those of `runs/f853071-20260912-034653-xgb-crossday`. Everything below is arithmetic on constructed inputs, checked by tests.
- Threshold rule: 40,000 benign scores, 0.1% budget shared by three layers (40 false alarms allowed) → `np.quantile` thresholds fire 42, `budget_threshold` fires 39 (`tests/test_stats_fusion.py`).
- Interval width: 20,000 flows in 20 one-minute blocks, half the blocks detected → flow-level bootstrap interval under 2 points wide, block-level over 30 points wide, same point estimate (same file).
- Min-p against Bonferroni with two identical layers at a 1% budget: Bonferroni uses 0.5% of validation benign flows, min-p uses 1.0%.
- Direction artefact: one server answering 150 client ports in a minute counts as 150 distinct destination ports as recorded and 1 after the endpoint swap (`tests/test_behaviour_layer.py`).
- Paired comparison: detector A = detector B plus 2 points in every block, B swinging from 10% to 90% between blocks → the two separate intervals overlap by tens of points; the paired interval of the difference is under 1 point wide and excludes zero.
- Tests: 73 → 236 passing. The end-to-end study tests take about two and a half minutes, the whole suite about four.

### 💡 Actionable Rule for Next Sessions
- **A class that is absent from training cannot be called undetectable.** Before writing "the model cannot see X", check that X was in the training split.
- **Every number in the paper traces to a run directory in `paper/SOURCES.md`.** Nothing from `--quick`, nothing from the synthetic captures, nothing from memory.
- **Fix the analysis before the run.** `paper/PROTOCOL.md` is edited only by appending to its Deviations section.
- **When a guarantee is stated ("at most the budget"), test it on awkward sizes**, not on round ones.
- **"A beats B" needs the paired interval of the difference**, not two intervals read side by side.
- **Run `python src/study.py doctor` after touching `data/`**, and read it.
- **Have someone else fact-check a draft against the code before it is handed over.** The author's own read-through missed twenty-one problems that a reader with no stake in the text found in one pass.
- **"Cannot" needs a proof that covers the thing named.** A theorem about monotone maps says nothing about a procedure that is not one. When in doubt, write "never tested" and test it.
- **Say what a plan was written knowing.** "Fixed in advance" means in advance of specific runs; name them.
- **Read `n_blocks` before an interval**, and treat a zero-width interval as a warning, not a result.
- **A rule in the plan must be a function in the code.** If `PROTOCOL.md` says "ignored below 20 blocks", some line of code has to do the ignoring and some test has to pin it.
- **Look before deleting.** Check that a rebuild can finish before removing what it replaces.
- **Smoke-test output never shares a folder with results.**
- **A rule that says "no more" must not be coded as "not shown to be more".** Say which one it is, and report the other beside it.
- **Skipping a missing row grants the condition it carried.** An incomplete comparison is a failure or "not evaluated".
- **Delete by list, not by name pattern.** A tool may remove what it wrote and nothing else.
- **Fact-check the Decision Log against the code like the paper.** It is a list of claims about what the code does, written by the person least able to see where it is wrong.
- **Write each decision into the Decision Log (last section of `CLAUDE.md`) in the session that makes it**, with the reason and, where it is one, the words "judgement call". Ayush asked for this on 2026-10-04; the log was back-filled with that day's decisions (`D-01` onwards).
- **Do not create a file whose name differs from an existing one only by letter case** (`Claude.md` beside `CLAUDE.md`). This repo is used on Windows as well as Linux, where the two are one file.

## [2026-10-05] - First `doctor` runs on real files; third review of the study code

### ✅ What Worked
- **`doctor` earned its place on its first real run.** `data/` held seven `MachineLearningCSV` files and one hand-edited `GeneratedLabelledFlows` file. It said so before a single experiment ran, and the fault cost one download instead of a night of runs (`runs/96e8769-20261004-231545-study-doctor`).
- **The second run, on the eight files straight out of `GeneratedLabelledFlows.zip`, listed no problem** (`runs/96e8769-20261005-011910-study-doctor`). Every file quirk the code had been written for from descriptions was there as described: the non-UTF-8 label dash, the blank trailing rows, the 12-hour clock, minute stamps.
- **Reading the repo's own notebook as evidence.** `notebooks/01_dataset_exploration.ipynb` reads the Friday DDoS file twice. Its outputs date the hand edit and show exactly what it removed, which Ayush's memory of the cleaning did not.
- **A third pass by the same independent reviewer, over the second round of fixes**, found eight more faults in `report` and the verdict code. All fixed with tests.
- **One constant for one floor.** "At least 99% of rows have a source IP and a readable timestamp" was a bare `0.99` in four places. It is now `config.IDENTIFIER_COVERAGE`, so `doctor`, the cache, the time-ordered splits and the bootstrap blocks cannot drift apart.

### ❌ What Failed / Gotchas
- **`doctor` printed `time=NO` for a capture that has a timestamp column it could not read.** That reads as "the column is missing" and points at the wrong fault. It also reported the same fault twice ("0.0% parse" and "100.0% not on 2017-07-07"). Now `time=UNREADABLE`, one problem line, with a value it could not read.
- **The Friday DDoS file had been cleaned by hand.** The notebook's first reading: 225,745 rows, timestamps `7/7/2017 3:30`, minimum `Flow Duration` −1. Its second reading: 225,743 rows, no negative duration, 4 empty `Flow Bytes/s` values still there. `doctor`: 225,743 rows, timestamps `07-07-2017 03:30`, 0.0% readable. Saving the file rewrote its date column. Ayush remembered removing empty values and zero durations as well; the notebook shows only the two negative-duration rows gone.
- **A statement about a file was taken from a notebook output.** The dataset wiki page said the DDoS file "does show the `7/7/2017 3:30` timestamp style". That was the notebook's first reading, not the file on disk. Corrected with a dated block.
- **The third review's findings** (all in code written on 2026-10-04): `report` removed the tables of an experiment whose run folder was gone, leaving `paper/` with nothing for it; `all --data elsewhere` let `doctor` describe a folder the experiments did not read; files in `paper/tables` that no report had written went unmentioned; a run with no recorded command line was shown as "settings changed: none"; arguments an experiment never reads were flagged as deviations; a newer off-plan run replaced the planned run's tables; a file `report` could not delete ended `all` with a traceback and no summary; `fusion_helps` could be true in the RQ5 verdict table for a two-layer system although the summary said "not evaluated".
- **Clearing `data/` for the new CSVs also deleted the tracked `data/Readme.md`.** Restored. (`git status` shows such a deletion; `git restore data/Readme.md` undoes it.)
- **The test suite was not sealed off from the project's own `data/`, and nobody could tell until it ran on a machine that had real data there.** Thirteen calls in `tests/test_meta.py` loaded a fixture's cache with `load_clean(cache)` and left the data folder at its default, the project's `data/`. Where that folder holds no CSVs the staleness check has nothing to compare and the fixture's cache is used. On the laptop it held the real dataset under the same eight names: the sizes differed, so `load_clean` did what it is for and rebuilt the fixture's cache from 2.8 million real flows. Result there: 239 passed, 3 failed (326 s), and the tests that share those fixtures ran on real data whether they passed or not. Nothing in the project folder was written (checked on the laptop: `runs/`, `cache/`, `paper/`, `saved_models/` unchanged); the rebuilt caches went to pytest's temporary folder. Fixed three ways: `tests/conftest.py` points every `NIDS_*` folder at an empty temporary one before the code is imported; the thirteen calls now name their fixture's data folder; a new test fails if a default folder lies inside the project.
- **The smoke runs showed two things that invite a change of plan, and the plan was not changed** (Decision Log D-84). Per-class isotonic recalibration fitted on Thursday's flows gives zero probability to every class Thursday does not contain, so the score of an unseen attack that the model reads as a Tuesday or Wednesday class goes to zero; and the RQ6 rule, written before this was seen, looks at false alarms only. The behaviour layer's budget-fitted threshold is set on BENIGN-labelled validation flows that the layer itself flags.
- **A test that had only ever run on synthetic data asserted something that is not true of real data**: "no two rows of the de-duplicated training matrix are equal". De-duplication is on the features as recorded. Filling gaps, scaling to float32 and clipping to ±10 then make a few distinct rows equal.
- **`all --quick` was started on the real files with the 2026-10-04 code, before the fixes above were delivered.** Harmless for a smoke run, which goes to `paper/quick/` and proves only that the experiments start and finish. It was stopped during E2, at Claude's suggestion, so that the smoke run could be repeated on the delivered code. No file in the project folder was changed while it ran: `all` starts each experiment as a new process from the files on disk, so replacing `src/study.py` mid-run would have mixed two versions in one run.

### 🔬 Measured, Not Assumed
- **Still no experiment E1–E6 run on the real data as a result.** The numbers below describe the files, not a detector.
- The eight `GeneratedLabelledFlows` files (`runs/96e8769-20261005-011910-study-doctor`): 2,830,743 labelled rows; 85 columns each; 100% of timestamps parse and fall on the capture's date. Monday is stamped to the second, the other seven to the minute. Parsed hours per capture all lie inside 08:42–17:10.
- Rows per capture: Monday 529,918; Tuesday 445,909; Wednesday 692,703; Thursday morning 170,366 (plus 288,602 blank rows, dropped); Thursday afternoon 288,602; Friday morning 191,033; Friday PortScan 286,467; Friday DDoS 225,745.
- File order is time order in five captures and not in Monday, Tuesday, Wednesday (60.6%, 64.2%, 76.3% of consecutive rows go forward in time).
- The seven `MachineLearningCSV` files that were in `data/` on 2026-10-04 had the same labelled-row counts and per-label counts as their `GeneratedLabelledFlows` namesakes (`runs/96e8769-20261004-231545-study-doctor` against the run above). Feature values were not compared.
- The baseline run of 12 September 2026 tested 703,198 Friday flows: 414,275 BENIGN, 288,923 attack (`runs/f853071-20260912-034653-xgb-crossday/results.json`). The three Friday files hold 703,245 labelled rows as downloaded and held 703,243 with the edited DDoS copy. The count does not say which copy the run read: the two rows the hand edit removed are rows the cleaner drops too (next item).
- What the cleaner removes from the downloaded files (cache built on the laptop on 2026-10-05, `cache/clean_parts/_state.json`): 115 of 2,830,743 rows, all labelled BENIGN; 2,830,628 kept; no attack flow removed. From the Friday DDoS capture: 2 rows, leaving 225,743 (DDoS 128,027, BENIGN 97,716), which are the counts of the hand-edited copy. The hand edit therefore removed exactly the two negative-duration flows that the code removes anyway.
- Rows the cleaner removes per capture (terminal output of the cache build, 2026-10-05): Monday 15, Tuesday 17, Wednesday 21, Thursday morning 11, Thursday afternoon 4, Friday morning 9, Friday PortScan 36, Friday DDoS 2. That is 115 in all and 47 on Friday, so the cross-day test split of the downloaded files is 703,198 flows with 288,923 attacks: the same numbers as the September baseline (`runs/96e8769-20261005-012138-study-e1-quick/log.txt`; the test split is not subsampled by `--quick`).
- A `--quick` run on the real files is not "a few minutes": E1 alone took 570 s on the laptop, because every fit reloads and re-splits 2.8 million flows and scores full validation and test splits. Only the training rows are subsampled.
- From the accidental run of `tests/test_meta.py` on the real data (laptop, 2026-10-05): in the `random` split, after training-only de-duplication, 671 of 1,563,021 training rows are equal to another row once transformed. Checked on one downloaded capture (Friday DDoS, 225,743 cleaned rows, 221,288 distinct recorded feature vectors): 0 rows coincide after dropping constant columns and filling gaps, 2 after scaling to float32, 6 after clipping to ±10. So the transform, not the de-duplication, produces them.
- The suite on the laptop before the fix (Python 3.14.4, pandas 3.0.6): 239 passed, 3 failed, all three for the reason above. After the fix, in the sandbox with CSVs, a cache and runs placed in the project's own folders: 243 passed, and the files in `runs/`, `cache/`, `paper/` and `saved_models/` were byte-for-byte as before.
- Smoke runs (`--quick`) on the laptop, real files, 2026-10-05. Every experiment starts and finishes. Wall time: E1 570 s and E3 about 320 s (three seeds each), E2 about 525 s, E4 about 20 s, E5 48 s and E6 160 s (one seed each). The test suite there: 243 passed in 127 s. These are timings, not results.
- The behaviour layer's counting, checked on one real capture (Friday DDoS, 225,743 cleaned flows): with the direction repair 21,902 flows come from a source past the deployed thresholds, all labelled BENIGN, and the sources are ordinary workstations reaching 65 to 142 distinct hosts in a minute. Without the repair 49,118 flows are flagged, 31,687 of them with the web server 192.168.10.50 as "source" and up to 1,894 distinct "destination ports" in a minute: the server-as-source artefact. No DDoS flow is flagged with the repair, 3 without. This was a check that the counting is not broken, made before the full run; E4 reports the week.
- Tests: 236 → 243 passing.

### 💡 Actionable Rule for Next Sessions
- **Never clean a capture by hand, and never save one from a spreadsheet.** All cleaning is code: counted, the same for every file, repeatable by a reader. A zero `Flow Duration` is not a reason to drop a flow.
- **"The column is missing" and "the column cannot be read" are different faults.** A check must say which.
- **A check that precedes a long run must be able to stop it.** `all` now stops when `doctor` lists a problem.
- **A claim about a file comes from reading the file.** An old notebook output says what the file looked like then.
- **Ask what a person did to the data, then check the answer against a record.** Memory of a cleaning step is not a log of it.
- **A tool must not remove the only copy of something.** `report` keeps the tables of an experiment it has no run folder for.
- **"Nothing changed" needs evidence.** A run whose settings were not recorded is off-plan until shown otherwise.
- **Do not replace code under a running job.** Deliver between runs.
- **When a folder is cleared for new data, check `git status` afterwards** for tracked files that went with it.
- **A suite that has only run where `data/` is empty has not been shown to be hermetic.** Run it once with the project's folders populated, and compare them before and after.
- **A library call in a test names every folder it means.** A default that points at the project is a dependency on whatever the project holds that day.
- **A decision rule should name what it can be gamed by.** "Lower false alarms" can be reached by detecting nothing. A rule about one rate needs the other rate as a condition, or a sentence saying it has none.
- **When a smoke run on the real test day has been read, the plan is frozen.** What it shows goes into the plan as a disclosure, not as an edit to a rule.
- **Do not promise a duration that was never measured.** "A few minutes" for `--quick` came from the synthetic tests.
- **Smoke-test the code that will do the real run**, not the version before it. With `--seeds 1` the smoke run is shorter and exercises the same paths.
