---
title: "Dataset CICIDS2017 Benchmark"
category: "Data"
sources:
  - "raw/IMPLEMENTATION_REPORT.md"
  - "raw/PROJECT_DOCUMENTATION.md"
code_refs:
  - "src/preprocessing.py"
  - "src/config.py"
tags: ["cicids2017", "dataset", "features", "attacks", "benchmark"]
status: stale
updated: 2026-10-05
---

# Dataset CICIDS2017 Benchmark

> **⚠️ CORRECTION (2026-08-26) — split details and feature count.**
>
> - The random split was **70/30**, not 80/20, and it is no longer the default protocol —
>   it leaks. The primary protocol is now Mon–Wed + half of Thursday / half of Thursday /
>   Friday. See [[Training-Protocol]].
> - "Limitation: *can* artificially inflate performance if temporal leakage occurs" is
>   understated. It **did**: 99.88% accuracy under the random split versus 0.64 accuracy
>   and 0.00085 ATTACK recall on the same pipeline under a day split. See
>   [[Data-Leakage-Audit]].
> - The "78+ dimensional" feature count is now **62**. Seven feature pairs in this dataset
>   are byte-identical duplicates of one another, and one column header is literally
>   duplicated in the source CSVs. See [[Data-Leakage-Audit]].
> - Friday's DDoS is **not** a clean unknown: it overlaps the Wednesday DoS-Hulk cloud in
>   feature space. A "Friday = unknown" ground truth penalises the detector for correctly
>   recognising traffic it has genuinely seen. The defensible framing treats PortScan and
>   Bot as the true unknowns. See [[Rejection-Scoring]].
>
> The attack taxonomy table and the imbalance figures below remain accurate and useful.

> **⚠️ CORRECTION (2026-10-04) — which download, and what is in it.**
>
> `data/Readme.md` told the reader to download the `MachineLearningCSV` set. That set has
> **no source address and no timestamp** (79 columns). The same eight files in
> `GeneratedLabelledFlows.zip` have 85 columns: `Flow ID`, `Source IP`, `Source Port`,
> `Destination IP`, `Protocol` and `Timestamp` are added. (Whether the flow features of
> the two downloads are value-for-value identical has not been checked here; `doctor`
> prints the row and label counts to compare.) The behaviour layer, timestamp ordering and
> minute-block confidence intervals in
> [[Three-Layer-Study]] need the second one. `python src/study.py doctor` reports which is
> on disk and flags a mixture (this project's `data/` held seven files of one and one of
> the other).
>
> Three properties widely reported for the `GeneratedLabelledFlows` files that break a
> plain `pd.read_csv`. All three are handled in `preprocessing.read_capture` /
> `meta.parse_cic_timestamps` and covered by `tests/test_meta.py` on synthetic files that
> reproduce them; none has been observed on this project's disk yet, because the files
> have not been downloaded:
>
> - the Thursday-morning file writes its label dash as byte `0x96` (Windows-1252), which is
>   not valid UTF-8;
> - the same file ends in a long run of rows that are empty in every column;
> - timestamps are a 12-hour clock with no AM/PM marker, to the minute on most days.
>
> The one file of that distribution already here (`Friday-...-DDos`) does show the 85
> columns and the `7/7/2017 3:30` timestamp style (`notebooks/01_dataset_exploration.ipynb`).
> Exact row and column counts are whatever `doctor` prints; they are not asserted here.

> **⚠️ CORRECTION (2026-10-05) — the Friday DDoS file in `data/` is not the file as downloaded.**
>
> The last sentence of the block above was true of the notebook's first reading of that
> file. It is not true of the file that is in `data/` now. What is on record (the notebook
> reads the file twice; the numbers in brackets are its cell run counts):
>
> | where | rows | timestamps | negative `Flow Duration` | labels |
> | :--- | :--- | :--- | :--- | :--- |
> | notebook, first reading (runs 1–13) | 225,745 | `7/7/2017 3:30` | minimum is −1 | not printed |
> | notebook, second reading (runs 17–27) | 225,743 | not printed | none | DDoS 128,027, BENIGN 97,716 |
> | `doctor`, 2026-10-04 (`runs/96e8769-20261004-231545-study-doctor`) | 225,743 | `07-07-2017 03:30` | not checked | DDoS 128,027, BENIGN 97,716 |
>
> The file's modification date is 4 July 2026; the other seven files in `data/` carry the
> date 7 June 2018. So between the notebook's two readings the file lost two rows and its
> negative durations, and when `doctor` read it its timestamps had dashes and leading
> zeros.
> **What Ayush said when asked (2026-10-05):** he had cleaned the data by hand, removing
> rows with empty values, negative times or a zero duration. The notebook agrees on the
> negative durations and on nothing else: exactly two rows are gone, the four empty
> `Flow Bytes/s` values are still there in the second reading, and two rows cannot be all
> the zero-duration flows.
> **Inference, not a measured fact:** the edit was made in a spreadsheet program and the
> file was saved from there, which rewrote the date column (a spreadsheet writes dates in
> the computer's regional short-date style when it saves a CSV, and day-month-year with
> dashes is a common one). Nothing recorded says whether other columns changed too; a
> spreadsheet can shorten long decimals when it saves.
>
> Consequences:
> - `meta.parse_cic_timestamps` reads none of those timestamps, and `doctor` reports the
>   column as `UNREADABLE`. The parser was deliberately not widened: the fix is the
>   original file.
> - The baseline run of 12 September 2026 is later than the file's modification date, so
>   it most likely read this edited file. The two missing rows are rows the pipeline drops
>   anyway (it removes negative durations itself; that filter was in the code as found on
>   2026-10-04, and whether the September code had it was not checked). Whether the file's
>   feature values differ from the original's was not checked either.
> - The other seven files in `data/` are `MachineLearningCSV` files (79 columns). The
>   study needs all eight from `GeneratedLabelledFlows.zip`; see `data/Readme.md`.
> - Rule for this dataset: **never clean a capture by hand, and never open one in a
>   spreadsheet and save it.** Rows are filtered in code, where the filter is written down
>   and counted: `preprocessing._clean_frame` drops a negative or unreadable
>   `Flow Duration` and a blank label; `FrameCleaner` keeps a row with an empty or infinite
>   value and fills it with the training split's median. A zero duration is **not** a reason
>   to drop a flow: it is normally a one-packet flow, which a deployed detector sees too.
>   (How many attack flows have a zero duration here has not been measured.)

> **⚠️ CORRECTION (2026-10-05, later) — the eight `GeneratedLabelledFlows` files are now in
> `data/`, and what the 2026-10-04 block called "widely reported" has been measured.**
>
> That block says none of the three file properties "has been observed on this project's
> disk yet". All three now have, by `doctor` on the real files
> (`runs/96e8769-20261005-011910-study-doctor`, no `[problem]` line):
>
> | capture | labelled rows | timestamp as printed | resolution | hours it parses onto | consecutive rows that go forward in time |
> | :--- | ---: | :--- | :--- | :--- | ---: |
> | Monday | 529,918 | `03/07/2017 08:55:58` | second | 08:55–17:01 | 60.6% |
> | Tuesday | 445,909 | `4/7/2017 8:54` | minute | 08:53–17:00 | 64.2% |
> | Wednesday | 692,703 | `5/7/2017 8:42` | minute | 08:42–17:10 | 76.3% |
> | Thursday morning | 170,366 | `6/7/2017 8:59` | minute | 08:59–12:59 | 100% |
> | Thursday afternoon | 288,602 | `6/7/2017 1:00` | minute | 13:00–17:04 | 100% |
> | Friday morning | 191,033 | `7/7/2017 8:59` | minute | 08:59–12:59 | 100% |
> | Friday PortScan | 286,467 | `7/7/2017 1:00` | minute | 13:00–15:29 | 100% |
> | Friday DDoS | 225,745 | `7/7/2017 3:30` | minute | 15:30–17:02 | 100% |
>
> - **2,830,743 labelled rows** in all; every capture has 85 columns; 100% of timestamps
>   parse and all fall on the capture's own date.
> - **The label dash**: the Thursday-morning file is not valid UTF-8 and is read as
>   Windows-1252; its labels come out as `Web Attack – Brute Force` and so on.
> - **The blank rows**: the same file ends in 288,602 rows with no label, which are dropped.
> - **The clock**: afternoon files print `1:00` to `5:04`. Read with "an hour below 8 is
>   afternoon" every capture lands inside 08:42–17:10, and the five captures that are in
>   time order stay in time order. Only Monday is stamped to the second.
> - **File order is not time order on Monday, Tuesday and Wednesday.** Anything that needs
>   "earlier" and "later" must sort by the timestamp (`meta.time_order`), never trust the
>   row order.
> - **The two downloads agree on what they contain**, as far as `doctor` compares them: the
>   seven `MachineLearningCSV` files that were here on 2026-10-04 had the same labelled-row
>   count and the same per-label counts as their `GeneratedLabelledFlows` namesakes (the
>   web-attack labels differ only in the dash character). Feature values were not compared.
> - The Friday DDoS file is back to 225,745 rows (BENIGN 97,718, DDoS 128,027): two BENIGN
>   rows more than the edited copy described above.
> - **What the cleaner removes, measured** (cache built from these files on 2026-10-05;
>   `cache/clean_parts/_state.json`): 115 of the 2,830,743 rows, all labelled BENIGN, which
>   leaves 2,830,628. Every attack class keeps every flow. From the Friday DDoS capture it
>   removes exactly 2 rows and keeps 225,743 (DDoS 128,027, BENIGN 97,716): the row and
>   label counts of the hand-edited copy. So the hand edit took out the two flows of
>   negative duration and nothing else, and the cleaned capture has the same number of
>   flows whichever copy it starts from. Whether the edited copy's feature values differed
>   is still unchecked.


## 📌 Executive Summary
The **CICIDS2017** dataset is the primary training and evaluation benchmark for this project. Developed by the Canadian Institute for Cybersecurity, it contains realistic benign background traffic alongside common cyber-attacks collected over five days (Monday through Friday). Features were originally computed using `CICFlowMeter` in bidirectional 78+ dimensional statistical summaries.

---

## 🧠 Traffic Composition & Attack Taxonomy

The dataset captures diverse multi-class intrusion profiles across weekday captures:

| Capture Day | Traffic Profiles & Attacks Included | Class Balance |
| :--- | :--- | :--- |
| **Monday** | Pure Benign Normal Activity | 100% Benign |
| **Tuesday** | FTP-Patator (Brute Force), SSH-Patator | Benign + Brute Force |
| **Wednesday** | DoS (GoldenEye, Hulk, SlowHTTPTest, Slowloris), Heartbleed | Benign + DoS Attacks |
| **Thursday** | Web Attacks (Brute Force, XSS, Sql Injection), Infiltration | Benign + Web/Minority Intrusions |
| **Friday** | Botnet, PortScan, DDoS | Benign + Volumetric Attacks |

---

## 📐 Splitting Strategies: Random Split vs. Cross-Day Split

The codebase supports two distinct experimental evaluation paradigms in `src/preprocessing.py`:

### 1. Standard Stratified Random Split (`split_dataset`)
- Shuffles Monday–Friday data uniformly into 80% train and 20% test splits.
- Preserves relative class proportions across all known attack types.
- **Limitation**: Can artificially inflate performance if temporal leakage occurs across identical IP session streams.

### 2. Cross-Day Split (`split_dataset_by_day`)
- Trains exclusively on **Monday–Thursday** captures and tests against unseen **Friday** traffic (or vice versa).
- Mimics real-world deployment where tomorrow's attack vectors (e.g. novel Botnets or DDoS floods) have never been seen by the classifier.
- Interacts directly with [[Open-Set-Recognition]] to test zero-day rejection.

---

## ⚠️ Security Vulnerabilities & Dataset Biases
1. **Port-to-Attack Correlation**: CICIDS2017 attack scripts targeted static predefined ports (e.g. DoS on 80, Patator on 21/22). Failing to drop target ports results in [[Shortcut-Learning-Port-Bias]].
2. **Extreme Class Imbalance**: Benign traffic constitutes >80% of samples, while severe threats like `Heartbleed` (~11 samples) and `SQL Injection` (~21 samples) are heavily under-represented, necessitating balanced loss functions in [[Machine-Learning-Models]].
3. **Temporal Unit Conventions**: All duration and inter-arrival features are reported in **microseconds ($\mu s$)**, which must be strictly adhered to during live capture ([[Feature-Engineering-Timing]]).

---

## 🔗 Related Topics (Wikilinks)
- [[Shortcut-Learning-Port-Bias]] — The necessity of removing `Destination Port`.
- [[Feature-Engineering-Timing]] — Feature definitions and microsecond scaling.
- [[Open-Set-Recognition]] — Handling novel Friday attack classes during cross-day evaluation.
- [[Model-Evaluation-Metrics]] — Evaluating on imbalanced attack distributions.
- [[Three-Layer-Study]] — Which CSV distribution each experiment needs, and the `doctor` command that checks it.
- [[Index]] — Master Knowledge Graph Index.
