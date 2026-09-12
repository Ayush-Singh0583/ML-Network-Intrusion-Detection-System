# Knowledge Base Schema & Operational Guide

This repository contains an LLM Knowledge Base for the **ML Network Intrusion Detection System (ML-NIDS)**. This document specifies the schema, naming conventions, wiki page formats, wikilink syntax, and automatic ingestion rules for raw sources and session learnings.

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
├── CLAUDE.md                 # This schema, conventions, and processing instructions
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
