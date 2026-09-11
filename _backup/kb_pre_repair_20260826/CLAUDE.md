# Knowledge Base Schema & Operational Guide

This repository contains an LLM Knowledge Base for the **ML Network Intrusion Detection System (ML-NIDS)**. This document specifies the schema, naming conventions, wiki page formats, wikilink syntax, and automatic ingestion rules for raw sources and session learnings.

---

## 📁 Knowledge Base Directory Architecture

```
ML-Network-Intrusion-Detection-System/
├── raw/                      # Raw, unedited source files (reports, specs, notes, papers)
├── wiki/                     # Structured, atomic, interlinked knowledge graph pages
│   ├── Index.md              # Master knowledge index & navigation graph
│   └── *.md                  # Topic pages cross-referenced via [[Wikilinks]]
├── learnings.md              # Running log of what works, what fails, and engineering insights
├── CLAUDE.md                 # This schema, conventions, and processing instructions
└── .agents/
    ├── hooks.json            # Automated agent lifecycle hook trigger
    └── rules/                # Active behavioral rules for wiki maintenance
```

---

## 📑 Wiki Page Schema & Formatting Conventions

Every page in `wiki/` must adhere to this standard layout:

```markdown
---
title: "<Page Title>"
category: "<Architecture | ML | Security | Data | Backend | Frontend>"
sources:
  - "raw/<source_file_1>.md"
tags: ["<tag1>", "<tag2>"]
updated: YYYY-MM-DD
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

### Wikilink Rules
1. **Syntax**: Use `[[Page-Name]]` (e.g., `[[System-Architecture]]`, `[[Feature-Engineering-Timing]]`).
2. **Precision Anchor Links**: Use `[[Page-Name#Section-Heading|Display Text]]` when referencing specific subsections.
3. **Bidirectional Linking**: Whenever a concept is introduced on page A that relates to page B, both pages should reference each other via wikilinks.
4. **Index Registry**: Every new page must be indexed with a one-line summary in `[[Index]]`.

---

## 🔄 Raw Source Ingestion Protocol

When a new source file is dropped into `raw/` (or detected via hooks):
1. **Analyze & Extract**: Read the raw source, identify core entities, architectural decisions, mathematical formulas, and breaking changes.
2. **Decompose & Atomize**: Determine if the information belongs to an existing wiki page (update it) or requires new atomic pages (create them).
3. **Enforce Wikilinks**: Embed `[[Wikilinks]]` throughout the text for every referenced concept, dataset, model, metric, or pipeline stage.
4. **Update Master Graph**: Add/update the topic entry in `wiki/Index.md`.
5. **Record Session Learnings**: Extract what worked, what failed, and key takeaways into `learnings.md`.

---

## 📝 `learnings.md` Session Logging Convention

After every development or analysis session, append an entry using the following template:

```markdown
## [YYYY-MM-DD] - <Session Topic / Feature>
### ✅ What Worked
- High-level decision, optimal hyperparameter, or architectural pattern that succeeded.
### ❌ What Failed / Gotchas
- Bugs encountered, invalid assumptions, or security pitfalls discovered.
### 💡 Actionable Rule for Next Sessions
- Concrete rule to prevent regressions (e.g., "Always convert flow IAT to microseconds before scaling").
```

---

## 🛡️ Core Project Domain Invariants

- **Shortcut Learning**: Always omit `Destination Port` and network identifiers (`Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Timestamp`) from ML features (`[[Shortcut-Learning-Port-Bias]]`).
- **Timing Synchronization**: Live network packet statistics must be scaled in **microseconds ($\mu s$)**, not seconds, to match `CICFlowMeter` training distributions (`[[Feature-Engineering-Timing]]`).
- **Evaluation Integrity**: Always benchmark models using unweighted **Macro Precision, Macro Recall, and Macro F1**, rather than overall Accuracy or Weighted F1 (`[[Model-Evaluation-Metrics]]`).
- **Open-Set Rejection**: Use maximum posterior confidence thresholding ($\theta = 0.55$) to detect zero-day or out-of-distribution traffic (`[[Open-Set-Recognition]]`).
- **4-File Model Bundling**: Every saved model must export `<model>_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `feature_names.pkl` in sync (`[[Machine-Learning-Models]]`).
