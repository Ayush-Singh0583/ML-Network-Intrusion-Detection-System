---
trigger: always_on
---

# Knowledge Base Maintenance Rule

Whenever files are added or modified in `raw/`:
1. Read the raw source document thoroughly.
2. Structure new or updated topics as modular markdown files inside `wiki/` following the schema in `CLAUDE.md`.
3. Interlink all related concepts using `[[Wikilinks]]`.
4. Update `wiki/Index.md` with links and concise descriptions for all new/modified wiki pages.
5. Append any operational gotchas, successes, and domain rules to `learnings.md`.
