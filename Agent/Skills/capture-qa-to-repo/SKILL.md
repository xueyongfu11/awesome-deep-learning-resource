---
name: capture-qa-to-repo
description: Extract important question-and-answer knowledge from the current conversation, identify the most relevant Markdown file in the awesome-deep-learning-resource repository, detect and merge similar existing QA to avoid duplication, request user confirmation of the destination and planned change, then save it after explicit approval. Use when the user asks to 保存/沉淀/记录/追加关键 QA、FAQ、面试问答或技术问答到知识库/repo, especially when Codex must discover the repository, search existing topic files, deduplicate or consolidate related questions, choose an insertion point, preserve the file's style, and verify the resulting diff.
---

# Capture QA to Repo

Save durable technical knowledge from the conversation into the best existing file in the repository.

## Repository discovery

Try these paths in order and use the first existing Git worktree:

1. `D:\Codes\00-Synchronize-repo\awesome-deep-learning-resource`
2. `/work/awesome-deep-learning-resource`

Read [references/repo-index.md](references/repo-index.md) for the latest captured directory and topic map. If the snapshot is missing or stale, run:

```text
python scripts/repo_catalog.py --write-index references/repo-index.md
```

Use `python scripts/repo_catalog.py --query "<keywords>"` for an initial candidate ranking. Confirm candidates with `rg`; do not trust filename ranking alone.

## Workflow

### 1. Extract only key QA

- Identify questions whose answers contain reusable technical knowledge, decisions, explanations, troubleshooting steps, or interview-ready summaries.
- Rewrite conversational fragments into self-contained questions.
- Preserve technical accuracy; remove greetings, repetition, personal context, and unsupported claims.
- Combine duplicate or tightly coupled questions.
- Do not save secrets, credentials, private identifiers, or transient task chatter.
- If no durable QA exists, report that and do not edit the repo.

### 2. Find the best destination

Build search terms from the question, answer, aliases, English/Chinese names, libraries, model names, and error messages.

Search in this order:

1. Exact phrase and distinctive identifiers in file contents.
2. Topic terms in filenames and headings.
3. The directory/topic map in `references/repo-index.md`.
4. Broad concept matches in nearby files.

Prefer:

- An existing `常见问题*.md` or FAQ-style file for the same topic.
- Otherwise, the narrowest existing topic file.
- A new topic file only when no existing file is a defensible fit.
- `Other.md` only as a last resort.

Before editing, inspect the complete nearby section and at least the start of the file. Check `git status --short` and preserve unrelated user changes.

### 3. Detect similar QA and plan deduplication

Before proposing any write, search the entire repository for existing QA that may express the same intent.

Search using:

- The full proposed question and distinctive phrases.
- Core technical terms, aliases, acronyms, Chinese/English equivalents, model or library names, and error identifiers.
- Paraphrases and broader/narrower forms of the question.
- Relevant headings first, then answer bodies and nearby sections.

Compare candidates by meaning, not only exact wording:

- **Duplicate:** The existing QA asks the same question and already contains the proposed knowledge. Do not add another section. Propose no change, or only a small correction if the current answer is materially inaccurate.
- **Strong overlap:** The questions have the same core intent but each contains useful unique details. Propose updating one canonical QA by combining the best content and removing redundant wording.
- **Partial overlap:** The questions share context but answer distinct concerns. Keep separate QA sections, cross-reference or place them together when useful, and avoid repeating shared explanations.
- **Unrelated:** Add a new QA in the best destination.

When multiple duplicate or strongly overlapping QA sections already exist:

1. Select the clearest and best-located section as canonical.
2. Merge unique, accurate content into it.
3. Remove redundant sections only when doing so will not break links, indexes, or surrounding structure.
4. Preserve unique examples, caveats, code, and citations.
5. Do not silently discard conflicting claims; resolve them from available evidence or flag the conflict in the proposal.

Inspect every section that may be changed before requesting confirmation. Do not perform repository edits during this analysis.

### 4. Obtain user confirmation

Do not modify any repository file immediately after selecting a destination.

Present:

- The resolved repository path.
- The proposed destination file.
- Why this file is the best match.
- The QA question titles or a concise preview of the content to insert.
- Similar QA found, including their file paths and headings, or explicitly state that none were found.
- The planned operation: no change, add a new QA, merge into an existing QA, or consolidate multiple existing QA sections.
- For a merge, summarize what existing content remains, what new content is added, and what duplication is removed.

Ask the user for explicit confirmation. Treat clear responses such as “确认”“可以”“写入” or an unambiguous equivalent as approval.

- Do not interpret the original request to save QA as confirmation of the subsequently selected file.
- Do not edit while waiting for confirmation.
- If the user changes the destination or content, revise the proposal and request confirmation again.
- Approval applies only to the displayed destination and proposed QA. Material changes require new confirmation.

### 5. Match repository style

- Preserve the file's encoding, line endings, heading depth, language, and formatting conventions.
- Use the dominant QA form in the destination file. Commonly:

```markdown
## 问题？

直接回答，随后给出必要解释、示例和一句话总结。
```

- Avoid adding metadata or a new schema that the file does not already use.
- Apply the approved deduplication plan. Keep one canonical answer for duplicate or strongly overlapping QA.
- Keep links and code blocks valid. Do not invent citations.

### 6. Edit safely

- Proceed only after receiving explicit confirmation for the displayed destination and planned QA.
- Use `apply_patch` for targeted edits.
- Insert near semantically related questions, not merely at the end, unless the file is an append-only QA collection.
- Do not reformat the whole file.
- Do not modify generated files, binary assets, or unrelated notes.
- Do not commit, push, or open a PR unless the user explicitly requests it.

### 7. Verify

After editing:

1. Re-read the inserted section in context.
2. Run `git diff --check`.
3. Inspect `git diff -- <file>`.
4. Confirm the question is searchable with `rg`.
5. Search again for the main question terms and confirm no unintended duplicate QA remains.
6. Report the repo path, changed file, added or merged QA titles, deduplication result, and validation result.

If two destination files remain equally plausible, avoid writing and ask the user to choose between the concrete paths.

## Refreshing the index

Whenever repository structure changes materially, run the catalog script and include the refreshed `references/repo-index.md` in the skill update. The index is navigation assistance, not a replacement for searching current repository contents.
