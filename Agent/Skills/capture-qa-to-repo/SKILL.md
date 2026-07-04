---
name: capture-qa-to-repo
description: Extract durable question-and-answer knowledge or user-provided URL titles and addresses, find or create the most relevant Markdown file and section in the awesome-deep-learning-resource repository, deduplicate existing content, request confirmation, and save it safely. Use when the user asks to 保存/沉淀/记录/追加关键 QA、FAQ、面试问答、技术问答、链接、网址、URL 或资源到知识库/repo, especially when Codex must select a destination, create a missing topic file or section for a URL, preserve existing formatting, and verify the resulting diff.
---

# Capture QA and URLs to Repo

Save durable technical knowledge or useful URLs into the best location in the repository.

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

### 1. Normalize the requested content

Determine whether the request contains QA, URLs, or both.

For QA:

- Identify questions whose answers contain reusable technical knowledge, decisions, explanations, troubleshooting steps, or interview-ready summaries.
- Rewrite conversational fragments into self-contained questions.
- Preserve technical accuracy; remove greetings, repetition, personal context, and unsupported claims.
- Combine duplicate or tightly coupled questions.

For URLs:

- Capture each user-provided title and address as one entry.
- Preserve the address exactly unless the user explicitly requests normalization.
- Do not invent a missing title or address. Ask for the missing value when it cannot be recovered unambiguously from the request.
- Reject credentials, access tokens, private identifiers, or other secrets embedded in a URL. Do not save them to the repository.

Do not save transient task chatter. If the request contains neither durable QA nor a complete URL entry, report that and do not edit the repo.

### 2. Find the best destination

Build search terms from the QA or URL title, plus relevant aliases, English/Chinese names, libraries, model names, error messages, domains, and topic concepts.

Search in this order:

1. Exact phrase and distinctive identifiers in file contents.
2. Topic terms in filenames and headings.
3. The directory/topic map in `references/repo-index.md`.
4. Broad concept matches in nearby files.

Prefer:

- An existing `常见问题*.md` or FAQ-style file for the same topic.
- Otherwise, the narrowest existing topic file.
- For a URL, identify both the narrowest matching file and the most specific matching section.
- For a URL with no matching section in the best file, plan to create a clearly named section in that file.
- For a URL with no defensible destination file, plan to create a narrowly named topic file and an appropriate section in it.
- For QA, create a new topic file only when no existing file is a defensible fit.
- `Other.md` only as a last resort.

Do not treat a broad file or section as corresponding merely because it can contain anything. Choose based on the URL title and subject. If two destinations remain equally plausible, ask the user to choose between the concrete paths.

Before editing or proposing a new file or section, inspect the complete nearby section, the file's heading hierarchy, and at least the start of the file. Check `git status --short` and preserve unrelated user changes.

### 3. Detect duplicates and plan the operation

Before proposing any write, search the entire repository for duplicate or overlapping content.

For QA, search using:

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

For URLs, search the entire repository for:

- The exact address, including likely normalized variants only for duplicate detection.
- The exact title and meaningful title phrases.
- Other links in the candidate file and section that point to the same resource.

Classify URL matches as follows:

- **Same resource already present:** Do not add a duplicate. Propose no change, or propose correcting the existing title/address if the user explicitly supplied a replacement.
- **Same title, different address:** Surface the conflict and propose whether to update or retain both; do not decide silently.
- **New resource:** Propose inserting it into the matching section, creating the section, or creating the file and section as determined in step 2.

Inspect every section that may be changed before requesting confirmation. Do not perform repository edits during this analysis.

### 4. Obtain user confirmation

Do not modify any repository file immediately after selecting a destination.

Present:

- The resolved repository path.
- The proposed destination file.
- Why this file is the best match.
- The destination section for every URL, and whether the file or section will be created.
- The QA question titles, or each URL title and address, as a concise preview of the content to insert.
- Similar QA or duplicate URLs found, including their file paths and headings, or explicitly state that none were found.
- The planned operation: no change, add or merge QA, consolidate QA sections, insert a URL, create a section, or create a file and section.
- For a merge, summarize what existing content remains, what new content is added, and what duplication is removed.

Ask the user for explicit confirmation. Treat clear responses such as “确认”“可以”“写入” or an unambiguous equivalent as approval.

- Do not interpret the original request to save content as confirmation of the subsequently selected file and section.
- Do not edit while waiting for confirmation.
- If the user changes the destination or content, revise the proposal and request confirmation again.
- Approval applies only to the displayed destination and proposed QA, or to the displayed URL title, address, destination file, destination section, and creation plan. Material changes require new confirmation.

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

For each URL:

- Match the existing link-entry format used in the destination section, including its bullet marker, annotations, and surrounding spacing.
- If the destination section has no established link-entry format, use exactly `- [标题](地址)`.
- When the section already exists, insert the URL entry at the very beginning of the section body, immediately after its heading and required blank line, before existing prose or entries.
- When creating a section, match the file's heading depth and naming style, then add the URL as its first entry.
- When creating a file, match nearby filenames and document structure. Add a suitable title/section hierarchy and place the URL under the relevant section.

### 6. Edit safely

- Proceed only after receiving explicit confirmation for the displayed destination and planned QA or URL operation.
- Use `apply_patch` for targeted edits.
- Insert near semantically related questions, not merely at the end, unless the file is an append-only QA collection.
- Apply the URL placement rules from step 5 even when another insertion point would be more convenient.
- Do not reformat the whole file.
- Do not modify generated files, binary assets, or unrelated notes.
- Do not commit, push, or open a PR unless the user explicitly requests it.

### 7. Verify

After editing:

1. Re-read the inserted section in context.
2. Run `git diff --check`.
3. Inspect `git diff -- <file>`.
4. Confirm each saved question or exact URL is searchable with `rg`.
5. Search again for the main question terms or URL/title and confirm no unintended duplicate remains.
6. For URLs, confirm the entry is the first content in the intended section and matches the approved format.
7. Report the repo path, changed or created file and section, saved QA titles or URLs, deduplication result, and validation result.

## Refreshing the index

Whenever repository structure changes materially, run the catalog script and include the refreshed `references/repo-index.md` in the skill update. The index is navigation assistance, not a replacement for searching current repository contents.
