---
name: summarize-paper
description: Summarize and explain academic papers, research documents, PDFs, pasted text, or paper URLs in Chinese using a structured, reader-friendly style. Use when the user asks to summarize a paper, explain a paper, extract research background, methods, experiments, results, contributions, limitations, or produce a Chinese paper-reading note modeled after a provided sample.
---

# Summarize Paper

## Core Workflow

1. Identify the paper source: uploaded document, local file path, pasted text, arXiv/DOI/URL, or user-provided excerpt.
2. Read enough of the paper to ground the summary. Prefer the full paper when available; otherwise state the limitation and summarize only the accessible content.
3. Extract the paper's problem, motivation, core method, experimental setup, key results, and practical meaning before writing.
4. Write in Chinese, following the user's sample style: direct, explanatory, and easy to read.
5. Preserve technical accuracy. Do not invent results, datasets, baselines, numbers, or claims that are not present in the paper.

## Output Structure

Use this structure by default unless the user asks for another format:

```markdown
## 研究背景：为什么要做这件事

[Explain the research motivation in plain Chinese. State the real problem the paper tries to solve and why it matters.]

## 核心方法

### 第一步：[方法步骤名]

[Explain the first major design choice or pipeline step.]

### 第二步：[方法步骤名]

[Explain the next component, model, training setup, data process, system design, or algorithm.]

### 第三步：[方法步骤名]

[Add or remove steps based on the paper. Keep the sequence faithful to the paper.]

## 实验结果

[Summarize datasets, baselines, metrics, main numbers, and what the results prove.]
```

Add optional sections when useful:

- `## 主要贡献`: Use for papers with multiple explicit contributions.
- `## 局限和问题`: Use when the user wants critique or when limitations are important.
- `## 一句话总结`: Use for a compact final takeaway.
- `## 术语解释`: Use when the paper contains dense jargon, formulas, or domain-specific concepts.

## Style Rules

- Prefer clear explanatory Chinese over literal translation.
- Keep paragraphs short and concrete.
- Use section titles that explain the point, not generic labels only.
- Use bullet points for comparisons, baselines, datasets, metrics, or multi-part results.
- When mentioning numbers, include the metric and comparison target.
- When the paper has formulas, explain what each important term means before discussing why the formula matters.
- When the paper has a system or pipeline, describe it as a sequence of decisions and data flow.
- Avoid academic filler such as "本文提出了一种新颖的方法" unless followed by the concrete mechanism.

## Handling Sources

- For a URL, open or fetch the source when tools are available. If the source is inaccessible, ask the user for the PDF or text.
- For a PDF or document, extract text from the actual file. If extraction may miss figures, tables, or formulas, mention that limitation.
- For pasted text, summarize only the pasted content unless the user asks to search for the full paper.
- If the user provides a sample format, follow that sample over the default structure.

## Quality Checklist

Before finalizing, verify:

- The summary states the paper's motivation and problem clearly.
- The method explanation follows the paper's actual order and does not skip the central mechanism.
- Experimental claims include enough context: dataset, baseline, metric, and direction of improvement.
- Unsupported claims are removed or clearly marked as inference.
- The final answer is useful to a reader who wants to understand the paper without reading it immediately.
