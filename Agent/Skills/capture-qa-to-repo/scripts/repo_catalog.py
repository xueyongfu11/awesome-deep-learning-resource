#!/usr/bin/env python3
"""Discover the knowledge repo, rank relevant Markdown files, and write a directory index."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import date
from pathlib import Path


CANDIDATES = (
    Path(r"D:\Codes\00-Synchronize-repo\awesome-deep-learning-resource"),
    Path("/work/awesome-deep-learning-resource"),
)
MARKDOWN_SUFFIXES = {".md", ".mdx"}
SKIP_DIRS = {".git", "assets", "node_modules"}


def find_repo(explicit: str | None) -> Path:
    candidates = (Path(explicit),) if explicit else CANDIDATES
    for path in candidates:
        if path.is_dir() and (path / ".git").exists():
            return path.resolve()
    joined = "\n".join(f"- {path}" for path in candidates)
    raise SystemExit(f"No repository found. Checked:\n{joined}")


def markdown_files(repo: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in repo.rglob("*")
            if path.is_file()
            and path.suffix.lower() in MARKDOWN_SUFFIXES
            and not any(part in SKIP_DIRS for part in path.relative_to(repo).parts)
        ),
        key=lambda path: path.relative_to(repo).as_posix().lower(),
    )


def read_text(path: Path, limit: int = 200_000) -> str:
    raw = path.read_bytes()[:limit]
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def headings(path: Path, maximum: int = 8) -> list[str]:
    found = []
    for line in read_text(path, 80_000).splitlines():
        match = re.match(r"^#{1,4}\s+(.+?)\s*$", line)
        if match:
            found.append(match.group(1))
            if len(found) >= maximum:
                break
    return found


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9+_.-]*|[\u4e00-\u9fff]{2,}", text.lower())


def rank(repo: Path, query: str) -> list[tuple[int, Path]]:
    query_tokens = Counter(tokens(query))
    results = []
    for path in markdown_files(repo):
        relative = path.relative_to(repo).as_posix()
        path_text = relative.lower()
        heading_text = " ".join(headings(path)).lower()
        body_text = read_text(path, 120_000).lower()
        score = 0
        for token, count in query_tokens.items():
            score += count * (
                12 * path_text.count(token)
                + 7 * heading_text.count(token)
                + min(5, body_text.count(token))
            )
        if score:
            results.append((score, path))
    return sorted(results, key=lambda item: (-item[0], item[1].as_posix().lower()))


def build_index(repo: Path) -> str:
    files = markdown_files(repo)
    grouped: dict[str, list[Path]] = {}
    for path in files:
        relative = path.relative_to(repo)
        top = relative.parts[0] if len(relative.parts) > 1 else "(root)"
        grouped.setdefault(top, []).append(relative)

    lines = [
        "# Repository index",
        "",
        f"- Snapshot date: `{date.today().isoformat()}`",
        f"- Resolved Windows path: `{repo}`",
        "- Alternate Linux path: `/work/awesome-deep-learning-resource`",
        f"- Markdown files indexed: `{len(files)}`",
        "",
        "Use this snapshot for navigation, then search the live repo before editing.",
        "",
        "## Directory and Markdown file map",
        "",
    ]
    for group in sorted(grouped, key=str.lower):
        lines.append(f"### {group}")
        lines.append("")
        for relative in grouped[group]:
            title_list = headings(repo / relative, maximum=2)
            title = f" — {' / '.join(title_list)}" if title_list else ""
            lines.append(f"- `{relative.as_posix()}`{title}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Explicit repository path")
    parser.add_argument("--query", help="Rank files relevant to these keywords")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--write-index", help="Write a Markdown repository index")
    args = parser.parse_args()

    repo = find_repo(args.repo)
    print(f"repo: {repo}")

    if args.query:
        for score, path in rank(repo, args.query)[: args.limit]:
            print(f"{score:4d}  {path.relative_to(repo).as_posix()}")

    if args.write_index:
        output = Path(args.write_index)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_index(repo), encoding="utf-8", newline="\n")
        print(f"index: {output.resolve()}")

    if not args.query and not args.write_index:
        print(build_index(repo), end="")


if __name__ == "__main__":
    main()
