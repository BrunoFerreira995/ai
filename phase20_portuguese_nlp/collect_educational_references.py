#!/usr/bin/env python3
"""Collect 60 Crossref article records for every educational subtopic.

Only bibliographic metadata is collected. Article full text is not downloaded.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from urllib.error import URLError
from pathlib import Path

from .education import EDUCATIONAL_KNOWLEDGE


def educational_topics() -> list[tuple[str, str]]:
    return [
        (subject, topic)
        for subject, content in EDUCATIONAL_KNOWLEDGE.items()
        for topic in content["topics"]
    ]


def crossref_records(subject: str, topic: str, limit: int, delay: float, timeout: int = 20) -> list[dict[str, str]]:
    query = f"{subject} {topic} education"
    params = urllib.parse.urlencode({
        "query.bibliographic": query,
        "filter": "type:journal-article",
        "rows": min(limit * 2, 200),
        "select": "DOI,title,author,published,URL,type",
        "sort": "relevance",
        "order": "desc",
    })
    request = urllib.request.Request(
        f"https://api.crossref.org/works?{params}",
        headers={"User-Agent": "ai-educational-references/1.0 (mailto:research@example.org)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, URLError, OSError) as error:
        print(f"  aviso: Crossref indisponível para {topic}: {error}")
        return []
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in payload.get("message", {}).get("items", []):
        title = " ".join(item.get("title", []))
        doi = str(item.get("DOI", "")).strip()
        if not title or not doi or doi.lower() in seen:
            continue
        seen.add(doi.lower())
        authors = ", ".join(
            " ".join(part for part in (author.get("given"), author.get("family")) if part)
            for author in item.get("author", [])[:8]
        )
        date = item.get("published", {}).get("date-parts", [[]])[0]
        records.append({
            "title": title,
            "authors": authors,
            "year": str(date[0]) if date else "",
            "doi": doi,
            "url": item.get("URL", f"https://doi.org/{doi}"),
            "subject": subject,
            "topic": topic,
        })
        if len(records) == limit:
            break
    time.sleep(delay)
    return records


def collect(output: str | Path, limit: int = 60, delay: float = 0.2) -> dict[str, list[dict[str, str]]]:
    path = Path(output)
    result: dict[str, list[dict[str, str]]] = {}
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
    for index, (subject, topic) in enumerate(educational_topics(), 1):
        key = f"{subject} — {topic}"
        if len(result.get(key, [])) >= limit:
            print(f"[{index}/{len(educational_topics())}] {key} já concluído")
            continue
        print(f"[{index}/{len(educational_topics())}] {key}")
        result[key] = crossref_records(subject, topic, limit, delay)
        print(f"  {len(result[key])} artigos encontrados")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta referências científicas por subtema educacional")
    parser.add_argument("--output", default="data/educational_references_60.json")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    result = collect(args.output, args.limit, args.delay)
    total = sum(len(items) for items in result.values())
    print(f"Referências salvas: {total} em {args.output}")


if __name__ == "__main__":
    main()
