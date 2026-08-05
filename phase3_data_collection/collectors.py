"""Collectors for public datasets, web pages, APIs, and manual labels."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


def download_file(url: str, destination: str | Path, sha256: str | None = None, timeout: int = 30) -> Path:
    """Download a public dataset or media file and optionally verify its SHA-256."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "ai-roadmap-data-collector/1.0"})
    with urlopen(request, timeout=timeout) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)

    if sha256:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest.lower() != sha256.lower():
            target.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {target}: expected {sha256}, got {digest}")
    return target


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)


def scrape_links(url: str, extensions: Iterable[str] | None = None, timeout: int = 30) -> list[str]:
    """Extract matching absolute links from an HTML page.

    ``extensions`` can contain values such as ``(".jpg", ".png")``. With no
    extensions, all links are returned.
    """
    request = Request(url, headers={"User-Agent": "ai-roadmap-data-collector/1.0"})
    with urlopen(request, timeout=timeout) as response:
        html = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")
    parser = _LinkParser()
    parser.feed(html)
    normalized_extensions = tuple(extension.lower() for extension in (extensions or ()))
    links = [urljoin(url, link) for link in parser.links]
    if normalized_extensions:
        links = [link for link in links if Path(urlparse(link).path).suffix.lower() in normalized_extensions]
    return list(dict.fromkeys(links))


def fetch_json_api(
    url: str,
    params: dict[str, str | int | float] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> Any:
    """Fetch and decode a JSON API response."""
    query = f"?{urlencode(params)}" if params else ""
    request_headers = {"Accept": "application/json", "User-Agent": "ai-roadmap-data-collector/1.0"}
    request_headers.update(headers or {})
    request = Request(f"{url}{query}", headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode(response.headers.get_content_charset() or "utf-8"))


@dataclass(frozen=True)
class AnnotationRecord:
    """One manually reviewed item in a dataset."""

    file_path: str
    label: str
    annotator: str = ""
    notes: str = ""


class AnnotationWriter:
    """Create a CSV annotation template and append reviewed records."""

    fieldnames = ["file_path", "label", "annotator", "notes"]

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.fieldnames).writeheader()

    def append(self, record: AnnotationRecord) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.fieldnames).writerow(asdict(record))
