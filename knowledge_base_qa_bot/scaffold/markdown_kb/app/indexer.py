import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    filename = path.name
    lines = path.read_text(encoding="utf-8").splitlines()

    result: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush(heading: str, heading_path: list[str], body_lines: list[str]) -> None:
        content = "\n".join(body_lines).strip()
        all_tokens = tokenize(heading + " " + content)
        section_id = f"{filename}#{slugify(heading)}"
        result.append(Section(
            id=section_id,
            file=filename,
            heading=heading,
            heading_path=heading_path[:],
            content=content,
            tokens=all_tokens,
        ))

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if current_heading is not None:
                flush(current_heading, [h for _, h in heading_stack], current_lines)
            level = len(m.group(1))
            heading_text = m.group(2)
            # Pop stack entries at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_heading = heading_text
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None:
        flush(current_heading, [h for _, h in heading_stack], current_lines)

    return result


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [s.to_dict() for s in sections],
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": len(sections),
            "avg_doc_len": avg_doc_len,
        },
    }
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def rebuild_stats() -> None:
    global doc_freq, avg_doc_len, files_indexed
    doc_freq = Counter()
    for section in sections:
        for token in set(section.tokens):
            doc_freq[token] += 1
    avg_doc_len = sum(len(s.tokens) for s in sections) / len(sections) if sections else 0.0
    files_indexed = len({s.file for s in sections})


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    global sections
    if not index_path.exists():
        return 0, 0
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    sections = [
        Section(
            id=item["id"],
            file=item["file"],
            heading=item["heading"],
            heading_path=item["heading_path"],
            content=item["content"],
            tokens=item["tokens"],
        )
        for item in payload["sections"]
    ]
    rebuild_stats()
    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections, doc_freq, avg_doc_len, files_indexed

    sections = []
    doc_freq = Counter()
    avg_doc_len = 0.0
    files_indexed = 0

    for md_file in sorted(docs_dir.glob("*.md")):
        sections.extend(parse_markdown(md_file))

    rebuild_stats()
    write_index_json()
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_tokens or not section.tokens:
        return 0.0

    N = len(sections)
    dl = len(section.tokens)
    tf_map = Counter(section.tokens)
    score = 0.0

    for token in query_tokens:
        tf = tf_map.get(token, 0)
        if tf == 0:
            continue
        df = doc_freq.get(token, 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_doc_len))
        score += idf * tf_norm

        # Small heading boost
        if token in tokenize(" ".join(section.heading_path)):
            score += 0.5

    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
