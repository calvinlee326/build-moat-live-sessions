import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")  # matches "## My Heading"
TOKEN_RE = re.compile(r"[a-z0-9]+")                 # extracts alphanumeric words
STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from",
    "how", "i", "is", "it", "my", "of", "the", "to", "what", "when", "which",
}


# @dataclass auto-generates __init__, __repr__, etc. from the field annotations.
# It's a clean way to define a plain data container without boilerplate.
@dataclass
class Section:
    id: str              # e.g. "refund_policy.md#refund-timeline"
    file: str            # e.g. "refund_policy.md"
    heading: str         # e.g. "Refund Timeline"
    heading_path: list[str]  # breadcrumb: ["Refund Policy", "Refund Timeline"]
    content: str         # raw Markdown text under this heading
    tokens: list[str]    # pre-tokenized words for BM25 scoring

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


# Module-level globals act as an in-memory index.
# They're populated by build_index() and read by search().
sections: list[Section] = []
doc_freq: Counter[str] = Counter()  # how many sections contain each token
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    # Converts "Refund Timeline" -> "refund-timeline" for use in IDs/anchors
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    # Lowercase, extract words, remove stop words like "the", "is", "a"
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    """Split one Markdown file into heading-level Section records.

    Each heading starts a new section. Content is everything between
    this heading and the next one at the same or higher level.
    heading_path tracks the breadcrumb (parent headings) for context.
    """
    filename = path.name
    lines = path.read_text(encoding="utf-8").splitlines()

    result: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (heading_level, heading_text)
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush(heading: str, heading_path: list[str], body_lines: list[str]) -> None:
        content = "\n".join(body_lines).strip()
        # Tokenize heading + content together so heading words boost BM25 score
        all_tokens = tokenize(heading + " " + content)
        section_id = f"{filename}#{slugify(heading)}"
        result.append(Section(
            id=section_id,
            file=filename,
            heading=heading,
            heading_path=heading_path[:],  # copy so later mutations don't affect stored path
            content=content,
            tokens=all_tokens,
        ))

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            # Save the previous section before starting a new one
            if current_heading is not None:
                flush(current_heading, [h for _, h in heading_stack], current_lines)
            level = len(m.group(1))   # number of # symbols = heading depth
            heading_text = m.group(2)
            # Pop all headings at the same level or deeper (they're siblings/children)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_heading = heading_text
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section after the loop ends
    if current_heading is not None:
        flush(current_heading, [h for _, h in heading_stack], current_lines)

    return result


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    """Persist the in-memory section index to .kb/index.json.

    Writing to disk lets you inspect the index and reload it on restart
    without re-parsing all Markdown files.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)  # create .kb/ if missing
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
    """Recompute BM25 metadata from the current sections list.

    Called after loading or building the index so bm25_score() has
    accurate document frequency and average length figures.
    """
    global doc_freq, avg_doc_len, files_indexed
    doc_freq = Counter()
    for section in sections:
        # Use a set so each token counts once per section (document frequency, not term frequency)
        for token in set(section.tokens):
            doc_freq[token] += 1
    avg_doc_len = sum(len(s.tokens) for s in sections) / len(sections) if sections else 0.0
    files_indexed = len({s.file for s in sections})  # unique filenames


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    """Load .kb/index.json back into memory on server startup.

    Returns (files_indexed, sections_indexed) so the caller knows what was loaded.
    Returns (0, 0) if no index exists yet — caller should run POST /index first.
    """
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
    """Build the in-memory section index from all docs/*.md files.

    This is called by POST /index. It re-parses every Markdown file,
    recomputes BM25 stats, and overwrites .kb/index.json.
    """
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
    """Score how relevant a section is to the query using the BM25 algorithm.

    BM25 is the standard ranking function used by search engines like Elasticsearch.
    It rewards sections where query terms appear often (TF) and are rare across
    all sections (IDF), while penalizing very long sections that dilute term density.

    k1 controls TF saturation (higher = more weight to repeated terms).
    b controls length normalization (1.0 = full normalization, 0 = none).
    """
    if not query_tokens or not section.tokens:
        return 0.0

    N = len(sections)          # total number of sections in the index
    dl = len(section.tokens)   # length of this section in tokens
    tf_map = Counter(section.tokens)  # count of each token in this section
    score = 0.0

    for token in query_tokens:
        tf = tf_map.get(token, 0)
        if tf == 0:
            continue  # this query term doesn't appear in this section

        df = doc_freq.get(token, 0)
        # IDF: rare tokens score higher. +1 prevents log(0), +0.5 smooths edge cases.
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        # TF normalization: diminishing returns for repeated terms, adjusted for section length
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_doc_len))
        score += idf * tf_norm

        # Boost terms that appear in the heading breadcrumb — they're more topically central
        if token in tokenize(" ".join(section.heading_path)):
            score += 0.5

    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    # Score every section, sort descending, return top-k with non-zero scores
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
