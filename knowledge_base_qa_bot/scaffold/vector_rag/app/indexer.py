import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings


def load_markdown_sections(path: Path) -> list[Document]:
    filename = path.name
    lines = path.read_text(encoding="utf-8").splitlines()
    docs: list[Document] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush(heading: str, body_lines: list[str]) -> None:
        content = "\n".join(body_lines).strip()
        if not content:
            return
        breadcrumb = " > ".join(h for _, h in heading_stack)
        section_id = f"{filename}#{slugify(heading)}"
        docs.append(Document(
            page_content=f"{breadcrumb}\n\n{content}",
            metadata={"source": section_id, "heading": breadcrumb, "file": filename},
        ))

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if current_heading is not None:
                flush(current_heading, current_lines)
            level = len(m.group(1))
            heading_text = m.group(2)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_heading = heading_text
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None:
        flush(current_heading, current_lines)

    return docs


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    all_docs: list[Document] = []
    md_files = sorted(docs_dir.glob("*.md"))
    for md_file in md_files:
        all_docs.extend(load_markdown_sections(md_file))

    chunks = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    files_indexed = len(md_files)
    sections_indexed = len(chunks)

    save_vector_index()
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    if vectorstore is None:
        return
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed
    index_faiss = index_dir / "index.faiss"
    index_pkl = index_dir / "index.pkl"
    metadata_path = index_dir / "metadata.json"

    if not index_faiss.exists() or not index_pkl.exists():
        return 0, 0

    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        return 0, 0

    vectorstore = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    files_indexed = metadata.get("files_indexed", 0)
    sections_indexed = metadata.get("sections_indexed", 0)
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
