import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


# Explicit system prompt prevents the LLM from using its own knowledge
# and forces it to cite sources — critical for a grounded Q&A bot.
SYSTEM_PROMPT = """You are a knowledge base assistant. Answer questions using ONLY the CONTEXT provided below.

Rules:
- Cite every fact with its source in the format [filename#heading].
- If the context does not contain enough information to answer, reply exactly: "I cannot confirm from the knowledge base."
- Do not guess, infer, or use any knowledge outside the provided CONTEXT.
- Keep answers concise and grounded in the source text.
"""

_llm = None


def get_llm():
    # Lazy init — only creates the client when the first /chat call arrives
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_chunks: list) -> str:
    """Format the top retrieved chunks into a prompt for the LLM.

    Each chunk gets a [Source: ...] label so the LLM knows which document
    the text came from and can include it in citations.
    CONTEXT comes before QUESTION so the LLM reads evidence first.
    """
    context_parts = []
    for doc, _score in ranked_chunks:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    # Guard: vectorstore is None until POST /index is called
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    # Embed the question and find the k most similar chunks in the vector index
    ranked_chunks = indexer.search(question, k=3)

    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    # Include retrieved chunks in the response so callers can verify grounding
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],  # truncated preview
        }
        for doc, score in ranked_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
