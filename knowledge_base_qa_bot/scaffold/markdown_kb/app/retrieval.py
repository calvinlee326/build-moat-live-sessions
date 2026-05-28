import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


# The system prompt defines the LLM's behavior for every /chat call.
# Being explicit about "only use CONTEXT" and "say you cannot confirm"
# prevents hallucination — the LLM making up answers not in the docs.
SYSTEM_PROMPT = """You are a knowledge base assistant. Answer questions using ONLY the CONTEXT provided below.

Rules:
- Cite every fact with its source in the format [filename#heading].
- If the context does not contain enough information to answer, reply exactly: "I cannot confirm from the knowledge base."
- Do not guess, infer, or use any knowledge outside the provided CONTEXT.
- Keep answers concise and grounded in the source text.
"""

# Lazy-initialize the LLM so we don't connect on import (avoids startup errors
# when the API key isn't set yet or tests don't need the real LLM).
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_sections: list) -> str:
    """Format the top BM25 sections into a prompt the LLM can use.

    Each section gets a [Source: ...] header so the LLM knows where to
    cite from. The breadcrumb (heading_path) gives the LLM structural
    context — it can tell if a section is under "Refunds > International".
    """
    context_parts = []
    for section, _score in ranked_sections:
        breadcrumb = " > ".join(section.heading_path)
        context_parts.append(f"[Source: {section.id}]\n{breadcrumb}\n\n{section.content}")
    # Separate sections with a horizontal rule so the LLM can clearly see boundaries
    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    # Guard: if no index has been built yet, tell the user instead of erroring
    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=3)

    # If BM25 finds nothing relevant (all scores = 0), admit it honestly
    if not ranked_sections:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    # SystemMessage sets persistent behavior; HumanMessage is the actual question+context
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    # Return both the answer and the source sections so callers can inspect grounding
    sources = [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],  # truncate for readability
        }
        for section, score in ranked_sections
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
