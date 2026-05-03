from typing import List, Dict
from langchain.schema import Document
import re
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,   
    model="gpt-4o-mini"  
)

prompt_template = """
You are a highly accurate AI assistant.

You are given two types of context:

1. Detailed Context (fine-grained chunks with node IDs)
2. Combined Context (high-level optimized summary)

Instructions:
- Use Combined Context for overall understanding
- Use Detailed Context for precise facts
- Cite sources using [Node <id>]
- If information is missing, say "I don't know"
- Do NOT hallucinate

Detailed Context:
{detailed_text}

Combined Context:
{combined_context}

Question:
{query}

Answer in EXACT format:

Answer:
<your answer>

Sources:
<list of node_ids used>

Confidence:
<low/medium/high>
"""

def split_documents(docs: List[Document]):
    detailed_docs = []
    combined_context = ""

    for d in docs:
        if d.metadata.get("type") == "combined":
            combined_context = d.page_content
        else:
            detailed_docs.append(d)

    return detailed_docs, combined_context

def build_context(docs: List[Document]):
    detailed_docs, combined_context = split_documents(docs)

    detailed_text = "\n\n".join([
        f"[Node {d.metadata.get('node_id')} | Score {round(d.metadata.get('score', 0), 3)} | Level {d.metadata.get('level')}]\n{d.page_content}"
        for d in detailed_docs
    ])

    return detailed_text, combined_context, detailed_docs

def compute_confidence(detailed_docs: List[Document]) -> str:
    if not detailed_docs:
        return "low"

    scores = [d.metadata.get("score", 0) for d in detailed_docs]
    max_score = max(scores)

    if max_score > 0.75:
        return "high"
    elif max_score > 0.55:
        return "medium"
    else:
        return "low"

def parse_output(text: str) -> Dict:
    result = {
        "answer": "",
        "sources": [],
        "confidence": ""
    }

    # Extract sections
    answer_match = re.search(r"Answer:\s*(.*?)\s*Sources:", text, re.DOTALL)
    sources_match = re.search(r"Sources:\s*(.*?)\s*Confidence:", text, re.DOTALL)
    confidence_match = re.search(r"Confidence:\s*(.*)", text)

    if answer_match:
        result["answer"] = answer_match.group(1).strip()

    if sources_match:
        sources_text = sources_match.group(1)
        result["sources"] = re.findall(r"[a-f0-9\-]{36}", sources_text)

    if confidence_match:
        result["confidence"] = confidence_match.group(1).strip().lower()

    return result

def generate_answer(query: str, retriever) -> Dict:
    docs = retriever.get_relevant_documents(query)

    # 🔹 No docs fallback
    if not docs:
        return {
            "answer": "I don't know",
            "sources": [],
            "confidence": "low"
        }

    docs = [d for d in docs if d.metadata.get("score", 1) > 0.30 or d.metadata.get("type") == "combined"]
    detailed_text, combined_context, detailed_docs = build_context(docs)
    retrieval_confidence = compute_confidence(detailed_docs)

    prompt = prompt_template.format(
        detailed_text=detailed_text,
        combined_context=combined_context,
        query=query
    )

    response = llm.invoke(prompt)
    raw_output = response.content

    parsed = parse_output(raw_output)

    if not parsed["answer"]:
        parsed["answer"] = raw_output.strip()

    parsed["confidence"] = retrieval_confidence

    if not parsed["sources"]:
        parsed["sources"] = [d.metadata.get("node_id") for d in detailed_docs]

    return {
        "answer": parsed["answer"],
        "sources": parsed["sources"],
        "confidence": parsed["confidence"],
        "raw": raw_output
    }