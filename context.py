from langchain.schema import Document
from langchain.retrievers import BaseRetriever
from typing import List

from retriever import (
    hierarchical_rag_search,
    gather_context,
    node_store
)

class HybridHierarchicalRetriever(BaseRetriever):
    def __init__(self, top_k: int = 5, include_combined: bool = True):
        self.top_k = top_k
        self.include_combined = include_combined

    def _get_relevant_documents(self, query: str) -> List[Document]:
        nodes = hierarchical_rag_search(query)

        if not nodes:
            return []

        # 🔹 take top-k nodes
        top_nodes = nodes[:self.top_k]

        documents = []

        # 🔹 Option A → individual node docs
        for node_id, score in top_nodes:
            node = node_store[node_id]

            documents.append(
                Document(
                    page_content=node["text"],
                    metadata={
                        "type": "node",
                        "node_id": node_id,
                        "paper": node["paper_name"],
                        "level": node["level"],
                        "score": float(score)
                    }
                )
            )

        # 🔹 Option B → combined context
        if self.include_combined:
            combined_context = gather_context(nodes)

            documents.append(
                Document(
                    page_content=combined_context,
                    metadata={
                        "type": "combined",
                        "num_nodes": len(nodes)
                    }
                )
            )

        return documents