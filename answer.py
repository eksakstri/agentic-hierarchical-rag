from context import HybridHierarchicalRetriever
from generator import generate_answer

retriever = HybridHierarchicalRetriever(top_k=5)
result = generate_answer("Explain probability", retriever)

print("\nAnswer:\n", result["answer"])
print("\nSources:\n", result["sources"])
print("\nConfidence:\n", result["confidence"])
