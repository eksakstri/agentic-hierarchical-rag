import numpy as np
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import torch

encoder = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

with open("storage/node_store.json") as f:
    node_store = json.load(f)

with open("storage/paper_roots.json") as f:
    paper_roots = json.load(f)

embeddings_dict = np.load("storage/embeddings.npy", allow_pickle=True).item()

for node_id in node_store:
    node_store[node_id]["embedding"] = embeddings_dict[node_id]

def cosine(a, b):
    return float(np.dot(a, b))

def embed_query(query):
    return encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)

def dynamic_threshold(level):
    base = 0.20
    return base + (level * 0.05)

def traverse_node(node_id, query_emb):
    node = node_store[node_id]

    node_emb = node["embedding"]
    level = node["level"]
    parent_score = cosine(query_emb, node_emb)
    threshold = dynamic_threshold(level)

    if parent_score < threshold:
        return []

    if len(node["children"]) == 0:
        return [(node_id, parent_score)]

    child_results = []
    child_scores = []

    for child_id in node["children"]:
        res = traverse_node(child_id, query_emb)
        if res:
            child_results.extend(res)
            max_child_score = max([score for _, score in res])
            child_scores.append(max_child_score)
        else:
            child_scores.append(0)

    if len(child_scores) == 0 or max(child_scores) < parent_score:
        return [(node_id, parent_score)]

    return child_results

def hierarchical_rag_search(query):
    query_emb = embed_query(query)
    results = []

    for paper_name, root_id in paper_roots.items():
        nodes = traverse_node(root_id, query_emb)
        results.extend(nodes)

    results.sort(key=lambda x: x[1], reverse=True)

    return results

def gather_context(nodes, max_chars=8000):
    weighted_chunks = []

    for rank, (node_id, score) in enumerate(nodes, start=1):
        node = node_store[node_id]
        text = node["text"]
        level = node["level"]

        # deeper nodes = more specific → boost them
        depth_weight = 1 + (level * 0.2)

        weight = (score ** 1.5) * depth_weight / (rank ** 0.75)

        weighted_chunks.append((weight, text))

    weighted_chunks.sort(key=lambda x: x[0], reverse=True)

    combined = ""
    for weight, chunk in weighted_chunks:
        if len(combined) + len(chunk) <= max_chars:
            combined += "\n" + chunk

    return combined.strip()