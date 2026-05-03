from pypdf import PdfReader
import re
import os
import uuid
from typing import List
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import json
import nltk
nltk.download('punkt')
import torch

encoder = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

pdfs_path = r"pdfs"

papers = {}
paper_atomic_units = {}
paper_atomic_embeddings = {}
node_store = {}
paper_roots = {}

def new_id():
    return str(uuid.uuid4())

def to_sentences(text: str) -> List[str]:
    sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if len(s.strip())>10]

def split_into_atomic_units(text: str):
    parts = re.split(r'(?<=[\.\!\?\;\:])\s+', text)

    cleaned = []
    for p in parts:
        p = p.strip()
        if len(p) < 30 and cleaned:
            cleaned[-1] += " " + p
        else:
            cleaned.append(p)

    cleaned = [x for x in cleaned if len(x.strip()) > 0]
    return cleaned

def cluster_semantically(embs, threshold=0.20):
    sim_matrix = np.matmul(embs, embs.T)
    dist = 1 - sim_matrix

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        linkage='average',
        distance_threshold=threshold
    )

    labels = clusterer.fit_predict(dist)
    return labels

def embed_texts(texts, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        arr = encoder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embs.append(arr)
    return np.vstack(embs)

def build_tree_recursive(
        texts, embeddings, paper_name,
        parent_id=None, level=0, max_depth=3, threshold=0.20
    ):

    node_id = new_id()
    combined_text = " ".join(texts)
    node_embedding = encoder.encode(
        [combined_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    node_store[node_id] = {
        "node_id": node_id,
        "parent_id": parent_id,
        "paper_name": paper_name,
        "level": level,
        "text": combined_text,
        "embedding": node_embedding,
        "children": []
    }

    if level >= max_depth or len(texts) <= 2:
        return node_id

    labels = cluster_semantically(embeddings, threshold=threshold)

    cluster_map = {}
    for t, e, l in zip(texts, embeddings, labels):
        cluster_map.setdefault(l, {"texts": [], "embs": []})
        cluster_map[l]["texts"].append(t)
        cluster_map[l]["embs"].append(e)

    for cluster in cluster_map.values():
        child_id = build_tree_recursive(
            cluster["texts"],
            np.vstack(cluster["embs"]),
            paper_name,
            parent_id=node_id,
            level=level + 1,
            max_depth=max_depth,
            threshold=threshold
        )
        node_store[node_id]["children"].append(child_id)

    return node_id

for file in os.listdir(pdfs_path):
  if file.endswith(".pdf"):
    pdf_path = os.path.join(pdfs_path, file)
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\n|\r|\f)\s*\d{1,3}\s*(\n|\r|\f)', '\n', text)
    text = text.replace("\x0c", " ")
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s*-\s*', '-', text)
    text = text.replace("•", "-")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    papers[file] = text

for name, text in papers.items():
    units = split_into_atomic_units(text)
    paper_atomic_units[name] = units

for name, units in paper_atomic_units.items():
    embs = embed_texts(units)
    paper_atomic_embeddings[name] = {
        "units": units,
        "embs": embs
    }
    
for paper_name, data in paper_atomic_embeddings.items():
    units = data["units"]
    embs = data["embs"]
    root = build_tree_recursive(units, embs, paper_name)
    paper_roots[paper_name] = root

os.makedirs("storage", exist_ok=True)

embeddings_dict = {}
node_store_serializable = {}

for node_id, node in node_store.items():
    embeddings_dict[node_id] = node["embedding"]
    
    node_copy = node.copy()
    del node_copy["embedding"]  
    
    node_store_serializable[node_id] = node_copy

np.save("storage/embeddings.npy", embeddings_dict)

with open("storage/node_store.json", "w") as f:
    json.dump(node_store_serializable, f)

with open("storage/paper_roots.json", "w") as f:
    json.dump(paper_roots, f)