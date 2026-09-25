# memory_manager.py
import os
import json
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

MEMORY_FILE = "image_memory.json"

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f, indent=4)

model = SentenceTransformer('all-MiniLM-L6-v2')  # small & fast


def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)


def add_image_to_memory(path, description):
    memory = load_memory()

    embedding = model.encode(description).tolist()

    entry = {
        "id": len(memory) + 1,
        "path": path,
        "description": description,
        "embedding": embedding
    }

    memory.append(entry)
    save_memory(memory)

    rebuild_clusters()
    return entry


def rebuild_clusters(n_clusters=5):
    memory = load_memory()

    if len(memory) < n_clusters:
        return  # not enough data yet

    vectors = np.array([x["embedding"] for x in memory])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)

    for mem, cluster_id in zip(memory, labels):
        mem["cluster"] = int(cluster_id)

    save_memory(memory)


def search_memory(query_text, top_k=3):
    memory = load_memory()
    if not memory:
        return []

    query_vec = model.encode(query_text)

    scores = []
    for m in memory:
        v = np.array(m["embedding"])
        sim = np.dot(query_vec, v) / (np.linalg.norm(query_vec)*np.linalg.norm(v))
        scores.append((sim, m))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scores[:top_k]]
