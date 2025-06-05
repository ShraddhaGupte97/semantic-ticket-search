import json
import pandas as pd
import numpy as np
import faiss

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def load_dataframe(json_file):
    return pd.DataFrame(load_json(json_file))

def save_embeddings(embedding_matrix, path="artifacts/embedding_matrix.npy"):
    np.save(path, embedding_matrix)

def load_embeddings(path="artifacts/embedding_matrix.npy"):
    return np.load(path)

def save_ticket_ids(ticket_ids, path="artifacts/ticket_ids.json"):
    with open(path, "w") as f:
        json.dump(ticket_ids, f)

def load_ticket_ids(path="artifacts/ticket_ids.json"):
    with open(path, "r") as f:
        return json.load(f)

def save_faiss_index(index, path="artifacts/faiss.index"):
    faiss.write_index(index, path)

def load_faiss_index(path="artifacts/faiss.index"):
    return faiss.read_index(path)

