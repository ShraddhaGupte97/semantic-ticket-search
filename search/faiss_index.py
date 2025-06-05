import faiss

def build_index(embedding_matrix):
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)
    return index