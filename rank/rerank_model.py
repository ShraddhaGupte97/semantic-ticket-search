from sentence_transformers import CrossEncoder

def get_reranker(model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name, num_labels = 1)