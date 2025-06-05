from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np

def encode_texts(texts, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar = True)
    return normalize(embeddings, axis=1), model