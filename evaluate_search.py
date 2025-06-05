from prepare_data import prepare_ticket_data
from utils.io import save_embeddings, save_ticket_ids, save_faiss_index
from search.candidate_generator import encode_texts
from search.faiss_index import build_index
from search.recall import compute_recall_at_k

# Load and filter data
df_train = prepare_ticket_data("data/raw_data.json")

# Generate embeddings
embedding_matrix, model = encode_texts(df_train["embedding_input"].tolist())
ticket_ids = df_train["ticket_number"].tolist()

# Build FAISS index
index = build_index(embedding_matrix)

# Save embeddings, ticket ids, and FAISS index
save_embeddings(embedding_matrix)
save_ticket_ids(ticket_ids)
save_faiss_index(index)

# Evaluate candidate generation
recall = compute_recall_at_k(df_train, model, index, ticket_ids, k = 10)
print(f"Recall@10 for FAISS Candidate Generation: {recall:.3f}")