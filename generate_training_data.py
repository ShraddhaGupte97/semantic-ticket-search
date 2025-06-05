import json
from collections import defaultdict
from prepare_data import prepare_ticket_data
from sklearn.model_selection import train_test_split
from utils.io import (
    load_dataframe, save_json,
    save_embeddings, load_embeddings,
    save_ticket_ids, load_ticket_ids,
    save_faiss_index, load_faiss_index
)

# Load and prepare data
df_train = prepare_ticket_data("data/raw_data.json")

# Load or compute embeddings and index
embedding_matrix = load_embeddings()
ticket_ids = load_ticket_ids()
index = load_faiss_index()

# Generate training pairs
rerank_training_data = []
for i, row in df_train.iterrows():
    query_id = row["ticket_number"]
    query_text = row["embedding_input"]
    distances, indices = index.search(embedding_matrix[i].reshape(1, -1), 10)
    
    comment = row.get("comments", [{}])[0].get("comment", "")
    ground_truth = [
        t.strip().split("/")[-1].lower()
        for t in comment.split(" - ")[-1].split(",")
        if "ticket/" in t
    ]
    relevant_ids_set = set(ground_truth)

    for idx in indices[0]:
        candidate_id = ticket_ids[idx]
        if candidate_id == query_id:
            continue

        candidate_row = df_train[df_train["ticket_number"] == candidate_id].iloc[0]
        candidate_text = candidate_row["embedding_input"]

        # Check if candidate_id is in the training data
        if candidate_id.lower() in relevant_ids_set:
            label = 1
        else:
            folder_match = candidate_row["assigned_folder"] == row["assigned_folder"]
            metric_match = candidate_row["metrics"] == row["metrics"]
            shared_dims = any(
                any(dim.lower() in cand_dim.lower() or cand_dim.lower() in dim.lower()
                    for cand_dim in candidate_row["dimensions"])
                for dim in row["dimensions"]
            )
            if folder_match and metric_match and shared_dims:
                label = 1
            else:
                label = 0

        rerank_training_data.append({
            "query_id": query_id,
            "candidate_id": candidate_id,
            "query": query_text,
            "candidate": candidate_text,
            "label": label
        })

# Stratified split by query_id
grouped = defaultdict(list)
for pair in rerank_training_data:
    grouped[pair["query_id"]].append(pair)

query_ids = list(grouped.keys())
query_ids_train, query_ids_test = train_test_split(query_ids, test_size=0.2, random_state=42)

train_pairs = [pair for qid in query_ids_train for pair in grouped[qid]]
test_pairs = [pair for qid in query_ids_test for pair in grouped[qid]]

# Save to disk
save_json(train_pairs, "data/rerank_train.json")
save_json(test_pairs, "data/rerank_test.json")
print(f"Saved training pairs: {len(train_pairs)} | test pairs: {len(test_pairs)}")
