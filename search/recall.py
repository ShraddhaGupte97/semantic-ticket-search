from sklearn.preprocessing import normalize

def compute_recall_at_k(df_train, model, index, ticket_ids, k=10, text_fn=None):
    """
    Evaluate candidate generation using Recall@k.
    
    Parameters:
    - df_train: filtered DataFrame with ground truth ticket links in comments
    - model: embedding model (SentenceTransformer)
    - index: FAISS index
    - ticket_ids: list of ticket_numbers in same order as index
    - k: top-k retrieved to check recall
    - text_fn: function to build embedding input string from row
    
    Returns:
    - recall@k score (float)
    """
    recall_scores = []

    for i, row in df_train.iterrows():
        comment = row.get('comments', [{}])[0].get('comment', '')
        if "ticket/" not in comment:
            continue

        query_id = row['ticket_number'].strip().lower()
        ground_truth = [
            t.strip().split("/")[-1].lower()
            for t in comment.split(" - ")[-1].split(",")
            if "ticket/" in t
        ]

        if text_fn:
            query_text = text_fn(row)
        else:
            query_text = row['embedding_input']

        query_embedding = model.encode([query_text])
        query_embedding = normalize(query_embedding, axis=1)

        distances, indices = index.search(query_embedding, k + 1)

        retrieved = []
        for idx in indices[0]:
            candidate = ticket_ids[idx].strip().lower()
            if candidate != query_id:
                retrieved.append(candidate)
            if len(retrieved) == k:
                break

        relevant_retrieved = sum(1 for gt in ground_truth if gt in retrieved)
        total_relevant = len(ground_truth)
        recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        recall_scores.append(recall_at_k)

    return round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0