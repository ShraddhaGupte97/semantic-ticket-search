def rerank_candidates(reranker, query_id, query_text, candidate_info, top_k=3):
    pairs = [(query_text, c["embedding_input"]) for c in candidate_info]
    scores = reranker.predict(pairs)

    results = []
    for i, c in enumerate(candidate_info):
        results.append({
            "query_id": query_id,
            "ticket_number": c["ticket_number"],
            "score": float(scores[i]),
            "text": c["embedding_input"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def compute_precision_at_k_reranked(test_groups, reranker, top_k=3):
    precision_scores = []
    for query_id, candidates in test_groups.items():
        query_text = candidates[0]["query"]
        candidate_info = [{"ticket_number": c["candidate_id"], "embedding_input": c["candidate"]} for c in candidates]
        labels = {c["candidate_id"]: c["label"] for c in candidates}

        reranked = rerank_candidates(reranker, query_id, query_text, candidate_info, top_k)
        relevant_retrieved = sum(1 for item in reranked if labels.get(item["ticket_number"], 0) == 1)
        precision_at_k = relevant_retrieved / top_k
        precision_scores.append(precision_at_k)

    return round(sum(precision_scores) / len(precision_scores), 3)