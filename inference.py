import json
import numpy as np
from sklearn.preprocessing import normalize
from utils.io import (
    load_faiss_index,
    load_embeddings,
    load_ticket_ids,
    load_dataframe
)
from utils.text_builder import build_ticket_text
from search.candidate_generator import encode_texts
from rank.rerank_model import get_reranker
from rank.precision import rerank_candidates

def find_similar_tickets(query_ticket, top_k=3): 
    # Load cached artifacts
    embedding_matrix = load_embeddings("artifacts/embedding_matrix.npy")
    ticket_ids = load_ticket_ids("artifacts/ticket_ids.json")
    index = load_faiss_index("artifacts/faiss.index")

    # Load trained reranker
    reranker = get_reranker("./rank_model")

    # Load ticket metadata (for showing results only. please use with caution in production)
    ticket_df = load_dataframe("data/raw_data.json")
    ticket_map = {row["ticket_number"]: row for _, row in ticket_df.iterrows()}

    # Create embedding_input
    query_id = query_ticket.get("ticket_number", "T9999999")
    query_text = build_ticket_text(query_ticket)

    # Embed the query
    query_embedding, _ = encode_texts([query_text])

    # Search FAISS index
    distances, indices = index.search(query_embedding, 10)

    # Collect candidate info
    candidate_info = []
    for idx in indices[0]:
        cand_id = ticket_ids[idx]
        if cand_id == query_id:
            continue
        row = ticket_map.get(cand_id)
        candidate_info.append({
            "ticket_number": cand_id,
            "embedding_input": build_ticket_text(row)
        })

    # Rerank top-k
    top_k = 3
    reranked = rerank_candidates(reranker, query_id, query_text, candidate_info, top_k = top_k)

    # Output
    print(f"\nQuery Ticket [{query_id}]:\n{query_text}\n")
    print("Top Reranked Tickets:\n")
    for i, r in enumerate(reranked, 1):
        print(f"{i}. Ticket: {r['ticket_number']} | Score: {r['score']:.4f}")
        print(f"   Text: {r['text'][:120]}...\n")

    return reranked

# --- test code for standalone use ---
# if __name__ == "__main__":
#     # Example ticket for testing
#     input_json = input("Paste your query ticket JSON:\n")
#     # Take user input as JSON
#     query_ticket = json.loads(input_json)
#     results = find_similar_tickets(query_ticket, top_k=3)
