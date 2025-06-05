from rank.rerank_model import get_reranker
from rank.precision import compute_precision_at_k_reranked
from utils.io import load_json
from collections import defaultdict

# Load trained reranker model
reranker = get_reranker("./rank_model")

# Load test data
test_pairs = load_json("data/rerank_test.json")

# Group test pairs by query_id
grouped = defaultdict(list)
for pair in test_pairs:
    grouped[pair["query_id"]].append(pair)

# Evaluate reranker precision
precision = compute_precision_at_k_reranked(grouped, reranker, top_k = 3)
print(f"Reranker Precision@3: {precision}")