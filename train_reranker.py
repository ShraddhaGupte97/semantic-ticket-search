from utils.io import load_json
from rank.rerank_model import get_reranker
from rank.rerank_trainer import train_reranker

# Load training data
train_data = load_json("data/rerank_train.json")

# Load model
reranker = get_reranker()

# Train model
train_reranker(train_data, reranker, batch_size=16, epochs=10)
print("Reranker training complete. Model saved to ./reranker_model")
