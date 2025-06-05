import os
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

def train_reranker(train_data, reranker, batch_size=16, epochs=10):
    train_examples = [
        InputExample(texts=[item["query"], item["candidate"]], label=float(item["label"]))
        for item in train_data
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "rank_model")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving model to: {os.path.abspath(output_path)}")
    reranker.fit(train_dataloader=train_dataloader, epochs=epochs, warmup_steps=100, output_path=output_path)
    reranker.save(output_path)