import torch
import torch.nn as nn
import re
from datasets import load_dataset
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple tokenizer
def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Doc2Vec model class (DBOW-style)
class Doc2VecDBOW(nn.Module):
    def __init__(self, num_docs, vocab_size, emb_dim):
        super().__init__()
        self.doc_embeddings = nn.Embedding(num_docs, emb_dim)
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, doc_ids):
        emb = self.doc_embeddings(doc_ids)
        logits = self.out(emb)
        return logits

# Load a saved model and its config
def load_model(path):
    checkpoint = torch.load(path, map_location=device)
    model = Doc2VecDBOW(
        num_docs=checkpoint["num_docs"],
        vocab_size=checkpoint["vocab_size"],
        emb_dim=checkpoint["embedding_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["word2idx"]

# Index a new input phrase
def index_phrase(phrase, word2idx):
    tokens = simple_tokenize(phrase)
    return [word2idx.get(token, 0) for token in tokens]

# Embed a phrase by averaging word embeddings
def embed_phrase(indexed, model):
    with torch.no_grad():
        doc_tensor = torch.tensor(indexed, dtype=torch.long).to(device)
        word_embs = model.out.weight[doc_tensor]  # Use output layer's weights
        return word_embs.mean(dim=0).cpu().numpy()

# Get top K similar documents
def get_top_similar(phrase_embedding, doc_embeddings, docs, top_k=5):
    similarities = cosine_similarity([phrase_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(similarities[i], docs[i]) for i in top_indices]

# Load both models
model_general, word2idx_general = load_model("doc2vec_dbow_complete.pth")
model_horror, word2idx_horror = load_model("doc2vec_dbow_horror_complete.pth")

# Load same dataset for comparison (ensure order)
dataset = load_dataset("imdb", split="train[:25000]")
docs = [simple_tokenize(doc["text"]) for doc in dataset]

# Index and store original docs
indexed_docs_general = [[word2idx_general.get(w, 0) for w in d] for d in docs]
indexed_docs_horror = [[word2idx_horror.get(w, 0) for w in d] for d in docs]

# Get document embeddings from both models
with torch.no_grad():
    doc_embs_general = model_general.doc_embeddings.weight.cpu().numpy()
    doc_embs_horror = model_horror.doc_embeddings.weight.cpu().numpy()

# Input phrases
phrases = {
    "neutral": "The movie had a great performance and decent plot.",
    "horror": "I was terrified when the ghost appeared in the mirror.",
    "different": "The startup pitch involved blockchain and AI to optimize logistics.",
    "romantic": "Their love story was filled with tender moments and unforgettable memories that blossomed under the summer sun."
}

# Compare and print results
for label, phrase in phrases.items():
    print(f"\n\n=== Phrase type: {label.upper()} ===")
    for model_name, model, word2idx, doc_embs in [
        ("General", model_general, word2idx_general, doc_embs_general),
        ("Horror", model_horror, word2idx_horror, doc_embs_horror),
    ]:
        indexed = index_phrase(phrase, word2idx)
        embedding = embed_phrase(indexed, model)
        top_similar = get_top_similar(embedding, doc_embs, docs)

        print(f"\n[Model: {model_name}]")
        for sim, doc_tokens in top_similar:
            print(f"Similarity: {sim:.4f}")
            print(" ".join(doc_tokens) + "...\n")

