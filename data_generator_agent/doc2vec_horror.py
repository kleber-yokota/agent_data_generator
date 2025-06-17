import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import re
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

terror_keywords = ["horror", "scary", "ghost", "kill", "blood", "death", "monster", "fear", "dark", "evil"]
def is_horror(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in terror_keywords)

raw_dataset = load_dataset("imdb", split="train[:25000]")
docs = [simple_tokenize(doc["text"]) for doc in raw_dataset if is_horror(doc["text"])]

print(f"Documentos filtrados (terror): {len(docs)}")

word_freq = defaultdict(int)
for doc in docs:
    for word in doc:
        word_freq[word] += 1

min_freq = 10
word2idx = {"<UNK>": 0}
for word, freq in word_freq.items():
    if freq >= min_freq:
        word2idx[word] = len(word2idx)

idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)
num_docs = len(docs)
print(f"Tamanho do vocabulário: {vocab_size}")

indexed_docs = []
for doc in docs:
    indexed = [word2idx.get(word, 0) for word in doc]
    indexed_docs.append(indexed if indexed else [0])

class Doc2VecDBOW(nn.Module):
    def __init__(self, num_docs, vocab_size, emb_dim):
        super().__init__()
        self.doc_embeddings = nn.Embedding(num_docs, emb_dim)
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, doc_ids):
        emb = self.doc_embeddings(doc_ids)  # (batch, emb_dim)
        logits = self.out(emb)              # (batch, vocab_size)
        return logits

embedding_dim = 100
model = Doc2VecDBOW(num_docs, vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

pairs = []
for doc_id, word_ids in enumerate(indexed_docs):
    for word_id in word_ids:
        pairs.append((doc_id, word_id))

random.shuffle(pairs)

BATCH_SIZE = 1024
EPOCHS = 3

for epoch in range(EPOCHS):
    total_loss = 0
    for i in tqdm(range(0, len(pairs), BATCH_SIZE)):
        batch = pairs[i:i+BATCH_SIZE]
        doc_ids = torch.tensor([x[0] for x in batch], dtype=torch.long).to(device)
        word_targets = torch.tensor([x[1] for x in batch], dtype=torch.long).to(device)

        logits = model(doc_ids)
        loss = loss_fn(logits, word_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "word2idx": word2idx,
    "num_docs": num_docs,
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim
}

torch.save(checkpoint, "doc2vec_dbow_horror_complete.pth")
print("Modelo completo salvo em: doc2vec_dbow_horror_complete.pth")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    word2idx = checkpoint["word2idx"]
    num_docs = checkpoint["num_docs"]
    vocab_size = checkpoint["vocab_size"]
    embedding_dim = checkpoint["embedding_dim"]

    model = Doc2VecDBOW(num_docs, vocab_size, embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, word2idx

model_loaded, word2idx_loaded = load_model("doc2vec_dbow_horror_complete.pth", device)

with torch.no_grad():
    doc_embeddings = model_loaded.doc_embeddings.weight.cpu().numpy()

def print_similar_docs(query_idx, docs, embeddings, top_k=5):
    query_vec = embeddings[query_idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sims.argsort()[-top_k-1:][::-1]  # pegar top_k + 1 (porque o próprio doc)

    print("\n[Documento Consulta]")
    print(" ".join(docs[query_idx][:50]))

    print("\n[Documentos mais semelhantes]")
    for idx in top_indices:
        if idx == query_idx:
            continue
        print(f"\nSimilaridade: {sims[idx]:.4f}")
        print(" ".join(docs[idx][:50]))

random_idx = random.randint(0, num_docs - 1)
print_similar_docs(random_idx, docs, doc_embeddings)
