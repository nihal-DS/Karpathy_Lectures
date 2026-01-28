import torch
import torch.nn as nn
from torch.nn import functional as F


DATA_DIR = "/Users/nihaljayanth/Development/makemore/data/input.txt"
torch.manual_seed(1337)
batch_size = 32
block_size = 8
max_iter = 6000
eval_interval = 300
learning_rate = 1e-2
device = "cuda:1" if torch.cuda.is_available() else "cpu"
eval_iter = 200
n_embed = 32

def read_data(data_dir):
    with open(data_dir, "r") as f:
        text = f.read()
    return text

text = read_data(DATA_DIR)
chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = {c:idx for idx, c in enumerate(chars)}
itos = {idx:c for idx, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def split_data(data):
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    return train_data, val_data

train_data, val_data = split_data(data)

def get_batch(split):
    data = train_data if split == "train" else val_data
    sample = torch.randint(high=len(data)-block_size, size=(batch_size,))
    x = torch.stack([data[s:s+block_size] for s in sample])
    y = torch.stack([data[s+1:s+block_size+1] for s in sample])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        # B = batch_size, T = seq_len, C = output_classes
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_embed
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

lm = BigramLanguageModel()
lm = lm.to(device)
optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-3)

for step in range(max_iter):

    if step % eval_interval == 0:
        losses = estimate_loss(lm)
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    
    logits, loss = lm(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(lm.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))