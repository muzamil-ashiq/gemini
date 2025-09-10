import torch
from model import GPTConfig, GPT

print("--- Running Smoke Test 1: Forward pass (eval mode) ---")
model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=50304))
model.eval()
x = torch.randint(0, model.config.vocab_size, (1, 10), dtype=torch.long)
logits, loss = model(x, targets=None)
print("logits shape:", logits.shape)
print("loss:", loss)
print("Smoke Test 1 PASSED")

print("\n--- Running Smoke Test 2: Training step ---")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randint(0, model.config.vocab_size, (2, 16), dtype=torch.long)
y = torch.randint(0, model.config.vocab_size, (2, 16), dtype=torch.long)
logits, loss = model(x, y)
loss.backward()
optimizer.step()
print("loss:", loss.item())
print("Smoke Test 2 PASSED")

print("\n--- Running Smoke Test 3: Check buffers vs. parameters ---")
for name, p in model.named_parameters():
    print("param:", name, p.requires_grad)
for name, buf in model.named_buffers():
    print("buffer:", name, buf.shape)
print("Smoke Test 3 PASSED")
