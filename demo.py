import torch
from tokenizers import Tokenizer

from transformer import GPTLanguageModel

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using device: {device}')
tokenizer = Tokenizer.from_file("tokenizer.json")
encode = lambda s: torch.tensor(tokenizer.encode(s).ids, dtype=torch.long, device=device).unsqueeze(0)
decode = lambda l: tokenizer.decode(l)

model = GPTLanguageModel().to(device)
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prompt = "The architecture of the ancient"
context = encode(prompt)

print(f"\n--- PROMPT: {prompt} ---\n")
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=150)
    print(decode(generated[0].tolist()))