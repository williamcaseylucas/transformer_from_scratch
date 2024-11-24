import torch
from models import Transformer

device = torch.device("mps")

x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
    device
)  # (2, 9)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 3, 4, 5, 6, 2]]).to(
    device
)  # (2, 8)

x.shape, trg.shape
src_pad_idx = 0
trg_pad_idx = 0
model = Transformer(
    src_vocab_size=10,  # 9 is max we can have
    trg_vocab_size=10,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    device=device,
).to(device)

out = model(x, trg[:, :-1])
print(out.shape)
