import tiktoken
from torch_dataloader import FineWebDataset, make_loader

enc = tiktoken.get_encoding("gpt2")
SEQ_LEN = 20

ds = FineWebDataset("train", seq_len=SEQ_LEN)
x, y = ds[0]

# print(f"{len(ds):,} samples, sample shape {tuple(x.shape)}\n")
# print("x:", enc.decode(x.tolist()))
# print("y:", enc.decode(y.tolist()))

loader = make_loader("train", batch_size=4, seq_len=SEQ_LEN, num_workers=0)
xb, yb = next(iter(loader))

print(f"\nbatch shape {tuple(xb.shape)}, {len(loader):,} batches per epoch\n")
for i in range(len(xb)):
    print(f"row {i} x: {enc.decode(xb[i].tolist())}")
    print(f"row {i} y: {enc.decode(yb[i].tolist())}")
    print()
