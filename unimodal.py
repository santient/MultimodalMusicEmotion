import argparse
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import tqdm


class UnimodalModel(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers):
        super().__init__()
        layers = [nn.Linear(d_input, d_hidden), nn.Dropout(p=0.2), nn.GELU()]
        for i in range(n_layers):
            layers.extend([nn.Linear(d_hidden, d_hidden), nn.Dropout(p=0.2), nn.GELU()])
        layers.extend([nn.Linear(d_hidden, d_output), nn.Tanh()])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def load_data(data_dir, val_split):
    meta = torch.load(os.path.join(data_dir, "meta.pt"))
    opensmile = torch.load(os.path.join(data_dir, "opensmile.pt"))
    # opensmile = (opensmile - opensmile.mean(dim=0)) / opensmile.std(dim=0) # normalize
    emotion = torch.load(os.path.join(data_dir, "emotion.pt"))
    # n = len(meta)
    # n_val = round(n * val_split)
    # n_train = n - n_val
    dset = TensorDataset(meta, opensmile, emotion)
    # train_set = TensorDataset(meta[:-n_val], opensmile[:-n_val], emotion[:-n_val])
    # val_set = TensorDataset(meta[-n_val:], opensmile[-n_val:], emotion[-n_val:])
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dset, [1 - val_split, val_split], generator)
    return train_set, val_set


def train(model, train_set, val_set, criterion, optimizer, args):
    dloader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
    results = []
    for epoch in tqdm.tqdm(range(args.epochs)):
        for step, (meta, opensmile, emotion) in enumerate(dloader):
            opensmile = opensmile.cuda()
            emotion = emotion.cuda()
            out = model(opensmile)
            loss = criterion(out, emotion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[Epoch {epoch}] [Step {step}] [Loss {loss.item()}]", flush=True)
        val_loss = []
        for step, (meta, opensmile, emotion) in enumerate(val_set):
            with torch.no_grad():
                opensmile = opensmile.unsqueeze(0).cuda()
                emotion = emotion.unsqueeze(0).cuda()
                out = model(opensmile)
                loss = criterion(out, emotion).item()
                val_loss.append(loss)
                # print(f"[Epoch {epoch}] [Val Step {step}] [Loss {loss}]", flush=True)
        val_loss = torch.tensor(val_loss).mean().item()
        results.append({"epoch": epoch, "val_loss": val_loss})
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))
        print(f"Epoch {epoch} Validaton Loss: {val_loss}", flush=True)
    results = pd.DataFrame.from_records(results)
    results.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    torch.backends.cuda.benchmark = True
    d_input = 159
    d_output = 2
    model = UnimodalModel(d_input, args.d_hidden, d_output, args.n_layers).cuda()
    train_set, val_set = load_data(args.data_dir, args.val_split)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, train_set, val_set, criterion, optimizer, args)


if __name__ == "__main__":
    main()
