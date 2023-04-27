import math
import numpy as np
import pandas as pd
import torch
# from transformers import BertTokenizer, BertModel
import tqdm


if __name__ == "__main__":
    arousal = pd.read_csv("/results/sbenoit/datasets/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv")
    valence = pd.read_csv("/results/sbenoit/datasets/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv")
    meta_out = []
    emotion_out = []
    opensmile_out = []
    # bert_out = []
    for (i, a_row), (_, v_row) in tqdm.tqdm(zip(arousal.iterrows(), valence.iterrows()), total=arousal.shape[0]):
        # print(a_row, v_row)
        sid = int(a_row["song_id"])
        print(sid)
        opensmile = pd.read_csv(f"/results/sbenoit/datasets/DEAM/features/{sid}.csv", sep=";")
        for j, os_row in tqdm.tqdm(opensmile.iterrows(), total=opensmile.shape[0]):
            s = os_row["frameTime"]
            ms = round(s * 1000)
            a_ij = float("nan")
            v_ij = float("nan")
            key = f"sample_{ms}ms"
            # print(key)
            if key in a_row.keys():
                a_ij = a_row[key]
            if key in v_row.keys():
                v_ij = v_row[key]
            # print(a_ij, v_ij)
            if not (math.isnan(a_ij) or math.isnan(v_ij)):
                os_ij = os_row.to_numpy()[1:]
                meta_out.append(torch.tensor([sid, ms], dtype=torch.long))
                emotion_out.append(torch.tensor([a_ij, v_ij]))
                opensmile_out.append(torch.from_numpy(os_ij))
    meta_out = torch.stack(meta_out)
    emotion_out = torch.stack(emotion_out)
    opensmile_out = torch.stack(opensmile_out)
    torch.save(meta_out, "/results/sbenoit/datasets/DEAM/tensors/meta.pt")
    torch.save(emotion_out, "/results/sbenoit/datasets/DEAM/tensors/emotion.pt")
    torch.save(opensmile_out, "/results/sbenoit/datasets/DEAM/tensors/opensmile.pt")
