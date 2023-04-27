import math
import numpy as np
import pandas as pd
import torch
# from transformers import BertTokenizer, BertModel
import tqdm


KEEP = ['frameTime', 'audSpec_Rfilt_sma[1]_stddev', 'audspecRasta_lengthL1norm_sma_stddev', 'audSpec_Rfilt_sma[7]_stddev', 'audSpec_Rfilt_sma[8]_stddev', 'pcm_fftMag_spectralRollOff25.0_sma_de_stddev', 'audSpec_Rfilt_sma[5]_amean', 'audSpec_Rfilt_sma[12]_stddev', 'F0final_sma_amean', 'audSpec_Rfilt_sma_de[8]_stddev', 'pcm_fftMag_spectralVariance_sma_stddev', 'audspecRasta_lengthL1norm_sma_de_stddev', 'shimmerLocal_sma_de_stddev', 'audSpec_Rfilt_sma[20]_stddev', 'logHNR_sma_de_stddev', 'pcm_fftMag_spectralCentroid_sma_amean', 'audSpec_Rfilt_sma[7]_amean', 'jitterDDP_sma_de_stddev', 'audSpec_Rfilt_sma[6]_amean', 'pcm_fftMag_fband1000-4000_sma_stddev', 'F0final_sma_stddev', 'audSpec_Rfilt_sma[21]_amean', 'audSpec_Rfilt_sma_de[0]_stddev', 'audSpec_Rfilt_sma_de[25]_stddev', 'voicingFinalUnclipped_sma_amean', 'audSpec_Rfilt_sma[12]_amean', 'pcm_fftMag_spectralRollOff50.0_sma_stddev', 'logHNR_sma_amean', 'audSpec_Rfilt_sma[5]_stddev', 'audSpec_Rfilt_sma_de[15]_stddev', 'audSpec_Rfilt_sma_de[13]_stddev', 'audSpec_Rfilt_sma[11]_amean', 'audSpec_Rfilt_sma[15]_stddev', 'audSpec_Rfilt_sma_de[5]_stddev', 'pcm_fftMag_spectralSlope_sma_de_stddev', 'pcm_zcr_sma_stddev', 'pcm_fftMag_spectralFlux_sma_de_stddev', 'audspec_lengthL1norm_sma_amean', 'jitterLocal_sma_de_stddev', 'pcm_fftMag_spectralRollOff25.0_sma_amean', 'pcm_fftMag_spectralRollOff75.0_sma_stddev', 'pcm_fftMag_spectralRollOff90.0_sma_amean', 'pcm_fftMag_spectralKurtosis_sma_stddev', 'jitterLocal_sma_de_amean', 'audSpec_Rfilt_sma[9]_stddev', 'audSpec_Rfilt_sma[25]_stddev', 'audSpec_Rfilt_sma[16]_stddev', 'audSpec_Rfilt_sma[22]_stddev', 'pcm_fftMag_spectralHarmonicity_sma_stddev', 'pcm_fftMag_spectralHarmonicity_sma_de_stddev', 'audSpec_Rfilt_sma[13]_stddev', 'pcm_fftMag_spectralCentroid_sma_de_stddev', 'pcm_fftMag_fband1000-4000_sma_amean', 'pcm_fftMag_spectralFlux_sma_amean', 'voicingFinalUnclipped_sma_de_stddev', 'F0final_sma_de_stddev', 'audSpec_Rfilt_sma_de[22]_stddev', 'audSpec_Rfilt_sma[21]_stddev', 'audSpec_Rfilt_sma_de[21]_stddev', 'pcm_fftMag_spectralKurtosis_sma_amean', 'jitterDDP_sma_amean', 'jitterLocal_sma_stddev', 'pcm_fftMag_spectralHarmonicity_sma_amean', 'pcm_zcr_sma_amean', 'audSpec_Rfilt_sma[1]_amean', 'audSpec_Rfilt_sma[14]_stddev', 'audSpec_Rfilt_sma[4]_stddev', 'pcm_fftMag_spectralCentroid_sma_stddev', 'pcm_RMSenergy_sma_amean', 'audSpec_Rfilt_sma[10]_stddev', 'audspec_lengthL1norm_sma_de_stddev', 'pcm_fftMag_spectralEntropy_sma_de_stddev', 'audSpec_Rfilt_sma_de[20]_stddev', 'shimmerLocal_sma_amean', 'pcm_fftMag_spectralSkewness_sma_amean', 'audSpec_Rfilt_sma[3]_stddev', 'audSpec_Rfilt_sma_de[23]_stddev', 'pcm_fftMag_spectralKurtosis_sma_de_stddev', 'audSpec_Rfilt_sma_de[2]_stddev', 'audSpec_Rfilt_sma[20]_amean', 'audSpec_Rfilt_sma_de[16]_stddev', 'audSpec_Rfilt_sma_de[18]_stddev', 'pcm_fftMag_fband1000-4000_sma_de_stddev', 'audSpec_Rfilt_sma[2]_stddev', 'audSpec_Rfilt_sma[0]_amean', 'audSpec_Rfilt_sma_de[9]_stddev', 'audSpec_Rfilt_sma[8]_amean', 'pcm_zcr_sma_de_stddev', 'pcm_fftMag_spectralRollOff75.0_sma_de_stddev', 'jitterDDP_sma_stddev', 'audSpec_Rfilt_sma[0]_stddev', 'audSpec_Rfilt_sma[16]_amean', 'F0final_sma_de_amean', 'jitterLocal_sma_amean', 'audSpec_Rfilt_sma_de[4]_stddev', 'pcm_fftMag_fband250-650_sma_de_stddev', 'audSpec_Rfilt_sma[10]_amean', 'pcm_fftMag_spectralVariance_sma_de_stddev', 'voicingFinalUnclipped_sma_stddev', 'audSpec_Rfilt_sma[24]_stddev', 'pcm_fftMag_spectralRollOff50.0_sma_de_stddev', 'audSpec_Rfilt_sma[2]_amean', 'audSpec_Rfilt_sma[17]_amean', 'audSpec_Rfilt_sma[3]_amean', 'jitterDDP_sma_de_amean', 'audSpec_Rfilt_sma_de[7]_stddev', 'audSpec_Rfilt_sma_de[12]_stddev', 'audSpec_Rfilt_sma_de[11]_stddev', 'audSpec_Rfilt_sma[15]_amean', 'pcm_fftMag_spectralRollOff50.0_sma_amean', 'audSpec_Rfilt_sma[17]_stddev', 'audSpec_Rfilt_sma[24]_amean', 'audSpec_Rfilt_sma[19]_stddev', 'audSpec_Rfilt_sma[13]_amean', 'pcm_RMSenergy_sma_de_stddev', 'audSpec_Rfilt_sma_de[24]_stddev', 'pcm_fftMag_spectralSkewness_sma_de_stddev', 'pcm_fftMag_spectralRollOff90.0_sma_de_stddev', 'pcm_fftMag_fband250-650_sma_amean', 'pcm_fftMag_spectralVariance_sma_amean', 'audSpec_Rfilt_sma[23]_stddev', 'pcm_fftMag_spectralEntropy_sma_stddev', 'audSpec_Rfilt_sma_de[17]_stddev', 'audSpec_Rfilt_sma[22]_amean', 'audSpec_Rfilt_sma_de[6]_stddev', 'pcm_fftMag_spectralEntropy_sma_amean', 'pcm_fftMag_spectralRollOff90.0_sma_stddev', 'audspecRasta_lengthL1norm_sma_amean', 'audSpec_Rfilt_sma_de[1]_stddev', 'audspec_lengthL1norm_sma_stddev', 'pcm_fftMag_spectralRollOff75.0_sma_amean', 'pcm_fftMag_fband250-650_sma_stddev', 'pcm_RMSenergy_sma_stddev', 'audSpec_Rfilt_sma[18]_amean', 'audSpec_Rfilt_sma_de[19]_stddev', 'audSpec_Rfilt_sma[11]_stddev', 'audSpec_Rfilt_sma[14]_amean', 'audSpec_Rfilt_sma[23]_amean', 'logHNR_sma_de_amean', 'audSpec_Rfilt_sma_de[10]_stddev', 'audSpec_Rfilt_sma_de[3]_stddev', 'audSpec_Rfilt_sma[18]_stddev', 'audSpec_Rfilt_sma[19]_amean', 'pcm_fftMag_spectralFlux_sma_stddev', 'shimmerLocal_sma_de_amean', 'pcm_fftMag_psySharpness_sma_stddev', 'pcm_fftMag_psySharpness_sma_de_stddev', 'audSpec_Rfilt_sma[4]_amean', 'audSpec_Rfilt_sma[25]_amean', 'pcm_fftMag_spectralSkewness_sma_stddev', 'shimmerLocal_sma_stddev', 'pcm_fftMag_spectralRollOff25.0_sma_stddev', 'audSpec_Rfilt_sma_de[14]_stddev', 'pcm_fftMag_psySharpness_sma_amean', 'pcm_fftMag_spectralSlope_sma_stddev', 'voicingFinalUnclipped_sma_de_amean', 'audSpec_Rfilt_sma[9]_amean', 'logHNR_sma_stddev', 'audSpec_Rfilt_sma[6]_stddev', 'pcm_fftMag_spectralSlope_sma_amean']


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
        # print(sid)
        opensmile = pd.read_csv(f"/results/sbenoit/datasets/DEAM/features/{sid}.csv", sep=";")[KEEP]
        for j, os_row in opensmile.iterrows():
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
                meta_out.append(torch.tensor([sid, ms]).long())
                emotion_out.append(torch.tensor([a_ij, v_ij]).float())
                opensmile_out.append(torch.from_numpy(os_ij).float())
    meta_out = torch.stack(meta_out)
    emotion_out = torch.stack(emotion_out)
    opensmile_out = torch.stack(opensmile_out)
    opensmile_out = (opensmile_out - opensmile_out.mean(dim=0)) / opensmile_out.std(dim=0) # normalize
    torch.save(meta_out, "/results/sbenoit/datasets/DEAM/tensors/meta.pt")
    torch.save(emotion_out, "/results/sbenoit/datasets/DEAM/tensors/emotion.pt")
    torch.save(opensmile_out, "/results/sbenoit/datasets/DEAM/tensors/opensmile.pt")
