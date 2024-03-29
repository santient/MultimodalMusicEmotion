import pandas as pd
import numpy as np
# import opensmile
from pydub import AudioSegment
import torch
from MultimodalMusicEmotion.unimodal import UnimodalModel
# from unimodal import UnimodalModel


KEEP = ['audspec_lengthL1norm_sma_stddev',
    'audspecRasta_lengthL1norm_sma_stddev',
    'pcm_RMSenergy_sma_stddev',
    'pcm_zcr_sma_stddev',
    'audspec_lengthL1norm_sma_de_stddev',
    'audspecRasta_lengthL1norm_sma_de_stddev',
    'pcm_RMSenergy_sma_de_stddev',
    'pcm_zcr_sma_de_stddev',
    'audSpec_Rfilt_sma[0]_stddev',
    'audSpec_Rfilt_sma[1]_stddev',
    'audSpec_Rfilt_sma[2]_stddev',
    'audSpec_Rfilt_sma[3]_stddev',
    'audSpec_Rfilt_sma[4]_stddev',
    'audSpec_Rfilt_sma[5]_stddev',
    'audSpec_Rfilt_sma[6]_stddev',
    'audSpec_Rfilt_sma[7]_stddev',
    'audSpec_Rfilt_sma[8]_stddev',
    'audSpec_Rfilt_sma[9]_stddev',
    'audSpec_Rfilt_sma[10]_stddev',
    'audSpec_Rfilt_sma[11]_stddev',
    'audSpec_Rfilt_sma[12]_stddev',
    'audSpec_Rfilt_sma[13]_stddev',
    'audSpec_Rfilt_sma[14]_stddev',
    'audSpec_Rfilt_sma[15]_stddev',
    'audSpec_Rfilt_sma[16]_stddev',
    'audSpec_Rfilt_sma[17]_stddev',
    'audSpec_Rfilt_sma[18]_stddev',
    'audSpec_Rfilt_sma[19]_stddev',
    'audSpec_Rfilt_sma[20]_stddev',
    'audSpec_Rfilt_sma[21]_stddev',
    'audSpec_Rfilt_sma[22]_stddev',
    'audSpec_Rfilt_sma[23]_stddev',
    'audSpec_Rfilt_sma[24]_stddev',
    'audSpec_Rfilt_sma[25]_stddev',
    'pcm_fftMag_fband250-650_sma_stddev',
    'pcm_fftMag_fband1000-4000_sma_stddev',
    'pcm_fftMag_spectralRollOff25.0_sma_stddev',
    'pcm_fftMag_spectralRollOff50.0_sma_stddev',
    'pcm_fftMag_spectralRollOff75.0_sma_stddev',
    'pcm_fftMag_spectralRollOff90.0_sma_stddev',
    'pcm_fftMag_spectralFlux_sma_stddev',
    'pcm_fftMag_spectralCentroid_sma_stddev',
    'pcm_fftMag_spectralEntropy_sma_stddev',
    'pcm_fftMag_spectralVariance_sma_stddev',
    'pcm_fftMag_spectralSkewness_sma_stddev',
    'pcm_fftMag_spectralKurtosis_sma_stddev',
    'pcm_fftMag_spectralSlope_sma_stddev',
    'pcm_fftMag_psySharpness_sma_stddev',
    'pcm_fftMag_spectralHarmonicity_sma_stddev',
    'audSpec_Rfilt_sma_de[0]_stddev',
    'audSpec_Rfilt_sma_de[1]_stddev',
    'audSpec_Rfilt_sma_de[2]_stddev',
    'audSpec_Rfilt_sma_de[3]_stddev',
    'audSpec_Rfilt_sma_de[4]_stddev',
    'audSpec_Rfilt_sma_de[5]_stddev',
    'audSpec_Rfilt_sma_de[6]_stddev',
    'audSpec_Rfilt_sma_de[7]_stddev',
    'audSpec_Rfilt_sma_de[8]_stddev',
    'audSpec_Rfilt_sma_de[9]_stddev',
    'audSpec_Rfilt_sma_de[10]_stddev',
    'audSpec_Rfilt_sma_de[11]_stddev',
    'audSpec_Rfilt_sma_de[12]_stddev',
    'audSpec_Rfilt_sma_de[13]_stddev',
    'audSpec_Rfilt_sma_de[14]_stddev',
    'audSpec_Rfilt_sma_de[15]_stddev',
    'audSpec_Rfilt_sma_de[16]_stddev',
    'audSpec_Rfilt_sma_de[17]_stddev',
    'audSpec_Rfilt_sma_de[18]_stddev',
    'audSpec_Rfilt_sma_de[19]_stddev',
    'audSpec_Rfilt_sma_de[20]_stddev',
    'audSpec_Rfilt_sma_de[21]_stddev',
    'audSpec_Rfilt_sma_de[22]_stddev',
    'audSpec_Rfilt_sma_de[23]_stddev',
    'audSpec_Rfilt_sma_de[24]_stddev',
    'audSpec_Rfilt_sma_de[25]_stddev',
    'pcm_fftMag_fband250-650_sma_de_stddev',
    'pcm_fftMag_fband1000-4000_sma_de_stddev',
    'pcm_fftMag_spectralRollOff25.0_sma_de_stddev',
    'pcm_fftMag_spectralRollOff50.0_sma_de_stddev',
    'pcm_fftMag_spectralRollOff75.0_sma_de_stddev',
    'pcm_fftMag_spectralRollOff90.0_sma_de_stddev',
    'pcm_fftMag_spectralFlux_sma_de_stddev',
    'pcm_fftMag_spectralCentroid_sma_de_stddev',
    'pcm_fftMag_spectralEntropy_sma_de_stddev',
    'pcm_fftMag_spectralVariance_sma_de_stddev',
    'pcm_fftMag_spectralSkewness_sma_de_stddev',
    'pcm_fftMag_spectralKurtosis_sma_de_stddev',
    'pcm_fftMag_spectralSlope_sma_de_stddev',
    'pcm_fftMag_psySharpness_sma_de_stddev',
    'pcm_fftMag_spectralHarmonicity_sma_de_stddev',
    'F0final_sma_amean',
    'F0final_sma_stddev',
    'voicingFinalUnclipped_sma_amean',
    'voicingFinalUnclipped_sma_stddev',
    'jitterLocal_sma_amean',
    'jitterLocal_sma_stddev',
    'jitterDDP_sma_amean',
    'jitterDDP_sma_stddev',
    'shimmerLocal_sma_amean',
    'shimmerLocal_sma_stddev',
    'logHNR_sma_amean',
    'logHNR_sma_stddev',
    'F0final_sma_de_amean',
    'F0final_sma_de_stddev',
    'voicingFinalUnclipped_sma_de_amean',
    'voicingFinalUnclipped_sma_de_stddev',
    'jitterLocal_sma_de_amean',
    'jitterLocal_sma_de_stddev',
    'jitterDDP_sma_de_amean',
    'jitterDDP_sma_de_stddev',
    'shimmerLocal_sma_de_amean',
    'shimmerLocal_sma_de_stddev',
    'logHNR_sma_de_amean',
    'logHNR_sma_de_stddev',
    'audspec_lengthL1norm_sma_amean',
    'audspecRasta_lengthL1norm_sma_amean',
    'pcm_RMSenergy_sma_amean',
    'pcm_zcr_sma_amean',
    'audSpec_Rfilt_sma[0]_amean',
    'audSpec_Rfilt_sma[1]_amean',
    'audSpec_Rfilt_sma[2]_amean',
    'audSpec_Rfilt_sma[3]_amean',
    'audSpec_Rfilt_sma[4]_amean',
    'audSpec_Rfilt_sma[5]_amean',
    'audSpec_Rfilt_sma[6]_amean',
    'audSpec_Rfilt_sma[7]_amean',
    'audSpec_Rfilt_sma[8]_amean',
    'audSpec_Rfilt_sma[9]_amean',
    'audSpec_Rfilt_sma[10]_amean',
    'audSpec_Rfilt_sma[11]_amean',
    'audSpec_Rfilt_sma[12]_amean',
    'audSpec_Rfilt_sma[13]_amean',
    'audSpec_Rfilt_sma[14]_amean',
    'audSpec_Rfilt_sma[15]_amean',
    'audSpec_Rfilt_sma[16]_amean',
    'audSpec_Rfilt_sma[17]_amean',
    'audSpec_Rfilt_sma[18]_amean',
    'audSpec_Rfilt_sma[19]_amean',
    'audSpec_Rfilt_sma[20]_amean',
    'audSpec_Rfilt_sma[21]_amean',
    'audSpec_Rfilt_sma[22]_amean',
    'audSpec_Rfilt_sma[23]_amean',
    'audSpec_Rfilt_sma[24]_amean',
    'audSpec_Rfilt_sma[25]_amean',
    'pcm_fftMag_fband250-650_sma_amean',
    'pcm_fftMag_fband1000-4000_sma_amean',
    'pcm_fftMag_spectralRollOff25.0_sma_amean',
    'pcm_fftMag_spectralRollOff50.0_sma_amean',
    'pcm_fftMag_spectralRollOff75.0_sma_amean',
    'pcm_fftMag_spectralRollOff90.0_sma_amean',
    'pcm_fftMag_spectralFlux_sma_amean',
    'pcm_fftMag_spectralCentroid_sma_amean',
    'pcm_fftMag_spectralEntropy_sma_amean',
    'pcm_fftMag_spectralVariance_sma_amean',
    'pcm_fftMag_spectralSkewness_sma_amean',
    'pcm_fftMag_spectralKurtosis_sma_amean',
    'pcm_fftMag_spectralSlope_sma_amean',
    'pcm_fftMag_psySharpness_sma_amean',
    'pcm_fftMag_spectralHarmonicity_sma_amean']


def av_to_emotion(arousal, valence):
    # if arousal == 0 and valence == 0:
    #     return "default"
    if arousal < 0:
        if valence < 0:
            return "melancholy"
        else:
            return "serene"
    else:
        if valence < 0:
            return "tense"
        else:
            return "euphoric"


def emotion_changes(emotions):
    prev = None
    out = []
    start = 0
    t = 0
    for emotion in emotions:
        t += 0.5
        if prev is None or emotion != prev:
            out.append((start, t, emotion))
            start = t
        prev = emotion
    return out


def emotion_changes_smooth(av):
    prev = None
    valence = 0
    arousal = 0
    out = []
    start = 0
    t = 0
    av = 0.8 * av + 0.2 * av.mean(dim=0, keepdim=True)
    for a, v in av:
        t += 0.5
        arousal += a.item()
        valence += v.item()
        emotion = av_to_emotion(arousal, valence)
        if prev is None or emotion != prev:
            out.append((start, t, prev))
            start = t
        prev = emotion
    out.append((start, t, prev))
    return out


def extract_emotions(audio_path, model_path):
    df = extract_opensmile(audio_path)
    opensmile = torch.from_numpy(df[KEEP].to_numpy()).float().cuda()
    opensmile = (opensmile - opensmile.mean(dim=0)) / opensmile.std(dim=0) # normalize
    model = load_checkpoint(model_path)
    with torch.no_grad():
        av = model(opensmile)
    print(av)
    # emotions = list(map(lambda x: av_to_emotion(x[0].item(), x[1].item()), av))
    # emotions = emotion_changes(emotions)
    emotions = emotion_changes_smooth(av)
    print(emotions)
    return emotions


def extract_emotions_debug(csv_path, model_path):
    opensmile = torch.from_numpy(pd.read_csv(csv_path)[KEEP].to_numpy()).float().cuda()
    opensmile = (opensmile - opensmile.mean(dim=0)) / opensmile.std(dim=0) # normalize
    model = load_checkpoint(model_path)
    with torch.no_grad():
        av = model(opensmile)
    print(av)
    # emotions = list(map(lambda x: av_to_emotion(x[0].item(), x[1].item()), av))
    # print(emotions)
    # emotions = emotion_changes(emotions)
    emotions = emotion_changes_smooth(av)
    print(emotions)
    return emotions


def load_checkpoint(path):
    print(f"Loading emotion extraction model from {path}")
    checkpoint = torch.load(path)
    model = UnimodalModel(159, 256, 2, 2).cuda()
    model.load_state_dict(checkpoint["model"])
    return model


def extract_opensmile(AUDIOFILE):

    test_smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.Functionals,)

    overall_smile_feat_set = set(test_smile.feature_names)

    lim_smile_feature_str = "F0final_sma_stddev;F0final_sma_amean;voicingFinalUnclipped_sma_stddev;voicingFinalUnclipped_sma_amean;jitterLocal_sma_stddev;jitterLocal_sma_amean;jitterDDP_sma_stddev;jitterDDP_sma_amean;shimmerLocal_sma_stddev;shimmerLocal_sma_amean;logHNR_sma_stddev;logHNR_sma_amean;audspec_lengthL1norm_sma_stddev;audspec_lengthL1norm_sma_amean;audspecRasta_lengthL1norm_sma_stddev;audspecRasta_lengthL1norm_sma_amean;pcm_RMSenergy_sma_stddev;pcm_RMSenergy_sma_amean;pcm_zcr_sma_stddev;pcm_zcr_sma_amean;audSpec_Rfilt_sma[0]_stddev;audSpec_Rfilt_sma[0]_amean;audSpec_Rfilt_sma[1]_stddev;audSpec_Rfilt_sma[1]_amean;audSpec_Rfilt_sma[2]_stddev;audSpec_Rfilt_sma[2]_amean;audSpec_Rfilt_sma[3]_stddev;audSpec_Rfilt_sma[3]_amean;audSpec_Rfilt_sma[4]_stddev;audSpec_Rfilt_sma[4]_amean;audSpec_Rfilt_sma[5]_stddev;audSpec_Rfilt_sma[5]_amean;audSpec_Rfilt_sma[6]_stddev;audSpec_Rfilt_sma[6]_amean;audSpec_Rfilt_sma[7]_stddev;audSpec_Rfilt_sma[7]_amean;audSpec_Rfilt_sma[8]_stddev;audSpec_Rfilt_sma[8]_amean;audSpec_Rfilt_sma[9]_stddev;audSpec_Rfilt_sma[9]_amean;audSpec_Rfilt_sma[10]_stddev;audSpec_Rfilt_sma[10]_amean;audSpec_Rfilt_sma[11]_stddev;audSpec_Rfilt_sma[11]_amean;audSpec_Rfilt_sma[12]_stddev;audSpec_Rfilt_sma[12]_amean;audSpec_Rfilt_sma[13]_stddev;audSpec_Rfilt_sma[13]_amean;audSpec_Rfilt_sma[14]_stddev;audSpec_Rfilt_sma[14]_amean;audSpec_Rfilt_sma[15]_stddev;audSpec_Rfilt_sma[15]_amean;audSpec_Rfilt_sma[16]_stddev;audSpec_Rfilt_sma[16]_amean;audSpec_Rfilt_sma[17]_stddev;audSpec_Rfilt_sma[17]_amean;audSpec_Rfilt_sma[18]_stddev;audSpec_Rfilt_sma[18]_amean;audSpec_Rfilt_sma[19]_stddev;audSpec_Rfilt_sma[19]_amean;audSpec_Rfilt_sma[20]_stddev;audSpec_Rfilt_sma[20]_amean;audSpec_Rfilt_sma[21]_stddev;audSpec_Rfilt_sma[21]_amean;audSpec_Rfilt_sma[22]_stddev;audSpec_Rfilt_sma[22]_amean;audSpec_Rfilt_sma[23]_stddev;audSpec_Rfilt_sma[23]_amean;audSpec_Rfilt_sma[24]_stddev;audSpec_Rfilt_sma[24]_amean;audSpec_Rfilt_sma[25]_stddev;audSpec_Rfilt_sma[25]_amean;pcm_fftMag_fband250-650_sma_stddev;pcm_fftMag_fband250-650_sma_amean;pcm_fftMag_fband1000-4000_sma_stddev;pcm_fftMag_fband1000-4000_sma_amean;pcm_fftMag_spectralRollOff25.0_sma_stddev;pcm_fftMag_spectralRollOff25.0_sma_amean;pcm_fftMag_spectralRollOff50.0_sma_stddev;pcm_fftMag_spectralRollOff50.0_sma_amean;pcm_fftMag_spectralRollOff75.0_sma_stddev;pcm_fftMag_spectralRollOff75.0_sma_amean;pcm_fftMag_spectralRollOff90.0_sma_stddev;pcm_fftMag_spectralRollOff90.0_sma_amean;pcm_fftMag_spectralFlux_sma_stddev;pcm_fftMag_spectralFlux_sma_amean;pcm_fftMag_spectralCentroid_sma_stddev;pcm_fftMag_spectralCentroid_sma_amean;pcm_fftMag_spectralEntropy_sma_stddev;pcm_fftMag_spectralEntropy_sma_amean;pcm_fftMag_spectralVariance_sma_stddev;pcm_fftMag_spectralVariance_sma_amean;pcm_fftMag_spectralSkewness_sma_stddev;pcm_fftMag_spectralSkewness_sma_amean;pcm_fftMag_spectralKurtosis_sma_stddev;pcm_fftMag_spectralKurtosis_sma_amean;pcm_fftMag_spectralSlope_sma_stddev;pcm_fftMag_spectralSlope_sma_amean;pcm_fftMag_psySharpness_sma_stddev;pcm_fftMag_psySharpness_sma_amean;pcm_fftMag_spectralHarmonicity_sma_stddev;pcm_fftMag_spectralHarmonicity_sma_amean;pcm_fftMag_mfcc_sma[1]_stddev;pcm_fftMag_mfcc_sma[1]_amean;pcm_fftMag_mfcc_sma[2]_stddev;pcm_fftMag_mfcc_sma[2]_amean;pcm_fftMag_mfcc_sma[3]_stddev;pcm_fftMag_mfcc_sma[3]_amean;pcm_fftMag_mfcc_sma[4]_stddev;pcm_fftMag_mfcc_sma[4]_amean;pcm_fftMag_mfcc_sma[5]_stddev;pcm_fftMag_mfcc_sma[5]_amean;pcm_fftMag_mfcc_sma[6]_stddev;pcm_fftMag_mfcc_sma[6]_amean;pcm_fftMag_mfcc_sma[7]_stddev;pcm_fftMag_mfcc_sma[7]_amean;pcm_fftMag_mfcc_sma[8]_stddev;pcm_fftMag_mfcc_sma[8]_amean;pcm_fftMag_mfcc_sma[9]_stddev;pcm_fftMag_mfcc_sma[9]_amean;pcm_fftMag_mfcc_sma[10]_stddev;pcm_fftMag_mfcc_sma[10]_amean;pcm_fftMag_mfcc_sma[11]_stddev;pcm_fftMag_mfcc_sma[11]_amean;pcm_fftMag_mfcc_sma[12]_stddev;pcm_fftMag_mfcc_sma[12]_amean;pcm_fftMag_mfcc_sma[13]_stddev;pcm_fftMag_mfcc_sma[13]_amean;pcm_fftMag_mfcc_sma[14]_stddev;pcm_fftMag_mfcc_sma[14]_amean;F0final_sma_de_stddev;F0final_sma_de_amean;voicingFinalUnclipped_sma_de_stddev;voicingFinalUnclipped_sma_de_amean;jitterLocal_sma_de_stddev;jitterLocal_sma_de_amean;jitterDDP_sma_de_stddev;jitterDDP_sma_de_amean;shimmerLocal_sma_de_stddev;shimmerLocal_sma_de_amean;logHNR_sma_de_stddev;logHNR_sma_de_amean;audspec_lengthL1norm_sma_de_stddev;audspec_lengthL1norm_sma_de_amean;audspecRasta_lengthL1norm_sma_de_stddev;audspecRasta_lengthL1norm_sma_de_amean;pcm_RMSenergy_sma_de_stddev;pcm_RMSenergy_sma_de_amean;pcm_zcr_sma_de_stddev;pcm_zcr_sma_de_amean;audSpec_Rfilt_sma_de[0]_stddev;audSpec_Rfilt_sma_de[0]_amean;audSpec_Rfilt_sma_de[1]_stddev;audSpec_Rfilt_sma_de[1]_amean;audSpec_Rfilt_sma_de[2]_stddev;audSpec_Rfilt_sma_de[2]_amean;audSpec_Rfilt_sma_de[3]_stddev;audSpec_Rfilt_sma_de[3]_amean;audSpec_Rfilt_sma_de[4]_stddev;audSpec_Rfilt_sma_de[4]_amean;audSpec_Rfilt_sma_de[5]_stddev;audSpec_Rfilt_sma_de[5]_amean;audSpec_Rfilt_sma_de[6]_stddev;audSpec_Rfilt_sma_de[6]_amean;audSpec_Rfilt_sma_de[7]_stddev;audSpec_Rfilt_sma_de[7]_amean;audSpec_Rfilt_sma_de[8]_stddev;audSpec_Rfilt_sma_de[8]_amean;audSpec_Rfilt_sma_de[9]_stddev;audSpec_Rfilt_sma_de[9]_amean;audSpec_Rfilt_sma_de[10]_stddev;audSpec_Rfilt_sma_de[10]_amean;audSpec_Rfilt_sma_de[11]_stddev;audSpec_Rfilt_sma_de[11]_amean;audSpec_Rfilt_sma_de[12]_stddev;audSpec_Rfilt_sma_de[12]_amean;audSpec_Rfilt_sma_de[13]_stddev;audSpec_Rfilt_sma_de[13]_amean;audSpec_Rfilt_sma_de[14]_stddev;audSpec_Rfilt_sma_de[14]_amean;audSpec_Rfilt_sma_de[15]_stddev;audSpec_Rfilt_sma_de[15]_amean;audSpec_Rfilt_sma_de[16]_stddev;audSpec_Rfilt_sma_de[16]_amean;audSpec_Rfilt_sma_de[17]_stddev;audSpec_Rfilt_sma_de[17]_amean;audSpec_Rfilt_sma_de[18]_stddev;audSpec_Rfilt_sma_de[18]_amean;audSpec_Rfilt_sma_de[19]_stddev;audSpec_Rfilt_sma_de[19]_amean;audSpec_Rfilt_sma_de[20]_stddev;audSpec_Rfilt_sma_de[20]_amean;audSpec_Rfilt_sma_de[21]_stddev;audSpec_Rfilt_sma_de[21]_amean;audSpec_Rfilt_sma_de[22]_stddev;audSpec_Rfilt_sma_de[22]_amean;audSpec_Rfilt_sma_de[23]_stddev;audSpec_Rfilt_sma_de[23]_amean;audSpec_Rfilt_sma_de[24]_stddev;audSpec_Rfilt_sma_de[24]_amean;audSpec_Rfilt_sma_de[25]_stddev;audSpec_Rfilt_sma_de[25]_amean;pcm_fftMag_fband250-650_sma_de_stddev;pcm_fftMag_fband250-650_sma_de_amean;pcm_fftMag_fband1000-4000_sma_de_stddev;pcm_fftMag_fband1000-4000_sma_de_amean;pcm_fftMag_spectralRollOff25.0_sma_de_stddev;pcm_fftMag_spectralRollOff25.0_sma_de_amean;pcm_fftMag_spectralRollOff50.0_sma_de_stddev;pcm_fftMag_spectralRollOff50.0_sma_de_amean;pcm_fftMag_spectralRollOff75.0_sma_de_stddev;pcm_fftMag_spectralRollOff75.0_sma_de_amean;pcm_fftMag_spectralRollOff90.0_sma_de_stddev;pcm_fftMag_spectralRollOff90.0_sma_de_amean;pcm_fftMag_spectralFlux_sma_de_stddev;pcm_fftMag_spectralFlux_sma_de_amean;pcm_fftMag_spectralCentroid_sma_de_stddev;pcm_fftMag_spectralCentroid_sma_de_amean;pcm_fftMag_spectralEntropy_sma_de_stddev;pcm_fftMag_spectralEntropy_sma_de_amean;pcm_fftMag_spectralVariance_sma_de_stddev;pcm_fftMag_spectralVariance_sma_de_amean;pcm_fftMag_spectralSkewness_sma_de_stddev;pcm_fftMag_spectralSkewness_sma_de_amean;pcm_fftMag_spectralKurtosis_sma_de_stddev;pcm_fftMag_spectralKurtosis_sma_de_amean;pcm_fftMag_spectralSlope_sma_de_stddev;pcm_fftMag_spectralSlope_sma_de_amean;pcm_fftMag_psySharpness_sma_de_stddev;pcm_fftMag_psySharpness_sma_de_amean;pcm_fftMag_spectralHarmonicity_sma_de_stddev;pcm_fftMag_spectralHarmonicity_sma_de_amean;pcm_fftMag_mfcc_sma_de[1]_stddev;pcm_fftMag_mfcc_sma_de[1]_amean;pcm_fftMag_mfcc_sma_de[2]_stddev;pcm_fftMag_mfcc_sma_de[2]_amean;pcm_fftMag_mfcc_sma_de[3]_stddev;pcm_fftMag_mfcc_sma_de[3]_amean;pcm_fftMag_mfcc_sma_de[4]_stddev;pcm_fftMag_mfcc_sma_de[4]_amean;pcm_fftMag_mfcc_sma_de[5]_stddev;pcm_fftMag_mfcc_sma_de[5]_amean;pcm_fftMag_mfcc_sma_de[6]_stddev;pcm_fftMag_mfcc_sma_de[6]_amean;pcm_fftMag_mfcc_sma_de[7]_stddev;pcm_fftMag_mfcc_sma_de[7]_amean;pcm_fftMag_mfcc_sma_de[8]_stddev;pcm_fftMag_mfcc_sma_de[8]_amean;pcm_fftMag_mfcc_sma_de[9]_stddev;pcm_fftMag_mfcc_sma_de[9]_amean;pcm_fftMag_mfcc_sma_de[10]_stddev;pcm_fftMag_mfcc_sma_de[10]_amean;pcm_fftMag_mfcc_sma_de[11]_stddev;pcm_fftMag_mfcc_sma_de[11]_amean;pcm_fftMag_mfcc_sma_de[12]_stddev;pcm_fftMag_mfcc_sma_de[12]_amean;pcm_fftMag_mfcc_sma_de[13]_stddev;pcm_fftMag_mfcc_sma_de[13]_amean;pcm_fftMag_mfcc_sma_de[14]_stddev;pcm_fftMag_mfcc_sma_de[14]_amean"

    lim_smile_feat_set = set(lim_smile_feature_str.split(';'))

    # Set difference of overall and lim_smile_feat_set
    overall_set_diff = overall_smile_feat_set.difference(lim_smile_feat_set)

    # Read audio.mp3 as chunks of 500 ms
    audio_chunks = []
    chunk_length_ms = 500 # pydub calculates in millisec

    overall_audio = AudioSegment.from_mp3(AUDIOFILE)

    for i in range(0, len(overall_audio), chunk_length_ms):
        audio_chunks.append(overall_audio[i:i+chunk_length_ms])

    # Export all of the individual chunks as wav files
    for i, chunk in enumerate(audio_chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")

    # Read all the wav files and extract features using test_smile
    overall_df = pd.DataFrame()
    for i in range(0, len(audio_chunks)):
        chunk_name = "chunk{0}.wav".format(i)
        df = test_smile.process_file(chunk_name)
        overall_df = overall_df.append(df)

    # Drop the columns that are in the set difference
    overall_df.drop(columns=overall_set_diff, inplace=True)

    # Reorder the df according to this order:
    feat_order_159 = ['audSpec_Rfilt_sma[1]_stddev', 'audspecRasta_lengthL1norm_sma_stddev', 'audSpec_Rfilt_sma[7]_stddev', 'audSpec_Rfilt_sma[8]_stddev', 'pcm_fftMag_spectralRollOff25.0_sma_de_stddev', 'audSpec_Rfilt_sma[5]_amean', 'audSpec_Rfilt_sma[12]_stddev', 'F0final_sma_amean', 'audSpec_Rfilt_sma_de[8]_stddev', 'pcm_fftMag_spectralVariance_sma_stddev', 'audspecRasta_lengthL1norm_sma_de_stddev', 'shimmerLocal_sma_de_stddev', 'audSpec_Rfilt_sma[20]_stddev', 'logHNR_sma_de_stddev', 'pcm_fftMag_spectralCentroid_sma_amean', 'audSpec_Rfilt_sma[7]_amean', 'jitterDDP_sma_de_stddev', 'audSpec_Rfilt_sma[6]_amean', 'pcm_fftMag_fband1000-4000_sma_stddev', 'F0final_sma_stddev', 'audSpec_Rfilt_sma[21]_amean', 'audSpec_Rfilt_sma_de[0]_stddev', 'audSpec_Rfilt_sma_de[25]_stddev', 'voicingFinalUnclipped_sma_amean', 'audSpec_Rfilt_sma[12]_amean', 'pcm_fftMag_spectralRollOff50.0_sma_stddev', 'logHNR_sma_amean', 'audSpec_Rfilt_sma[5]_stddev', 'audSpec_Rfilt_sma_de[15]_stddev', 'audSpec_Rfilt_sma_de[13]_stddev', 'audSpec_Rfilt_sma[11]_amean', 'audSpec_Rfilt_sma[15]_stddev', 'audSpec_Rfilt_sma_de[5]_stddev', 'pcm_fftMag_spectralSlope_sma_de_stddev', 'pcm_zcr_sma_stddev', 'pcm_fftMag_spectralFlux_sma_de_stddev', 'audspec_lengthL1norm_sma_amean', 'jitterLocal_sma_de_stddev', 'pcm_fftMag_spectralRollOff25.0_sma_amean', 'pcm_fftMag_spectralRollOff75.0_sma_stddev', 'pcm_fftMag_spectralRollOff90.0_sma_amean', 'pcm_fftMag_spectralKurtosis_sma_stddev', 'jitterLocal_sma_de_amean', 'audSpec_Rfilt_sma[9]_stddev', 'audSpec_Rfilt_sma[25]_stddev', 'audSpec_Rfilt_sma[16]_stddev', 'audSpec_Rfilt_sma[22]_stddev', 'pcm_fftMag_spectralHarmonicity_sma_stddev', 'pcm_fftMag_spectralHarmonicity_sma_de_stddev', 'audSpec_Rfilt_sma[13]_stddev', 'pcm_fftMag_spectralCentroid_sma_de_stddev', 'pcm_fftMag_fband1000-4000_sma_amean', 'pcm_fftMag_spectralFlux_sma_amean', 'voicingFinalUnclipped_sma_de_stddev', 'F0final_sma_de_stddev', 'audSpec_Rfilt_sma_de[22]_stddev', 'audSpec_Rfilt_sma[21]_stddev', 'audSpec_Rfilt_sma_de[21]_stddev', 'pcm_fftMag_spectralKurtosis_sma_amean', 'jitterDDP_sma_amean', 'jitterLocal_sma_stddev', 'pcm_fftMag_spectralHarmonicity_sma_amean', 'pcm_zcr_sma_amean', 'audSpec_Rfilt_sma[1]_amean', 'audSpec_Rfilt_sma[14]_stddev', 'audSpec_Rfilt_sma[4]_stddev', 'pcm_fftMag_spectralCentroid_sma_stddev', 'pcm_RMSenergy_sma_amean', 'audSpec_Rfilt_sma[10]_stddev', 'audspec_lengthL1norm_sma_de_stddev', 'pcm_fftMag_spectralEntropy_sma_de_stddev', 'audSpec_Rfilt_sma_de[20]_stddev', 'shimmerLocal_sma_amean', 'pcm_fftMag_spectralSkewness_sma_amean', 'audSpec_Rfilt_sma[3]_stddev', 'audSpec_Rfilt_sma_de[23]_stddev', 'pcm_fftMag_spectralKurtosis_sma_de_stddev', 'audSpec_Rfilt_sma_de[2]_stddev', 'audSpec_Rfilt_sma[20]_amean', 'audSpec_Rfilt_sma_de[16]_stddev', 'audSpec_Rfilt_sma_de[18]_stddev', 'pcm_fftMag_fband1000-4000_sma_de_stddev', 'audSpec_Rfilt_sma[2]_stddev', 'audSpec_Rfilt_sma[0]_amean', 'audSpec_Rfilt_sma_de[9]_stddev', 'audSpec_Rfilt_sma[8]_amean', 'pcm_zcr_sma_de_stddev', 'pcm_fftMag_spectralRollOff75.0_sma_de_stddev', 'jitterDDP_sma_stddev', 'audSpec_Rfilt_sma[0]_stddev', 'audSpec_Rfilt_sma[16]_amean', 'F0final_sma_de_amean', 'jitterLocal_sma_amean', 'audSpec_Rfilt_sma_de[4]_stddev', 'pcm_fftMag_fband250-650_sma_de_stddev', 'audSpec_Rfilt_sma[10]_amean', 'pcm_fftMag_spectralVariance_sma_de_stddev', 'voicingFinalUnclipped_sma_stddev', 'audSpec_Rfilt_sma[24]_stddev', 'pcm_fftMag_spectralRollOff50.0_sma_de_stddev', 'audSpec_Rfilt_sma[2]_amean', 'audSpec_Rfilt_sma[17]_amean', 'audSpec_Rfilt_sma[3]_amean', 'jitterDDP_sma_de_amean', 'audSpec_Rfilt_sma_de[7]_stddev', 'audSpec_Rfilt_sma_de[12]_stddev', 'audSpec_Rfilt_sma_de[11]_stddev', 'audSpec_Rfilt_sma[15]_amean', 'pcm_fftMag_spectralRollOff50.0_sma_amean', 'audSpec_Rfilt_sma[17]_stddev', 'audSpec_Rfilt_sma[24]_amean', 'audSpec_Rfilt_sma[19]_stddev', 'audSpec_Rfilt_sma[13]_amean', 'pcm_RMSenergy_sma_de_stddev', 'audSpec_Rfilt_sma_de[24]_stddev', 'pcm_fftMag_spectralSkewness_sma_de_stddev', 'pcm_fftMag_spectralRollOff90.0_sma_de_stddev', 'pcm_fftMag_fband250-650_sma_amean', 'pcm_fftMag_spectralVariance_sma_amean', 'audSpec_Rfilt_sma[23]_stddev', 'pcm_fftMag_spectralEntropy_sma_stddev', 'audSpec_Rfilt_sma_de[17]_stddev', 'audSpec_Rfilt_sma[22]_amean', 'audSpec_Rfilt_sma_de[6]_stddev', 'pcm_fftMag_spectralEntropy_sma_amean', 'pcm_fftMag_spectralRollOff90.0_sma_stddev', 'audspecRasta_lengthL1norm_sma_amean', 'audSpec_Rfilt_sma_de[1]_stddev', 'audspec_lengthL1norm_sma_stddev', 'pcm_fftMag_spectralRollOff75.0_sma_amean', 'pcm_fftMag_fband250-650_sma_stddev', 'pcm_RMSenergy_sma_stddev', 'audSpec_Rfilt_sma[18]_amean', 'audSpec_Rfilt_sma_de[19]_stddev', 'audSpec_Rfilt_sma[11]_stddev', 'audSpec_Rfilt_sma[14]_amean', 'audSpec_Rfilt_sma[23]_amean', 'logHNR_sma_de_amean', 'audSpec_Rfilt_sma_de[10]_stddev', 'audSpec_Rfilt_sma_de[3]_stddev', 'audSpec_Rfilt_sma[18]_stddev', 'audSpec_Rfilt_sma[19]_amean', 'pcm_fftMag_spectralFlux_sma_stddev', 'shimmerLocal_sma_de_amean', 'pcm_fftMag_psySharpness_sma_stddev', 'pcm_fftMag_psySharpness_sma_de_stddev', 'audSpec_Rfilt_sma[4]_amean', 'audSpec_Rfilt_sma[25]_amean', 'pcm_fftMag_spectralSkewness_sma_stddev', 'shimmerLocal_sma_stddev', 'pcm_fftMag_spectralRollOff25.0_sma_stddev', 'audSpec_Rfilt_sma_de[14]_stddev', 'pcm_fftMag_psySharpness_sma_amean', 'pcm_fftMag_spectralSlope_sma_stddev', 'voicingFinalUnclipped_sma_de_amean', 'audSpec_Rfilt_sma[9]_amean', 'logHNR_sma_stddev', 'audSpec_Rfilt_sma[6]_stddev', 'pcm_fftMag_spectralSlope_sma_amean']
    feat_order_159 = feat_order_159.insert(0, 'end')
    feat_order_159 = feat_order_159.insert(0, 'start')

    overall_df.loc[:, feat_order_159]

    return overall_df


# def bert_embedding(text):

