import torch


def av_to_emotion(arousal, valence):
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


# def bert_embedding(text):
    
