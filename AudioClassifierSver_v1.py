import librosa
import librosa.display
import numpy as np
import os, sys, copy, time, pickle


def AudioFeatures(filename):
    y, sr = librosa.load(filename, mono=True, duration=30)
    rms = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = [filename, str(np.mean(chroma_stft)), str(np.mean(rms)), str(np.mean(spec_cent)),
                     str(np.mean(spec_bw)), str(np.mean(rolloff)), str(np.mean(zcr))]    
    for e in mfcc:
        to_append = to_append + [np.mean(e)] #f' {np.mean(e)}'
    return np.array(to_append[1:]).astype(float)
    

def AClassifier(querry, weights=r'last_model.pkl', treshold=0.7):
    Q = AudioFeatures(querry)
    ff = open(weights, 'rb')
    W_forest_new = pickle.load(ff)
    ff.close()
    W_forest_new_ = pickle.loads(W_forest_new)
    preds = (W_forest_new_.predict_proba([Q, ])[0][1]>treshold)*1
    if preds:
        print('У данного кандидата нет существенных дефектов речи.')
        return W_forest_new_.predict_proba([Q, ])
    else:
        print('У данного кандидата присутсвуют дефекты речи!')
        return W_forest_new_.predict_proba([Q, ])