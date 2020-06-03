import librosa
import librosa.display
import numpy as np
import os, sys, copy, time, pickle


class AudioClassifier():

    def __init__(self, threshold=0.7):
        """
        :param threshold:
        """
        self.treashold = threshold
        
    def HH(self):
        return 'Hyev class'
    
    def load(self, weights):
        """
        :param path:
        :return:
        """
        ff = open(weights, 'rb')
        W_forest_new = pickle.load(ff)
        ff.close()
        W_forest_new_ = pickle.loads(W_forest_new)
        self.Algorithm = W_forest_new_
        return self.Algorithm

    def transform(self, filename):
        """
        :param audio:
        :return:
        """
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
        self.data = np.array(to_append[1:]).astype(float)
        return self.data

    def predict(self, audio):
        """
        :param audio:
        :return:
        """
        res = None
        features = self.transform(audio)
        try: 
            res = self.Algorithm.predict_proba(features)
        except ValueError: 
            res = self.Algorithm.predict_proba(np.expand_dims(features, axis=0))
        return res