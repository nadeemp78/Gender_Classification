import os
import _pickle as cPickle
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import librosa

def feature_extraction(wav_path):
    y, sr = librosa.load(wav_path)
    n_mfcc = 13
    n_mels = 40
    win_length = 552 # 22050*0.025
    hop_length = 221 # 22050*0.010
    n_fft = 1024
    window = 'hamming'
    fmin = 20
    fmax = 3800
    D = np.abs(librosa.stft(y, window=window, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**2
    S = librosa.feature.melspectrogram(S=D, y=y, n_mels=n_mels, fmin=fmin, fmax=fmax)		
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
    features = mfcc_delta_delta.T 
    features = preprocessing.scale(features)
    return features

def gmm_model(x,y):
    gmm = GMM(n_components = 2, max_iter = 100, covariance_type='diag', n_init = 10)
    gmm.fit(features,classes)
    return gmm

#main
source   = "data/train/"   
dest     = "models/"

for f in os.listdir(source) :
    folders = os.path.join(source,f)
    files    = [os.path.join(folders,f1) for f1 in os.listdir(folders) if f1.endswith('.wav') ]

    features = np.asarray(());
    classes = np.asarray(())
    for f in files:
        vector = feature_extraction(f)
        if features.size == 0:
            features = vector
            classes = folders.split('/')[2]
        else:
            features = np.vstack((features, vector))
            classes = folders.split('/')[2]

    gmm = gmm_model(features, classes)

    picklefile = folders.split('/')[2]+".gmm"
    cPickle.dump(gmm,open(dest + picklefile,'wb'))
    print('modeling completed for gender:',picklefile)
