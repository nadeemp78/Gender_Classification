#test_gender.py
import os
import _pickle as cPickle
import numpy as np
import librosa
from sklearn import preprocessing

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

    feat     = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i,:]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat;
    features = preprocessing.scale(features)
    return features


sourcepath = "data/test/male/"      
  
modelpath  = "models/"   

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
genders   = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

files     = [os.path.join(sourcepath,f) for f in os.listdir(sourcepath) if f.endswith(".wav")] 

for f in files:
    print(f)
    features   = feature_extraction(f)
    scores     = None
    log_likelihood = np.zeros(len(models)) 
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(features))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print("detected as - " + genders[winner])
