import librosa
import numpy as np
import joblib

def extract_features(file_path):
    y, sr =librosa.load(file_path, sr = 16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)
def load_model(path="model/accent_model.pkl"):
    return joblib.load(path)

def predict_accent(file_path, model):
    featurs = extract_features(file_path)
    features = features.reshape(1,-1)
    return model.predict(features)[0]
