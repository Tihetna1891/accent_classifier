import os
import librosa
import numpy as np
from sklearn.svm import SVC
import joblib

data_dir = "data/"
X, y =[],[]
for lable in os.listdir(lable_dir):
    label_dir = os.path.join(data_dir, label)
    for file in os.listdir(label_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(label_dir,file)
            try: 
                audio, sr = librosa.load(file_path, sr = 16000)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                feature = np.mean(mfcc.T, axis=0)
                X.append(feature)
                y.append(label)
            except Exception as e:
                print(f"Error processing{file_path}: {e}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

model = SVC(kernel = 'linear', probability = True)
model.fit(X_train, y_train)
joblib.dump(model, 'model/accent_model.pkl')
print("Model trained and saved to model/accent_model.pkl")