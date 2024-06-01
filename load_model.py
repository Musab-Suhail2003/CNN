import os

import PIL.Image
from PIL import Image
import joblib
import numpy as np
from dataset_loader import load_cifar_test
import cv2

loaded_model = joblib.load("trained_model.joblib")

file = os.listdir("SAMPLE_IMAGES")
X = []
Y = []
for filename in file:
    Y.append(filename)
    # Load image using OpenCV
    path = os.path.join(os.path.join("SAMPLE_IMAGES", filename))
    image = PIL.Image.open(path)
    image = np.array(image)/255.0
    image.resize((32, 32))
    X.append(image)
X = np.array(X)
X= np.expand_dims(X, axis=-1)

def predict(X):
    pred, conf = loaded_model.predict(X)
    conf *= 100
    conf = conf//1
    print(f'PREDICTION: {pred}, CONFIDENCE: {conf}')

