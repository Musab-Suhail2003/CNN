import PIL.Image
import joblib
import numpy as np

loaded_model = joblib.load("trained_model.joblib")


def predict(X):
    pred, conf = loaded_model.predict(X)
    conf *= 100
    conf = conf//1
    pred = "human" if pred == 1 else "non human"
    print(f'PREDICTION: {pred}, CONFIDENCE: {conf[0]}%')


def PREDICT(image):
    image = np.array(image)/255.0
    image.resize((32, 32, 3))
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    predict(image)


def main(image=None):
    PREDICT(image)


if __name__ == '__main__':
    image = PIL.Image.open('SAMPLE_IMAGES/CAT.jpeg')
    image1 = PIL.Image.open('SAMPLE_IMAGES/HUMAN.jpeg')
    main(image)
    main(image1)
