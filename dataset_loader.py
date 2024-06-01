import os
import random

import PIL
import cv2
import numpy as np
import pickle

from PIL import Image


"""def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch[b'data'], batch[b'labels']


def load_cifar_data():
    data_batches = []
    label_batches = []
    for i in range(1,2):  # Load training data batches
        data, labels = load_cifar_batch(f'cifar-10-python/cifar-10-batches-py/data_batch_{i}')
        data_batches.append(data)
        label_batches.append(labels)

    # Concatenate data and labels
    train_data = np.concatenate(data_batches, axis=0)
    train_labels = np.concatenate(label_batches, axis=0)
    train_data = train_data[:1000]
    train_labels = train_labels[:1000]
    # Reshape data to (num_samples, 32, 32, 3)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        # Normalize pixel values
    train_data = train_data / 255.0
    return train_data, train_labels

def load_cifar_test(path = 'cifar-10-python/cifar-10-batches-py/test_batch'):
    test_data, test_labels = load_cifar_batch(path)  # Load test data
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data / 255.0
    return test_data[0:100], test_labels[0:100]"""

def load_human_dataset(path0 = "human detection dataset"):
    images, labels = [], []
    for filename in os.listdir(path0):
        path = os.path.join(path0, filename)
        if(filename == '0'):
            for nonhuman in os.listdir(path):
                img = PIL.Image.open(os.path.join(path, nonhuman))

                img = np.array(img)
                img.resize((32, 32, 3))
                images.append(img)
                labl = 0
                labels.append(labl)
        else:
            for human in os.listdir(path):
                img = PIL.Image.open(os.path.join(path, human))

                img = np.array(img)
                img.resize((128,128, 3))
                images.append(img)
                labl = 1
                labels.append(labl)

    combine = list(zip(images, labels))
    random.shuffle(combine)
    images, labels = zip(*combine)
    images = np.array(images)
    labels = np.array(labels)
    return images[0:2], labels[0:2]


"""im, l = load_human_dataset()
image = im[im.shape[0]-1]
print(image.shape)
image = Image.fromarray(image)
image.show()
print(l[-1])"""