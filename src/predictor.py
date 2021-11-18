'''

This scripts loads an already trained model and show its prediction on a single image
'''


import importlib
import src.padding as p
import matplotlib.pyplot as plt
import src.Models as mdl
import numpy as np
import tensorflow as tf
import sys
from pathlib import PurePath
import random as r

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = []

class_types = {'fashion':fashion_mnist_classes, 'mnist':mnist_classes, 'cifar10':cifar10_classes}
data_types = {'fashion':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist, 'cifar10':tf.keras.datasets.cifar10}
models = {"modelA": mdl.modelA, "modelB": mdl.modelB}
train_mode = {"CTR": "ctr", "CBC": "cbc", "ECB": "ecb",
              "PERMUTATED": "permutated", "UNENCRYPTED": "unencrypted"}

DATASET = "DATASET"
MODEL = "MODEL"
TRAIN_WITH_ME = "TRAIN_WITH_ME"
PADDING = "PADDING"
NORM = "NORM"
FILE = "FILE"
INDEX = "INDEX"

params = {DATASET: None, MODEL: None, TRAIN_WITH_ME: None, PADDING: 0, NORM: 0, INDEX:-1}


def plot_image(predictions, true_label, img):
    '''
    Draws the image. ONLY for greyscale images (the third dimension should be 1)
    :param predictions: vector of probablities
    :param true_label: the true label
    :param img: the image itself
    :return:
    '''
    img += params[NORM]

    # in order to plot the image, we need a 2D array
    if len(np.array(img).shape) == 3:
        img = img[:, :, 0]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary, vmin=0.0, vmax=1.0)

    predicted_label = np.argmax(predictions)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions),
                                         class_names[true_label]), color=color)
    plt.show()


def predict(model, img, label):
    predictions = model.predict(img)

    d = img.shape[1]
    img = np.reshape(img, (d, d))

    plot_image(predictions, label, img)


def main():
    global class_names

    data = data_types[params[DATASET]]
    class_names = class_types[params[DATASET]]

    # loading the data
    _, (x_test, y_test) = data.load_data()

    img = x_test[params[INDEX]]
    label = y_test[params[INDEX]]

    # normalizing
    img = img / 255.0 - params[NORM]

    # padding
    img = p.pad(img, number_of_paddings=params[PADDING], padder=0.0 - params[NORM])

    d = img.shape[1]
    img = np.reshape(img, (1, d, d, 1))

    helper = importlib.import_module("src.encryptions." + train_mode[params[TRAIN_WITH_ME]])

    if params[TRAIN_WITH_ME] in ["ECB", "CBC", "CTR"]:
        helper.NORM = params[NORM]

    input_shape = np.array(img[0]).shape

    # getting the desired model
    model = models[params[MODEL]](input_shape, encrypt=helper.encrypt)
    model.load(params[FILE])

    predict(model, img, label)


if __name__ == '__main__':
    # getting command line arguments
    if len(sys.argv) == 1:
        print("Missing arguments. Try -h for help")
        exit()

    if sys.argv[1] == '-h':
        print("python .\src\\predictor.py [-h] <-f filename> [-i index]")
        print("\t-h\tshow this help text")
        print("\t-f\tspecifying the filename of the model <must>")
        print("\t-i\tspecifying the index, if non specified than randomly chosen [optional]")
        exit()

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-f':
            if ".h5" in sys.argv[i + 1]:
                params[FILE] = PurePath(sys.argv[i+1]).parts[-1].split(".h5")[0]
            else:
                params[FILE] = sys.argv[i + 1]
        if sys.argv[i] == '-i':
            params[INDEX] = int(sys.argv[i + 1], 10)

    if params[INDEX] == -1:
        params[INDEX] = r.randint(0, 999)

    MODEL_NAME = params[FILE]

    name_parts = MODEL_NAME.split('_')
    for i,part in enumerate(name_parts):
        if part == "mnist" or part == "fashion":
            params[DATASET] = part
        if part == "modelA" or part == "modelB":
            params[MODEL] = part
        if part == "UNENCRYPTED" or part == "PERMUTATED" or part == "ECB" or part == "CBC" or part == "CTR":
            params[TRAIN_WITH_ME] = part
        if "NORM" in part:
            params[NORM] = float(part.split("NORM")[0])
        if "PADDED" in part:
            params[PADDING] = int(part.split("PADDED")[0])

    print("DATASET\t= {}".format(params[DATASET]))
    print("MODEL\t= {}".format(params[MODEL]))
    print("TRAINER\t= {}".format(params[TRAIN_WITH_ME]))
    print("NORM\t= {}".format(params[NORM]))
    print("PADDING\t= {}".format(params[PADDING]))
    print("NAME\t= {}".format(MODEL_NAME))
    print("INDEX\t= {}".format(params[INDEX]))

    main()
