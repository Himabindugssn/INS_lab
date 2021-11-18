'''
A script designed to load an already trained model and evaluate it on the test-set
'''


from pathlib import PurePath
import tensorflow as tf
import numpy as np
import sys
import src.Models as mdl
import src.padding as p
import importlib

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
AMOUNT = "AMOUNT"

params = {DATASET: None, MODEL: None, TRAIN_WITH_ME: None, PADDING: 0, NORM: 0, AMOUNT:10000}

# getting command line arguments
if len(sys.argv) == 1:
    print("Missing arguments. Try -h for help")
    exit()

if sys.argv[1] == '-h':
    print("python .\src\\evaluation.py [-h] <-f filename> [-n amount]")
    print("\t-h\tshow this help text")
    print("\t-f\tspecifying the filename of the model <must>")
    print("\t-n\tspecifying the amount of images, default is 10000 [optional]")
    exit()

for i in range(1, len(sys.argv)):
    if sys.argv[i] == '-f':
        if ".h5" in sys.argv[i + 1]:
            params[FILE] = PurePath(sys.argv[i + 1]).parts[-1].split(".h5")[0]
        else:
            params[FILE] = sys.argv[i + 1]
    if sys.argv[i] == '-n':
        params[AMOUNT] = int(sys.argv[i + 1], 10)


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
print("AMOUNT\t= {}".format(params[AMOUNT]))
print("NAME\t= {}".format(MODEL_NAME))

data = data_types[params[DATASET]]

# loading data
(x_train, y_train), (x_test, y_test) = data.load_data()

x_test = x_test[:params[AMOUNT]]
y_test = y_test[:params[AMOUNT]]

# normalizing
x_test = x_test / 255.0 - params[NORM]

# padding
x_test = np.array([p.pad(img, number_of_paddings=params[PADDING], padder=0.0) for img in x_test])

helper = importlib.import_module("src.encryptions." + train_mode[params[TRAIN_WITH_ME]])

if params[TRAIN_WITH_ME] in ["ECB", "CBC", "CTR"]:
    helper.NORM = params[NORM]

dims = x_test.shape

if len(dims) != 4:
    # expanding the images to get a third dimension (needed for conv layers)
    x_test = np.expand_dims(x_test, -1)

input_shape = np.array(x_test[0]).shape

# getting the desired model
model = models[params[MODEL]](input_shape, encrypt=helper.encrypt)
model.load(params[FILE])

test_loss, test_acc = model.evaluate(x_test, y_test)

print("\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))
