'''

This script is running an attack on a dataset of images (mnist or fashion)
'''


import importlib
from pathlib import PurePath
import src.padding as p
import tensorflow as tf
import numpy as np
import src.Models as m
import sys

data_types = {'fashion':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist, 'cifar10':tf.keras.datasets.cifar10}
models = {"modelA": m.modelA, "modelB": m.modelB}

train_mode = {"CTR": "ctr", "CBC": "cbc", "ECB": "ecb",
              "PERMUTATED": "permutated", "UNENCRYPTED": "unencrypted"}


DATASET = "DATASET"
MODEL = "MODEL"
TRAIN_WITH_ME = "TRAIN_WITH_ME"
PADDING = "PADDING"
NORM = "NORM"
FILE = "FILE"
INDEX = "INDEX"
CARLINI = "CARLINI"
AMOUNT = "AMOUNT"

params = {DATASET: None, TRAIN_WITH_ME: None, PADDING: 0, NORM: 0, FILE:None, AMOUNT:1000, CARLINI:None}


'''
saving the indexes of images that were successfully attacked and the ones not
'''
def save_indexes(model, adv, labels, name):
    good = 0.0
    bad = 0.0

    safe = open("safe_indexes_{}".format(name), 'w')
    unsafe = open("unsafe_indexes_{}".format(name), 'w')

    for i in range(len(adv)):
        prob_adv = model.predict(np.float32(np.reshape(adv[i], (1, 28, 28, 1))))[0]  # the output is 2D array
        pred_adv = np.argmax(prob_adv).tolist()

        prob_orig = model.predict(np.float32(np.reshape(x_test[i], (1, 28, 28, 1))))[0]  # the output is 2D array
        pred_orig = np.argmax(prob_orig).tolist()

        good += y_test[i] == pred_adv
        bad += y_test[i] != pred_adv

        if pred_adv == labels[i]:
            safe.write("{}\n".format(i))
        else:
            if pred_orig == labels[i]:
                unsafe.write("{}*\n".format(i))  # images that the attacker successfully misleaded the model
            else:
                unsafe.write("{}\n".format(i))  # the model is wrong
    safe.close()
    unsafe.close()

    test_acc = good / (good + bad)
    print("accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))


# getting command line arguments
if len(sys.argv) == 1:
    print("Missing arguments. Try -h for help")
    exit()

if sys.argv[1] == '-h':
    print("python .\src\\dataset_attack.py [-h] <-f filename> [-i amount] [-c CW_mode]")
    print("\t-h\tshow this help text")
    print("\t-f\tspecifying the filename of the model <must>")
    print("\t-i\tspecifying the amount, default is 1000 [optional]")
    print("\t-c\tspecifying carlini mode; 2,0 or i. default is 2 [optional]")
    exit()

for i in range(1, len(sys.argv)):
    if sys.argv[i] == '-f':
        if ".h5" in sys.argv[i + 1]:
            params[FILE] = PurePath(sys.argv[i + 1]).parts[-1].split(".h5")[0]
        else:
            params[FILE] = sys.argv[i + 1]
    if sys.argv[i] == '-i':
        params[AMOUNT] = int(sys.argv[i + 1], 10)
    if sys.argv[i] == '-c':
        params[CARLINI] = sys.argv[i+1]

MODEL_NAME = params[FILE]

name_parts = MODEL_NAME.split('_')
for i, part in enumerate(name_parts):
    if part == "modelA":
        params[MODEL] = part
        from src.CW import CW_attack as a
        a.set_mode(params[CARLINI])
    if part == "modelB":
        params[MODEL] = part
        from src.FGSM import FGSM_attack as a
    if part == "mnist" or part == "fashion":
        params[DATASET] = part
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
print("AMOUNT\t= {}".format(params[AMOUNT]))

data = data_types[params[DATASET]]

# loading data
_, (x_test, y_test) = data.load_data()

x_test = x_test[:params[AMOUNT]]
y_test = y_test[:params[AMOUNT]]

x_test = x_test / 255.0 - params[NORM]
if len(np.array(x_test).shape) != 4:
    # expanding the images to get a third dimension (needed for conv layers)
    x_test = np.expand_dims(x_test, -1)

# gray-box attack, so the attacker gets to know only the architecture of the model
# which is like giving him the unencrypted version
lab_model = MODEL_NAME.replace("PERMUTATED", "UNENCRYPTED")
lab_model = lab_model.replace("ECB", "UNENCRYPTED")
lab_model = lab_model.replace("CBC", "UNENCRYPTED")
lab_model = lab_model.replace("CTR", "UNENCRYPTED")

adv = a.attack(x_test, y_test, lab_model, evaluate=True)

helper = importlib.import_module("src.encryptions." + train_mode[params[TRAIN_WITH_ME]])
if params[TRAIN_WITH_ME] in ["ECB", "CBC", "CTR"]:
    helper.NORM = params[NORM]

input_shape = np.array(x_test[0]).shape
model = models[params[MODEL]](input_shape, encrypt=helper.encrypt)
model.load(MODEL_NAME)

_, test_acc = model.evaluate(adv, y_test)
print("accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))
# r = open("attacked_ctr", 'a')
# r.write("{}\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(MODEL_NAME, 100 * test_acc, (1.0 - test_acc) * 100))
# r.close()
# save_indexes(model, adv, y_test, MODEL_NAME)