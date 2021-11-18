'''
This scripts build models according to various parameters (encryption, padding, architecture etc.)
'''
import importlib
import src.Models as mdl
import numpy as np
import tensorflow as tf
import json
import src.padding as p
import sys
from keras.datasets import mnist #extra added

#added to resolve SSL issues
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

#for the AVX error for tensorflow


data_types = {'fashion': tf.keras.datasets.fashion_mnist, 'mnist': tf.keras.datasets.mnist,#mnist.load_data(),#
              'cifar10': tf.keras.datasets.cifar10}
models = {"modelA": mdl.modelA, "modelB": mdl.modelB}
train_mode = {"CTR": "ctr", "CBC": "cbc", "ECB": "ecb",
              "PERMUTATED": "permutated", "UNENCRYPTED": "unencrypted"}

DATASET = "DATASET"
MODEL = "MODEL"
TRAIN_WITH_ME = "TRAIN_WITH_ME"
PADDING = "PADDING"
NORM = "NORM"

params = {DATASET: None, MODEL: None, TRAIN_WITH_ME: "UNENCRYPTED", PADDING: 0, NORM: 0}


def main():
    data = data_types[params[DATASET]]
    # print("data")
    try:
        (x_train, y_train), (x_test, y_test) = data.load_data()
    except:
        print("issue in printing the data")
    # normalizing
    # print("train & test shapes:",x_train.shape,y_train.shape)
    x_train, x_test = x_train / 255.0 - params[NORM], x_test / 255.0 - params[NORM]

    # padding
    x_train = [p.pad(img, number_of_paddings=params[PADDING], padder=0.0) for img in x_train]
    x_test = [p.pad(img, number_of_paddings=params[PADDING], padder=0.0) for img in x_test]

    helper = importlib.import_module("src.encryptions." + train_mode[params[TRAIN_WITH_ME]])

    if params[TRAIN_WITH_ME] in ["ECB", "CBC", "CTR"]:
        helper.NORM = params[NORM]

    dims = np.array(x_train).shape

    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_train[0]).shape

    # getting the desired model
    model = models[params[MODEL]](input_shape, encrypt=helper.encrypt)
    print(model.summary())
    # training
    loss, epoch_accs, epochs = model.train(x_train, y_train, ep=5)

    # evaluating
    model.compile()
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # r.write("{}\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(MODEL_NAME, 100 * test_acc, (1.0 - test_acc) * 100))
    # helper.print_encryption_details(out=r)
    # r.write("#####################################################\n")

    # results = {
    #     "name": MODEL_NAME,
    #     "acc": epoch_accs,
    #     "epochs": epochs
    # }
    #
    # # writing the training results
    # with open("json/{}_{}_models.json".format(DATASET, TRAIN_WITH_ME), 'a') as j:
    #     json.dump(results, j)
    #     j.write('\n')
    #
    # # writing the training results
    # with open("json/{}_{}{}_models.json".format(DATASET, TRAIN_WITH_ME, VERSION), 'a') as j:
    #     json.dump(results, j)
    #     j.write('\n')

    # saving model
    model.save(MODEL_NAME)


if __name__ == '__main__':
    # getting command line arguments
    if len(sys.argv) == 1:
        print("Missing arguments. Try -h for help")
        exit()

    if sys.argv[1] == '-h':
        print("\npython .\src\\trainer.py  [-h] <-d dataset> <-m architecture> [-e encryption] [-p padsize] [-n normalization]")
        print("\t-h\tshow this help text")
        print("\t-d\tspecifying the dataset; mnist or fashion <must>")
        print("\t-m\tspecifying the model architecture; modelA or modelB <must>")
        print("\t-e\tspecifying the encryption method; UNENCRYPTED, PERMUTATED, ECB, CBC or CTR. default is UNENCRYPTED [optional]")
        print("\t-p\tspecifying the number of rows to pad, default is 0 [optional]")
        print("\t-n\tspecifying the normalization (img / 255.0 - n), default is 0 [optional]")
        exit()

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-d':
            params[DATASET] = sys.argv[i+1]
        if sys.argv[i] == '-m':
            params[MODEL] = sys.argv[i+1]
        if sys.argv[i] == '-e':
            params[TRAIN_WITH_ME] = sys.argv[i+1]
        if sys.argv[i] == '-p':
            params[PADDING] = int(sys.argv[i+1],10)
        if sys.argv[i] == '-n':
            params[NORM] = float(sys.argv[i+1])

    MODEL_NAME = params[DATASET] + "_" + params[MODEL] + "_" + params[TRAIN_WITH_ME] + "_" + str(params[NORM]) + "NORM_" + str(params[PADDING]) + "PADDED"

    print("DATASET\t= {}".format(params[DATASET]))
    print("MODEL\t= {}".format(params[MODEL]))
    print("TRAINER\t= {}".format(params[TRAIN_WITH_ME]))
    print("NORM\t= {}".format(params[NORM]))
    print("PADDING\t= {}".format(params[PADDING]))
    print("NAME\t= {}".format(MODEL_NAME))

    try:
        # r = open("results", "a")
        main()
        # r.close()
    except:
        print("\n\tBad configuration. Try -h for help")
        exit()
