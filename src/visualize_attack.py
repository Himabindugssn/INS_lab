'''
This script visualizes an attack on a single image, it plots the original image, the noise and the adversarial image
'''


import importlib
from pathlib import PurePath
import src.padding as p
import tensorflow as tf
import numpy as np
import src.Models as m
import matplotlib.pyplot as plt
import sys
import random as r

data_types = {'fashion':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist, 'cifar10':tf.keras.datasets.cifar10}
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_types = {'fashion':fashion_mnist_classes, 'mnist':mnist_classes, 'cifar10':cifar10_classes}
class_names = []
models = {"modelA": m.modelA, "modelB": m.modelB}

train_mode = {"CTR": "ctr", "CBC": "cbc", "ECB": "ecb",
              "PERMUTATED": "permutated", "UNENCRYPTED": "unencrypted"}
attacks = {"modelA":"CW", "modelB":"FGSM"}


DATASET = "DATASET"
MODEL = "MODEL"
TRAIN_WITH_ME = "TRAIN_WITH_ME"
PADDING = "PADDING"
NORM = "NORM"
FILE = "FILE"
INDEX = "INDEX"
CARLINI = "CARLINI"

params = {DATASET: None, TRAIN_WITH_ME: None, PADDING: 0, NORM: 0, FILE:None, CARLINI:None, INDEX:-1}


def multicolor_xlabel(ax,list_of_strings,list_of_colors, h=0.0, anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker

    boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw))
                for text,color in zip(list_of_strings,list_of_colors) ]
    xbox = HPacker(children=boxes,align="bottom",pad=0, sep=5)
    anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0,h),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_xbox)


def plot_original_adversarial(orig_img, adv_img, noise, orig_prob_attacked, orig_prob_safe, adv_prob_attacked, adv_prob_safe, true_label):
    if len(np.array(orig_img).shape) > 2:
        orig_img = orig_img[:, :, 0]

    if len(np.array(adv_img).shape) > 2:
        adv_img = adv_img[:, :, 0]

    if len(np.array(noise).shape) > 2:
        noise = noise[:, :, 0]

    orig_img += params[NORM]
    adv_img += params[NORM]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    axes = [ax1, ax2, ax3]
    images = [orig_img, noise, adv_img]
    titles = ["Original", "Noise", "Adversarial"]
    probs = [[orig_prob_attacked, orig_prob_safe], [], [adv_prob_attacked, adv_prob_safe]]
    preds = [[np.argmax(orig_prob_attacked), np.argmax(orig_prob_safe)], [], [np.argmax(adv_prob_attacked), np.argmax(adv_prob_safe)]]

    for i in range(3):
        axes[i].grid(False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].title.set_text(titles[i])
        # axes[i].axis('off')

        axes[i].imshow(images[i], cmap=plt.cm.binary, vmin=0.0, vmax=1.0)  # row=0, col=0

        if i == 1:
            continue

        if preds[i][0] == true_label:
            color0 = 'green'
        else:
            color0 = 'red'

        if preds[i][1] == true_label:
            color1 = 'green'
        else:
            color1 = 'red'

        label0 = "{:2.0f}% it's {}".format(100 * np.max(probs[i][0]), class_names[preds[i][0]])
        label1 = "{:2.0f}% it's {}".format(100 * np.max(probs[i][1]), class_names[preds[i][1]])
        multicolor_xlabel(axes[i], ["insecured: ", label0], ['black', color0], h=-0.18)
        multicolor_xlabel(axes[i], ["secured: ", label1], ['black', color1], h=-0.30)

    plt.suptitle("\nencryption: " + params[TRAIN_WITH_ME] + "\nattack: " + attacks[params[MODEL]]
                 + "\n\ntrue label: " + class_names[true_label], fontsize=15, fontweight='bold', color=(74.0/255.0, 146.0/255.0, 156.0/255.0))
    plt.show()


# getting command line arguments
if len(sys.argv) == 1:
    print("Missing arguments. Try -h for help")
    exit()

if sys.argv[1] == '-h':
    print("python .\src\\visualize_attack.py [-h] <-f filename> [-i index] [-c CW_mode]")
    print("\t-h\tshow this help text")
    print("\t-f\tspecifying the filename of the model <must>")
    print("\t-i\tspecifying the index, if non specified than randomly chosen [optional]")
    print("\t-c\tspecifying carlini mode; 2,0 or i. default is 2 [optional]")
    exit()

for i in range(1, len(sys.argv)):
    if sys.argv[i] == '-f':
        if ".h5" in sys.argv[i + 1]:
            params[FILE] = PurePath(sys.argv[i + 1]).parts[-1].split(".h5")[0]
        else:
            params[FILE] = sys.argv[i + 1]
    if sys.argv[i] == '-i':
        params[INDEX] = int(sys.argv[i + 1], 10)
    if sys.argv[i] == '-c':
        params[CARLINI] = sys.argv[i+1]

if params[INDEX] == -1:
    params[INDEX] = r.randint(0, 999)

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
print("INDEX\t= {}".format(params[INDEX]))

data = data_types[params[DATASET]]
class_names = class_types[params[DATASET]]

_, (x_test, y_test) = data.load_data()

img = x_test[params[INDEX]]
label = y_test[params[INDEX]]

# normalizing
img = img / 255.0 - params[NORM]

# padding
img = p.pad(img, number_of_paddings=params[PADDING], padder=0.0 - params[NORM])

d = img.shape[1]
img = np.reshape(img, (1, d, d, 1))

# gray-box attack, so the attacker gets to know only the architecture of the model
# which is like giving him the unencrypted version
lab_model = MODEL_NAME.replace("PERMUTATED", "UNENCRYPTED")
lab_model = lab_model.replace("ECB", "UNENCRYPTED")
lab_model = lab_model.replace("CBC", "UNENCRYPTED")
lab_model = lab_model.replace("CTR", "UNENCRYPTED")

adv = a.attack(img, label, lab_model)

input_shape = np.array(img[0]).shape
model = models[params[MODEL]](input_shape)
model.load(lab_model)

prob_adv_attacked_model = model.predict(np.float32(adv))[0]  # the output is 2D array
prob_orig_attacked_model = model.predict(np.float32(img))[0] # likewise

pred_adv_attacked_model = np.argmax(prob_adv_attacked_model)
pred_orig_attacked_model = np.argmax(prob_orig_attacked_model)

helper = importlib.import_module("src.encryptions." + train_mode[params[TRAIN_WITH_ME]])
if params[TRAIN_WITH_ME] in ["ECB", "CBC", "CTR"]:
    helper.NORM = params[NORM]

input_shape = np.array(img[0]).shape
model = models[params[MODEL]](input_shape, encrypt=helper.encrypt)
model.load(MODEL_NAME)

prob_adv_safe_model = model.predict(np.float32(adv))[0]  # the output is 2D array
prob_orig_safe_model = model.predict(np.float32(img))[0] # likewise

pred_adv_safe_model = np.argmax(prob_adv_safe_model)
pred_orig_safe_model = np.argmax(prob_orig_safe_model)

# plot_original_adversarial(orig_img=img[0], adv_img=adv[0], orig_pred=pred_orig_safe_model, adv_pred=pred_adv_safe_model, orig_prob=prob_orig_safe_model,
#                           adv_prob=prob_adv_safe_model, true_label=label, noise=adv[0] - img[0])

plot_original_adversarial(orig_img=img[0], adv_img=adv[0], noise=adv[0] - img[0], orig_prob_safe=prob_orig_safe_model,
                          orig_prob_attacked=prob_orig_attacked_model, adv_prob_safe=prob_adv_safe_model,
                          adv_prob_attacked=prob_adv_attacked_model, true_label=label)

# adv_img = np.reshape(adv[i], (28,28))
# np.save("src/adversarial_images/adv1", adv_img)