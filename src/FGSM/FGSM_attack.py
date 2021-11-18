

import tensorflow as tf
import numpy as np
import src.Models as m
from src.FGSM.cleverhans.cleverhans.attacks.fast_gradient_method import FastGradientMethod
from src.FGSM.cleverhans.cleverhans.utils_keras import KerasModelWrapper

'''
performing an FGSM attack on a set of images
'''
def attack(img, label, model_name, evaluate=False):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        input_shape = np.array(img[0]).shape
        model = m.modelB(input_shape)
        model.load(model_name)

        wrap = KerasModelWrapper(model)
        fgsm = FastGradientMethod(wrap, sess=sess)
        fgsm_params = {'eps': 0.1,
                       'clip_min': 0,
                       'clip_max': 1}

        adv = fgsm.generate_np(img, **fgsm_params)

        if evaluate:
            _, test_acc = model.evaluate(adv, label)
            print("accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))

        return adv
