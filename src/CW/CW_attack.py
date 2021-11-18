
import tensorflow as tf
import numpy as np
import src.Models as m

# default value
from src.CW.l2_attack import CarliniL2 as c


def set_mode(mode):
    global c
    if mode == '2':
        from src.CW.l2_attack import CarliniL2 as c
    if mode == '0':
        from src.CW.l0_attack import CarliniL0 as c
    if mode == 'i':
        from src.CW.li_attack import CarliniLi as c

'''
performing a CW attack on a set of images
'''
def attack(img, label, model_name, evaluate=False):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        input_shape = np.array(img[0]).shape
        model = m.modelA_no_softmax(input_shape)
        model.load(model_name)

        attack = c(sess=sess, model=model, targeted=False, max_iterations=1000)

        target = np.eye(10)[np.array(label).reshape(-1)]

        adv = attack.attack(img, target)

        if evaluate:
            _, test_acc = model.evaluate(adv, label)
            print("accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))

        return adv
