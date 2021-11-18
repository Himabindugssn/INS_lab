
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

seed = 42

'''
random permutation on the image
'''
def encrypt(inputs):
    # performs numpy permutation on inputs
    dims = np.array(inputs).shape

    permutated_flattened = np.random.RandomState(seed=seed).permutation(inputs.flatten())
    enc_inputs = np.reshape(permutated_flattened, dims)

    return enc_inputs

# permutation_indexes = np.random.RandomState(seed=seed).permutation(list(range(784)))
# reverse = tfp.bijectors.Permute(permutation=permutation_indexes)
#
#
# def tf_encrypt(inputs):
#     # performs numpy permutation with tensorflow tool
#
#     shape = tf.shape(inputs)
#
#     enc_inputs = reverse.forward(tf.reshape(inputs, [-1]))
#     dec_inputs = reverse.inverse(enc_inputs)
#
#     enc_inputs = tf.reshape(enc_inputs, shape)
#     # dec_inputs = tf.reshape(dec_inputs, shape)
#
#     return enc_inputs


def print_encryption_details(out):
    out.write("key: {}\n".format(seed))
