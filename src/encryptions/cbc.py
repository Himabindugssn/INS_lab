

import json
import os
from Crypto.Cipher import AES
import numpy as np

key = b'&\xcc\xa4\xaa\x88\xbc\xad\xcf\x0f\xe9`\xe1w]\x1eo'
IV = b'\x96\xb8\xcf\xf6\x89\x9f0\x13H\xbd\xf59Y\x15\x947'

NORM = 0


def encrypt(inputs):
    return flattening((inputs + NORM) * 255.0)


'''
flattens the image, encrypts and reshapes
'''
def flattening(inputs):
    dims = np.array(inputs).shape

    aes_cipher = AES.new(key, AES.MODE_CBC,  IV=IV)

    flattened = bytes(map(int, inputs.flatten().tolist()))
    aes_flattened = list(aes_cipher.encrypt(flattened))
    enc_inputs = np.reshape(aes_flattened, dims)

    return (enc_inputs / 255.0) - NORM


'''
flattens according to blocks, encrypts and reshapes
'''
def blocking(inputs):
    dims = np.array(inputs).shape

    aes_cipher = AES.new(key, AES.MODE_CBC, IV=IV)

    np_image = np.array(inputs)
    blocked_image = blockshaped(np_image, 4, 4)
    flattened = bytes(map(int,[item for block in blocked_image for item in block.flatten().tolist()]))
    aes_flattened = list(aes_cipher.encrypt(flattened))
    enc_inputs = np.reshape(aes_flattened, dims)

    return (enc_inputs / 255.0) - NORM


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w, d = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def print_encryption_details(out):
    out.write("key: {}\tIV: {}\n".format(key, IV))
