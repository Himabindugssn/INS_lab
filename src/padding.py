

import numpy as np

'''
padding an image
'''
def pad(img, number_of_paddings, padder=0):
    if number_of_paddings == 0:
        return np.array(img)

    number_of_paddings = int(number_of_paddings / 2)

    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    return np.pad(img, number_of_paddings, pad_with, padder=padder)

# img = [[1,2],[3,4]]
# print(pad(img, 0, 9))
# print(pad(img, 12, 9))
