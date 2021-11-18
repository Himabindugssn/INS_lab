'''
The DNN architectures used in this project
'''


import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from pathlib import Path


class Encrypted_Model():
    def __init__(self, encrypt=lambda a:a):
        self.encrypt = encrypt

    def train(self, x, y_train, ep):
        x_train = x.copy()
        for i,image in enumerate(x_train):
            x_train[i] = self.encrypt(image)

        h = self.model.fit(x_train, y_train, epochs=ep)
        return h.history['loss'], h.history['acc'], list(range(ep))

    def compile(self):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


    def evaluate(self, x, y_test):
        x_test = x.copy()
        for i,image in enumerate(x_test):
            x_test[i] = self.encrypt(image)

        return self.model.evaluate(x_test, y_test)

    def predict(self, image):
        enc_img = self.encrypt(image)
        return self.model.predict(enc_img)

    def save(self, m_file):
        self.model.save_weights(filepath='saved_models/{}.h5'.format(m_file))

    def load(self, m_file):
        if Path('saved_models/{}.h5'.format(m_file)).is_file():
            self.model.load_weights(filepath='saved_models/{}.h5'.format(m_file))
        else:
            print("WARNING! No saved model found, train one from scratch.")
            self.model = None
        return self

    def load_old(self, m_file):
        if Path('saved_models/' + m_file).is_file():
            # self.model = tf.keras.models.load_model('saved_models/' + m_file)
            elf.model = keras.models.load_model('saved_models/' + m_file)
            self.compile()
        else:
            print("WARNING! No saved model found, train one from scratch.")
            exit()
        return self

    def summary(self):
        self.model.summary()


class modelA(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a:a):
        super(modelA, self).__init__(encrypt)

        self.model = Sequential()

        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, name="before-softmax"))
        self.model.add(tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


class modelA_no_softmax(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a: a):
        super(modelA_no_softmax, self).__init__(encrypt)

        self.num_channels = 1
        self.image_size = input_shape[0]
        self.num_labels = 10

        self.model = Sequential()

        self.model.add(
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, name="before-softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def predict(self, image):
        enc_img = self.encrypt(image)
        return self.model(enc_img)


class modelB(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a: a):
        super(modelB, self).__init__(encrypt)

        self.model = Sequential()

        self.model.add(
            tf.keras.layers.Conv2D(filters=64, kernel_size=8, input_shape=input_shape, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=6, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, name="before-softmax"))
        self.model.add(tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

