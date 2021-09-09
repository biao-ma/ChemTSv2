import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import GRU, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import Callback, ModelCheckpoint

import tensorflow as tf
from tqdm import tqdm
import yaml
import h5py
import json
import time
import datetime
import os
import glob

SNAPSHOTS_PATH = 'snapshots'
class RNN(object):
    def __init__(self, use_gpu, batch_size, validation_split, vocabulary,
                 is_LSTM=False, is_biLSTM=False):
        print("make model")
        self.use_gpu = use_gpu
        self._set_gpu()

        self.batch_size = batch_size
        self.validation_split = validation_split

        self.vocabulary = vocabulary

        self.is_LSTM = is_LSTM
        self.is_biLSTM = is_biLSTM

    def _set_gpu(self,):
        # limiting gpu memory
        cpu_devices = tf.config.experimental.list_physical_devices("CPU")
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(gpu_devices) > 0:
            for k in range(len(gpu_devices)):
                if self.use_gpu:
                    tf.config.experimental.set_memory_growth(
                        gpu_devices[k], True)
                    print("memory growth:",tf.config.experimental.get_memory_growth(gpu_devices[k]))
            if bool(self.use_gpu) is False:
                tf.config.set_visible_devices([], "GPU")
                tf.config.experimental.set_visible_devices(cpu_devices, "CPU")
        else:
            print("Not enough GPU hardware devices available")
    
    def make_model(self, vocab_size, embed_size, N, hidden_size, dropout_rate, lr_val):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size,
                                 output_dim=embed_size,
                                 input_length=N,
                                 mask_zero=False))

        # encoder
        if self.is_LSTM:
            self.model.add(LSTM(output_dim=hidden_size, activation="tanh",return_sequences=True))
        elif self.is_biLSTM:
            self.model.add(Bidirectional(LSTM(output_dim=hidden_size,activation="tanh",return_sequences=True)))
        else:
            self.model.add(GRU(output_dim=hidden_size,activation="tanh",return_sequences=True))
        self.model.add(Dropout(dropout_rate))

        # decoder
        if self.is_LSTM:
            self.model.add(LSTM(hidden_size, activation="tanh", return_sequences=True))
            self.model.add(Dropout(dropout_rate))
            self.model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
        elif self.is_biLSTM:
            self.model.add(Bidirectional(LSTM(hidden_size, activation="tanh", return_sequences=True)))
            self.model.add(Dropout(dropout_rate))
            self.model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
        else:
            self.model.add(GRU(hidden_size, activation="tanh", return_sequences=True))
            self.model.add(Dropout(dropout_rate))
            self.model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))

        print(self.model.summary())
        optimizer = Adam(lr=lr_val)
        self.model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])

    def train_model(self,train,valid, epochs, stat):
        call = MyCallback(stat)
        os.makedirs(SNAPSHOTS_PATH, exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(SNAPSHOTS_PATH, 'model_{epoch:02d}_{val_loss:.2f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1)

        #self.model.fit_generator(train,
        return self.model.fit_generator(train,
                  epochs = epochs,
                  steps_per_epoch = len(train),
                  validation_data = valid,
                  validation_steps = len(valid),
                  shuffle=True,
                  callbacks = [call, model_checkpoint]
                  )

    def save_model(self, output_json, output_weight):
        print("Now saving")
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(output_json, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(output_weight)

        # save charset table
        with h5py.File(output_weight, "a") as h5_file:
            h5_file.create_dataset("charset",
                                   data=np.string_(self.vocabulary),
                                   dtype=h5py.special_dtype(vlen=str))

        weight_files = glob.glob(SNAPSHOTS_PATH + '/*.h5')
        print(weight_files)
        for weight_file in weight_files:
            with h5py.File(weight_file, "a") as h5_file:
                h5_file.create_dataset("charset",
                                       data=np.string_(self.vocabulary),
                                       dtype=h5py.special_dtype(vlen=str))
        print("Saved model to disk")



class MyCallback(Callback):

    def __init__(self, stat):
        self.stat = stat

    def on_epoch_begin(self, epoch, logs=None):
        print("Start Epoch {}".format(epoch))
        self.stat.write("Start Epoch {}".format(epoch))

    def on_train_end(self, logs=None):
        print("Train End")
        self.stat.write("Train End")

