# -*- coding: utf-8 -*-

import keras
from keras.utils.data_utils import Sequence
from keras.utils.np_utils import to_categorical
import math

class DataGenerator(Sequence):
    def __init__(self, X, Y, vocab_size, batch_size):
        self.X = X
        self.Y = Y
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.length = len(X)

    def __getitem__(self,idx):
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        Y = self.Y[start_pos : end_pos]
        y_train_one_hot = to_categorical(Y, num_classes=self.vocab_size)
        X = self.X[start_pos : end_pos]
        return (X, y_train_one_hot)

    def __len__(self):
        return math.ceil(len(self.X)/self.batch_size)

    def on_epoch_end(self):
        """Task when end of epoch"""
        pass

