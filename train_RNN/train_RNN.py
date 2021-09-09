import numpy as np
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from make_smile import zinc_data_with_bracket_smi, zinc_processed_predefined_vocabulary # zinc_processed_with_bracket
from SmilesEnumerator import SmilesEnumerator
from data_generator import DataGenerator
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import yaml
import h5py
import json
import time
import datetime
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import StatFile

from model import RNN

def prepare_data(smiles, all_smile):
    all_smile_index = []
    for i in range(len(all_smile)):
        smile_index = []
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train = all_smile_index
    y_train = []
    for i in range(len(X_train)):

        x1 = X_train[i]
        x2 = x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)

    return X_train, y_train


def num_per_smiles(smiles, enumerated_smiles_num, enumerated_smiles_time):
    """
    Enumerate specified number of smiles

    Parameters
    ----------
    smiles : list[str]
        smiles list
    enumerated_smile_num : int
        number of specified smiles to enumerate
    enumerated_smiles_time : int
        

    Returns
    -------
    enumerated_smiles : list[str]
        enumerated smiles
    """
    print("Now enumerating smiles")
    enumerated_smiles_num = enumerated_smiles_num
    sme = SmilesEnumerator()
    enumerated_smiles = []
    for smi in tqdm(smiles):
        check_unique_smile = []
        check_unique_smile.append(smi)
        append_count = 1
        s_time = time.time()
        while append_count < enumerated_smiles_num:
            enumrated_smi = sme.randomize_smiles(smi)
            if enumrated_smi not in np.unique(check_unique_smile):
                check_unique_smile.append(enumrated_smi)
                append_count += 1
            if time.time() > s_time + enumerated_smiles_time:
                print(f"{smi} can't enumerate the specified number")
                break
        enumerated_smiles.extend(check_unique_smile)
    print("Finish enumerating smiles")

    return enumerated_smiles

import pandas as pd
import matplotlib.pyplot as plt
def write_history(history):
    df = pd.DataFrame(history.history)
    print(df)
    df.to_csv('history.csv')
    #plt.figure()
    df[['loss', 'val_loss']].plot()
    plt.savefig('history_loss.png')
    df[['accuracy', 'val_accuracy']].plot()
    plt.savefig('history_acc.png')

    
if __name__ == "__main__":
    #argvs = sys.argv
    #argc = len(argvs)
    #if argc == 1:
    #    print("input configuration file")
    #    exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', help='config yaml file', default='train_RNN.yaml')
    parser.add_argument('-s', '--status', help='status file', default='status.txt')
    args = parser.parse_args()

    print('conf:', args.conf)
    print('status', args.status)

    f = open(args.conf, "r+")
    conf = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    dataset = conf.get("dataset")
    output_json = conf.get("output_json")
    output_weight = conf.get("output_weight")
    maxlen = conf.get("smiles_max_length", 82)
    dropout_rate = conf.get("dropout_rate", 0.2)
    lr_val = conf.get("lr", 0.01)
    epochs = conf.get("epoch", 100)
    batch_size = conf.get("batch_size", 512)
    validation_split = conf.get("validation_split", 0.1)
    hidden_size = conf.get("hidden_size", 256)
    use_gpu = conf.get("use_gpu", 0)

    # RNN network
    is_LSTM = conf.get("is_LSTM", False)
    is_biLSTM = conf.get("is_biLSTM", False)

    print("========== display configuration ==========")
    print("dataset = ", dataset)
    print("output_json = ", output_json)
    print("output_weight = ", output_weight)
    print("dropout_rate = ", dropout_rate)
    print("learning_rate = ", lr_val)
    print("epoch = ", epochs)
    print("batch_size = ", batch_size)
    print("validation_split = ", validation_split)
    print("smiles_max_length = ", maxlen)
    print("use_gpu = ", use_gpu)

    # for status.txt
    stat = StatFile(args.status)

    smile = zinc_data_with_bracket_smi("/work/" + dataset)
    # specify enumerate smiles per compound
    enumerated_smiles_num = conf.get("enumerated_smiles_num", 1)
    enumerated_smiles_time = conf.get("enumerated_smiles_time", 3)
    enumerated_smiles = num_per_smiles(smile, enumerated_smiles_num,
                                       enumerated_smiles_time)

    # vocabulary, all_smile = zinc_processed_with_bracket(enumerated_smiles)
    vocabulary, all_smile = zinc_processed_predefined_vocabulary(enumerated_smiles)

    print("vocabulary = {0} {1}".format(str(vocabulary), type(vocabulary)))
    X_train, y_train = prepare_data(vocabulary, all_smile)

    X = sequence.pad_sequences(X_train,
                               maxlen=maxlen,
                               dtype="int32",
                               padding="post",
                               truncating="pre",
                               value=0.)
    y = sequence.pad_sequences(y_train,
                               maxlen=maxlen,
                               dtype="int32",
                               padding="post",
                               truncating="pre",
                               value=0.)

    vocab_size = len(vocabulary)
    embed_size = len(vocabulary)

    N = X.shape[1]

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_split)
 
    train = DataGenerator(x_train, y_train, vocab_size, batch_size)
    valid = DataGenerator(x_valid, y_valid, vocab_size, batch_size)
    #train = DataGenerator(X_train, Y_train, vocab_size, 1)
    #valid = DataGenerator(X_valid, Y_valid, vocab_size, 1)

    rnn = RNN(use_gpu, batch_size, validation_split, vocabulary,
              is_LSTM=is_LSTM , is_biLSTM=is_biLSTM)
    rnn.make_model(vocab_size, embed_size, N, hidden_size, dropout_rate, lr_val)
    #rnn.train_model(train, valid, epochs)
    history = rnn.train_model(train, valid, epochs, stat)
    rnn.save_model(output_json, output_weight)

    write_history(history)

