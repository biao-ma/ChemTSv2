import sys
import numpy as np
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.optimizers import Adam
from make_smile import zinc_processed_with_bracket, zinc_data_with_bracket_smi
from train_RNN import prepare_data, num_per_smiles
from SmilesEnumerator import SmilesEnumerator
from data_generator import DataGenerator
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tqdm import tqdm
import yaml
import json
import h5py
import argparse

from util import StatFile
from model import RNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', help='config yaml file', default='fine_tuningtrain_RNN.yaml')
    parser.add_argument('-s', '--status', help='status file', default='status.txt')
    args = parser.parse_args()

    print('conf:', args.conf)
    print('status', args.status)
    with open(args.conf, "r") as f:
        conf = yaml.safe_load(f)

    # for status.txt
    stat = StatFile(args.status)

    dataset = conf.get("dataset")
    base_model_json = conf.get("base_json")
    base_model_weight = conf.get("base_weight")
    output_json = conf.get("output_json")
    output_weight = conf.get("output_weight")
    lr_val = conf.get("lr", 0.01)
    epochs = conf.get("epoch", 100)
    batch_size = conf.get("batch_size", 512)
    validation_split = conf.get("validation_split", 0.1)
    enumerated_smiles_num = conf.get("enumerated_smiles_num", 1)
    enumerated_smiles_time = conf.get("enumerated_smiles_time", 3)
    use_gpu = conf.get("use_gpu", 0)

    print("========== display configuration ==========")
    print("dataset = ", dataset)
    print("base_model_json = ", base_model_json)
    print("base_model_weight = ", base_model_weight)
    print("output_json = ", output_json)
    print("output_weight = ", output_weight)
    print("learning_rate = ", lr_val)
    print("epoch = ", epochs)
    print("batch_size = ", batch_size)
    print("validation_split = ", validation_split)
    print("use_gpu = ", use_gpu)

    # limiting gpu memory
    cpu_devices = tf.config.experimental.list_physical_devices("CPU")
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(gpu_devices) > 0:
        for k in range(len(gpu_devices)):
            if use_gpu:
                tf.config.experimental.set_memory_growth(
                    gpu_devices[k], True)
                print("memory growth:",tf.config.experimental.get_memory_growth(gpu_devices[k]))
        if bool(use_gpu) is False:
            tf.config.set_visible_devices([], "GPU")
            tf.config.experimental.set_visible_devices(cpu_devices, "CPU")
    else:
        print("Not enough GPU hardware devices available")

    # load base_model
    with open("/chemts/RNN-model/" + base_model_json, "r") as f:
        loaded_base_model_json = f.read()
    base_model = model_from_json(loaded_base_model_json)
    base_model.load_weights("/chemts/RNN-model/" + base_model_weight)

    # load vocabulary
    with h5py.File("/chemts/RNN-model/" + base_model_weight, "r") as f:
        print("Now loading base model")
        base_charset = list(f["charset"][()])
    base_charset = [i.decode("utf-8") if type(i) is bytes else i for i in base_charset]

    # load added smiles
    smiles = zinc_data_with_bracket_smi("/work/" + dataset)

    # specify enumerate smiles per compound
    enumerated_smiles = num_per_smiles(smiles, enumerated_smiles_num,
                                       enumerated_smiles_time)

    vocabulary, all_smile = zinc_processed_with_bracket(enumerated_smiles)

    # vocaburaly that does not duplicate smiles
    vocabulary = sorted(set(base_charset + vocabulary),
                         key=(base_charset + vocabulary).index)

    # print remove items
    # when vocabraly generated from added smiles is include base charaset
    # and remove smiles from data
    remove_items = set(vocabulary) - set(base_charset)
    if len(remove_items) > 0:
        print("charset does not match")
        print("use matching vocabularies")
        print("remove items {}".format(remove_items))
        for smi in np.array(all_smile):
            if len(set(smi) & remove_items) > 0:
                all_smile.remove(smi)
                print("removed \"{}\"".format(''.join(smi).rstrip('\n')))
        vocabulary = base_charset
    else:
        print("No remove items")

    print("vocabulary = {0} {1}".format(str(vocabulary), type(vocabulary)))
    X_train, y_train = prepare_data(vocabulary, all_smile)

    base_conf = base_model.get_config()
    base_conf = [i for i in base_conf["layers"]
                 if i["class_name"] == "Embedding"][0]["config"]
    maxlen = base_conf.get("input_length", 82)

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

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_split)

    train = DataGenerator(x_train, y_train, vocab_size, batch_size)
    valid = DataGenerator(x_valid, y_valid, vocab_size, batch_size)
    rnn = RNN(use_gpu, batch_size, validation_split, vocabulary)

    model = base_model

    optimizer = Adam(lr=lr_val)
    print(model.summary())

    model.compile(loss="categorical_crossentropy",
                        optimizer=optimizer,
                        metrics=["accuracy"])

    rnn.model = model
    rnn.train_model(train, valid, epochs, stat)
    rnn.save_model(output_json, output_weight)
