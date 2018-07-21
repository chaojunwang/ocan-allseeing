#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:56:06 2018

@author: chaojunwang
"""

import numpy as np 
import tensorflow as tf 
from tf.keras.layers import Input, Dense
from tf.keras.models import Model, Sequential
import os
# Parameters
from config import *

# Categorical and numerical features
train_cat = np.load('processed_np/train_cat_sparse.npy').tolist() # one-hot categorical features
train_cont = np.load('processed_np/train_cont.npy') # numerical features
test_ben_cat = np.load('processed_np/test_ben_cat_sparse.npy').tolist()
test_ben_cont = np.load('processed_np/test_ben_cont.npy')
test_van_cat = np.load('processed_np/test_van_cat_sparse.npy').tolist()
test_van_cont = np.load('processed_np/test_van_cont.npy')

train_cat = train_cat[:LEN_TRAIN] 
train_cont = train_cont[:LEN_TRAIN]


def create_autoencoder(input_dim, encoding_dim):
    """
    Args:
        input_dim: dimension of one-hot encoded categorical features
        encoding_dim: dimension of encoded data(hidden layer representation)
    Return: 
        model
    """
    one_hot_in = Input(shape=(input_dim,), name='input', sparse=True)
    X = Dense(HIDDEN_UNITS, activation='selu')(one_hot_in)
    encoding = Dense(encoding_dim, activation='selu', name='enco')(X)
    X = Dense(HIDDEN_UNITS, activation='selu')(encoding)
    output = Dense(input_dim, activation='sigmoid')(X)
    
    model = Model(inputs=one_hot_in, outputs=output)
    return model

def main(encoding_dim):
    """ Train autoencoder and get encoded categorical features
    Args:
        encoding_dim
    """
    autoencoder = create_autoencoder(TOTAL_DIM, encoding_dim)
    autoencoder.compile(optimizer='adagrad',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    # Train autoencoder
    autoencoder.fit(train_cat, train_cat, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Get encoded categorical features
    encoder = Sequential()
    for i in range(3):
        encoder.add(autoencoder.layers[i])
    train_out = encoder.predict(train_cat)
    test_ben_out = encoder.predict(test_ben_cat)
    test_van_out = encoder.predict(test_van_cat)

    # Concatnate encoded categorical features and numerical features
    res_train = np.concatenate((train_out, train_cont), axis=1)
    res_test_ben = np.concatenate((test_ben_out, test_ben_cont), axis=1)
    res_test_van = np.concatenate((test_van_out, test_van_cont), axis=1)

    dir_path = f'encoded_data/dim_{encoding_dim}'

    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    np.save(dir_path + f'/encoded_train_{encoding_dim}.npy', res_train)
    np.save(dir_path + f'/encoded_test_benign_{encoding_dim}.npy', res_test_ben)
    np.save(dir_path + f'/encoded_test_vandal_{encoding_dim}.npy', res_test_van)

if __name__ == '__main__':
    dims = [10]
    for d in dims:
        main(d)

