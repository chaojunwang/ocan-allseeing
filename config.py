#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:49:14 2018

@author: chaojunwang
"""

# Parameters for autoencoder
BATCH_SIZE = 256 # autoencoder training mini_batchsize
EPOCHS = 2 # autoencoder training epochs
TOTAL_DIM = 70474 
# dimension of categorical features after one-hot encoding
# need to verify on a new dataset
HIDDEN_UNITS = 512 # hidden layer units of autoencoder 
LEN_TRAIN = 400000 # how many samples we use to train


# Parameters for ocan
mb_size = 128 # ocan training mini_batchsize
dim_input = 14 
# real data(user representation) dimension
# = encoding dim + numerical features num  
dra_tra_pro = True # plot or not
n_round = 20 # ocan training epochs

D_dim = [dim_input, 200, 100, 2] # discriminator structure
G_dim = [100, 200, dim_input] # generator structure
Z_dim = G_dim[0]  # noise dimension (input of generator) 

# encoded data path 
train_data_path = 'dim_10/encoded_train_10.npy' # train data(only benign)
test_ben_path = 'dim_10/encoded_test_benign_10.npy' # test data(benign)
test_van_path = 'dim_10/encoded_test_vandal_10.npy' # test data(vandal)