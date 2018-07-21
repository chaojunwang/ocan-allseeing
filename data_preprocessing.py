#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:35:24 2018

@author: chaojunwang
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import gc


# Train data
train_data = pd.read_csv(
        'processed_train_no_resampling.csv', 
        usecols=['ip','app','device','os','channel','day','hour','minute','second','is_attributed'])

# Test data
test_data = pd.read_csv(
        'test_with_labels.csv', 
        usecols=['ip','app','device','os','channel','day','hour','minute','second','is_attributed'])

train_benigns = train_data.copy()
train_benigns = train_benigns.loc[train_benigns['is_attributed'] == 0] # Only use benigns to train
len1 = len(train_benigns)

test_benigns = test_data.copy()
test_vandals = test_data.copy()
test_benigns = test_benigns.loc[test_benigns['is_attributed'] == 0] 
test_vandals = test_vandals.loc[test_vandals['is_attributed'] == 1] 
len2 = len(test_benigns)
len3 = len(test_vandals)

# Save numerical features
try:
    os.mkdir('processed_np')
except FileExistsError:
    pass
train_cont = np.array(train_benigns[['day', 'hour','minute','second']])
np.save('processed_np/train_cont.npy', train_cont) 

test_ben_cont = np.array(test_benigns[['day', 'hour','minute','second']])
np.save('processed_np/test_ben_cont.npy', test_ben_cont)

test_van_cont = np.array(test_vandals[['day', 'hour','minute','second']])
np.save('processed_np/test_van_cont.npy', test_van_cont)


data = train_benigns.append(test_benigns).append(test_vandals)
del train_benigns,test_benigns,test_vandals; gc.collect()


# Drop labels
data.drop(['day','hour','minute','second','is_attributed'], axis=1, inplace=True)


# To categorical
data = data.apply(LabelEncoder().fit_transform)

train_benigns = data[:len1]
test_benigns = data[len1:len1+len2]
test_vandals = data[-len3:]


# One-hot encoding
enc = OneHotEncoder()
train_benigns = enc.fit_transform(train_benigns)
test_benigns = enc.transform(test_benigns)
test_vandals = enc.transform(test_vandals)

np.save('processed_np/train_cat_sparse.npy', np.array(train_benigns))
np.save('processed_np/test_ben_cat_sparse.npy', np.array(test_benigns))
np.save('processed_np/test_van_cat_sparse.npy', np.arrary(test_vandals))






