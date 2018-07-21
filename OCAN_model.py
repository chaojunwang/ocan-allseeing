#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:48:09 2018

@author: chaojunwang
"""

import tensorflow as tf
import numpy as np
# For loading data, ploting and verifing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_curve, auc
# Parameters and helper functions
from config import *
from utils import *


# PLACEHOLDERS
X_oc = tf.placeholder(tf.float32, shape=[None, dim_input]) 
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])  
X_tar = tf.placeholder(tf.float32, shape=[None, dim_input]) 
keep_prob = tf.placeholder(tf.float32)


# WEIGHTS,BIASES AND CACHE
# discriminator
D_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
D_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]])) 
D_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
D_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))
D_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
D_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3] 

# generator
G_W1 = tf.Variable(xavier_init([G_dim[0], G_dim[1]]))
G_b1 = tf.Variable(tf.zeros(shape=[G_dim[1]]))
G_W2 = tf.Variable(xavier_init([G_dim[1], G_dim[2]]))
G_b2 = tf.Variable(tf.zeros(shape=[G_dim[2]]))
theta_G = [G_W1, G_W2, G_b1, G_b2]

# pre-train net(discriminator_T) for density estimation(distribution of real data)
T_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
T_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))
T_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
T_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))
T_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
T_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))
theta_T = [T_W1, T_W2, T_W3, T_b1, T_b2, T_b3]


# BUILD NETWORKS
def generator(z):
    """
    Args: noise Z
    Return: generated data
    """
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h1_drop = tf.nn.dropout(G_h1, keep_prob)
    G_logit = tf.nn.tanh(tf.matmul(G_h1_drop, G_W2) + G_b2)
    return G_logit
def discriminator(x):
    """
    Args: real data
    Return: softmax output of classifier
    """
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h1_drop = tf.nn.dropout(D_h1, keep_prob)
    D_h2 = tf.nn.relu(tf.matmul(D_h1_drop, D_W2) + D_b2)
    D_h2_drop = tf.nn.dropout(D_h2, keep_prob)
    D_logit = tf.matmul(D_h2_drop, D_W3) + D_b3
    D_prob = tf.nn.softmax(D_logit)
    return D_prob, D_logit, D_h2
def discriminator_tar(x):
    """
    Args: real data
    Return: softmax output of pre-train estimator
    """
    T_h1 = tf.nn.relu(tf.matmul(x, T_W1) + T_b1)
    T_h2 = tf.nn.relu(tf.matmul(T_h1, T_W2) + T_b2)
    T_logit = tf.matmul(T_h2, T_W3) + T_b3
    T_prob = tf.nn.softmax(T_logit)
    return T_prob, T_logit, T_h2
# G -> D
D_prob_real, D_logit_real, D_h2_real = discriminator(X_oc) 
G_sample = generator(Z)
D_prob_gen, D_logit_gen, D_h2_gen = discriminator(G_sample)
# T
D_prob_tar, D_logit_tar, D_h2_tar = discriminator_tar(X_tar) 
D_prob_tar_gen, D_logit_tar_gen, D_h2_tar_gen = discriminator_tar(G_sample) 


# LOSS FUNCTIONS
# loss of discriminator_D
y_real = tf.placeholder(tf.int32, shape=[None, D_dim[3]])  
y_gen = tf.placeholder(tf.int32, shape=[None, D_dim[3]])
D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_real, labels=y_real))
D_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_gen, labels=y_gen))
ent_real_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(D_prob_real, tf.log(D_prob_real)), 1))
ent_gen_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(D_prob_gen, tf.log(D_prob_gen)), 1))
D_loss = D_loss_real + D_loss_gen + 1.65*ent_real_loss

# loss of discriminator_T
y_tar = tf.placeholder(tf.int32, shape=[None, D_dim[3]])
T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_tar, labels=y_tar))

# loss of generator
pt_loss = pull_away_loss(D_h2_tar_gen) # PT_term
tar_thrld = tf.divide(tf.reduce_max(D_prob_tar_gen[:,-1]) + tf.reduce_min(D_prob_tar_gen[:,-1]), 2)
indicator = tf.sign(tf.subtract(D_prob_tar_gen[:,-1], tar_thrld))  
condition = tf.greater(tf.zeros_like(indicator), indicator) 
mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator) 
G_ent_loss = tf.reduce_mean(tf.multiply(tf.log(D_prob_tar_gen[:,-1]), mask_tar))
fm_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(D_logit_real - D_logit_gen), 1)))
G_loss = pt_loss + G_ent_loss + fm_loss


# OPTIMIZERS -> SOLVERS
D_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
T_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=theta_T)


def load_data(train_data_path, test_ben_path, test_van_path):
    """load encoded data
    Args:
        train_data_path: train data(only benign)
        test_ben_path: test data(benign)
        test_van_path: test data(vandal)
    Return:
        x_pre,y_pre: train data and label for T
        x_train,y_real_mb,y_fake_mb: train data and label for ocan
        x_test,y_test: test data and label
    """
    # load data
    x_benign = np.load(train_data_path) 
    x_test_benign = np.load(test_ben_path) 
    x_test_vandal = np.load(test_van_path) 
    # scale
    min_max_scaler = MinMaxScaler()
    x_benign = min_max_scaler.fit_transform(x_benign) 
    x_test_benign = min_max_scaler.transform(x_test_benign)
    x_test_vandal = min_max_scaler.transform(x_test_vandal)
    
    x_benign = sample_shuffle_uspv(x_benign)[:10000]
    x_test_benign = sample_shuffle_uspv(x_test_benign)[:10000]
    x_test_vandal = sample_shuffle_uspv(x_test_vandal)
    
    x_pre = x_benign
    y_pre = np.zeros(len(x_pre)) 
    y_pre = one_hot(y_pre, 2) 
    
    x_train = x_pre
    y_real_mb = one_hot(np.zeros(mb_size), 2)  
    y_fake_mb = one_hot(np.ones(mb_size), 2)
    
    x_test = x_test_benign[:].tolist() + x_test_vandal[:].tolist()
    x_test = np.array(x_test)
    y_test = np.zeros(len(x_test))
    y_test[10000:] = 1 # benign -- 0    vandal -- 1
    return x_pre,y_pre,x_train,y_real_mb,y_fake_mb,x_test,y_test

def train(dra_tra_pro):
    """
    Args: 
        dra_tra_pro: plot trend during training or not
    Return:
        prob: softmax output
        y_pred: predictions of model
        conf_mat:
        d_ben_pro,d_fake_pro,d_val_pro,fm_loss_coll,f1_score: for ploting probs during training
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # pre-train 
    _ = sess.run(T_solver, feed_dict={X_tar:x_pre, y_tar:y_pre})
   
    if dra_tra_pro:
        d_ben_pro, d_fake_pro, fm_loss_coll = list(), list(), list()
        f1_score, d_val_pro= list(), list()
    
    q = int(len(x_train) / mb_size)
    for n_epoch in range(n_round):
        X_mb_oc = sample_shuffle_uspv(x_train)
        for n_batch in range(q):
            # fix G，train D
            _, D_loss_curr, ent_real_curr = sess.run([D_solver, D_loss, ent_real_loss],
                                                     feed_dict={
                                                         X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size], 
                                                         Z: sample_Z(mb_size, Z_dim), 
                                                         y_real: y_real_mb, 
                                                         y_gen: y_fake_mb, 
                                                         keep_prob:1.0
                                                     })
            # fix D，train G
            _, G_loss_curr, fm_loss_curr = sess.run([G_solver, G_loss, fm_loss], 
                                                    feed_dict={
                                                        Z: sample_Z(mb_size, Z_dim),
                                                        X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size], 
                                                        keep_prob:1.0
                                                        })
        if dra_tra_pro:
            D_prob_real_, D_prob_gen_ = sess.run([D_prob_real, D_prob_gen],
                                                 feed_dict={
                                                     X_oc: x_train,
                                                     Z: sample_Z(len(x_train), Z_dim),
                                                     keep_prob:1.0
                                                 })
            D_prob_vandal_ = sess.run(D_prob_real,
                                          feed_dict={
                                              X_oc:x_test[-1000:],
                                              keep_prob:1.0
                                          })
            d_ben_pro.append(np.mean(D_prob_real_[:, 0]))
            d_fake_pro.append(np.mean(D_prob_gen_[:, 0]))
            d_val_pro.append(np.mean(D_prob_vandal_[:, 0]))
            fm_loss_coll.append(fm_loss_curr)
    
        prob, _ = sess.run([D_prob_real, D_logit_real], feed_dict={X_oc: x_test, keep_prob:1.0})
        y_pred = np.argmax(prob, axis=1)
        conf_mat = classification_report(y_test, y_pred, target_names=['benign', 'vandal'], digits=4)
        if dra_tra_pro:
            f1_score.append(float(list(filter(None, conf_mat.strip().split(" ")))[12]))
        
    return prob,y_pred,conf_mat,d_ben_pro, d_fake_pro, d_val_pro, fm_loss_coll, f1_score

def roc_curve_auc(y_test,prob):
    """
    Args:
        prob: softmax output
    Return:    
        y_prediction: probilities that model predicts test data IS VANDAL(value is '1')
        false_positive_rate: FP/(FP+TN)
        true_positive_rate: TP/(TP+FN)
        thresholds: list of threshold
    """
    y_prediction = []
    for i in prob:
        y_prediction.append(i[1])
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prediction, pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return y_prediction, false_positive_rate, true_positive_rate, thresholds

def threshold(y_prediction, true_positive_rate, thresholds):
    """
    Args:
        y_prediction, true_positive_rate, thresholds
    Returns:
        threshold:if a sample's probility is larger than this scalar,model predicts it's a vandal
    """
    for i in range(len(true_positive_rate)):
        if true_positive_rate[i]>0.6:
            index = i
            break
    threshold = thresholds[index]
    print('Threshold is ', threshold)
    result = (y_prediction[-2271:] > threshold) * 1
    for amount in [8,12,20,100,500,2271]:
        print('OCAN model finds %d vandals out of first %d vandals.'%(np.sum(result[:amount],axis=0),amount))
    return threshold

# MAIN
if __name__ == '__main__':
    x_pre,y_pre,x_train,y_real_mb,y_fake_mb,x_test,y_test = load_data(train_data_path, 
                                                                      test_ben_path, 
                                                                      test_van_path)
    prob,y_pred,conf_mat,d_ben_pro,d_fake_pro,d_val_pro,fm_loss_coll,f1_score = train(dra_tra_pro)     
    acc = np.sum(y_pred == y_test)/float(len(y_pred))
    print(conf_mat)
    print("acc:%s"%acc)
    if dra_tra_pro:
        draw_trend(d_ben_pro, d_fake_pro, d_val_pro, fm_loss_coll, f1_score)
    y_prediction, false_positive_rate, true_positive_rate, thresholds = roc_curve_auc(y_test,prob)
    threshold = threshold(y_prediction, 
                          true_positive_rate, 
                          thresholds)
    








