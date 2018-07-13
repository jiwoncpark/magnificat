# Thanks to Burak Himmetoglu at
# https://github.com/healthDataScience/deep-learning-HAR
# for providing the framework

from __future__ import print_function
import numpy as np
import pandas as pd
import gc
import os, sys
#from sklearn.model_selection import train_test_split
data_path = os.path.join(os.environ['DEEPQSODIR'], 'data')
sys.path.insert(0, data_path)
from data_utils import *
import tensorflow as tf
import time
from sklearn.model_selection import KFold

features_path = os.path.join(data_path, 'features.npy')
label_path = os.path.join(data_path, 'labels.npy')

DEBUG = True

# Training hyperparameters
BATCH_SIZE = 500
LEARNING_RATE = 1.e-2
NUM_EPOCHS = 10
KEEP_PROB = 1.0
NUM_CLASSES = 2

#start_data = time.time()

#X = np.loadtxt(features_path, delimiter=',').reshape(NUM_OBJECTS, NUM_TIMES, NUM_CHANNELS)
#y = np.loadtxt(label_path, delimiter=',').astype(int)
X = np.load(features_path)
y = np.load(label_path).reshape(-1).astype(int)

#X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, stratify=y, random_state=123)
kf = KFold(n_splits=5, shuffle=True, random_state=123)

y = to_onehot(y, num_classes=NUM_CLASSES)
print(y.shape)
#end_data = time.time()

#print("Finished reading in data... in %0.2f seconds" %(end_data - start_data))

graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, NUM_TIMES, NUM_CHANNELS], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

print("Setting up network graph...")
with graph.as_default():
    # (batch, 738, 18) --> (batch, 738, 18)
    conv0 = tf.layers.conv1d(inputs=inputs_, filters=NUM_CHANNELS, kernel_size=1, strides=1,
                             padding='same', activation = tf.nn.relu)
    if DEBUG: print("After conv0: ", conv0.shape)

    # (batch, 738, 18) --> (batch, 369, 36)
    conv00 = tf.layers.conv1d(inputs=conv0, filters=NUM_CHANNELS*2, kernel_size=1, strides=1,
                              padding='same', activation = tf.nn.relu)
    max_pool_00 = tf.layers.max_pooling1d(inputs=conv00, pool_size=2, strides=2, padding='same')

    if DEBUG: print("After conv00, max_pool00: ", max_pool_00.shape)

    # (batch, 369, 36) --> (batch, 123, 108)
    conv1 = tf.layers.conv1d(inputs=max_pool_00, filters=NUM_CHANNELS*2*3, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=3, padding='same')

    if DEBUG: print("After conv1, max_pool_1: ", max_pool_1.shape)
    
    # (batch, 123, 108) --> (batch, 41, 324)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=NUM_CHANNELS*2*3*3, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=3, padding='same')

    if DEBUG: print("After conv2, max_pool_2: ", max_pool_2.shape)
    
with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_2, (-1, 41*NUM_CHANNELS*2*3*3))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
    # Predictions
    logits = tf.layers.dense(flat, NUM_CLASSES)
    
    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
if (os.path.exists('checkpoints-cnn') == False):
    os.makedirs('checkpoints-cnn')

with graph.as_default():
    saver = tf.train.Saver()

train_acc, train_loss = [], []
print("Began training:")
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    # Loop over epochs
    for e in range(NUM_EPOCHS):
        cv_valacc, cv_valloss = [], []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]            
            if (e % 50 == 0) and (e != 0):
                LEARNING_RATE /= (1.0 + e*0.0005)
                
            # Loop over batches
            for x_t, y_t in fetch_batches(X_train, y_train, batch_size=BATCH_SIZE):
                # Feed dictionary
                feed = {inputs_ : x_t,
                        labels_ : y_t, 
                        keep_prob_ : KEEP_PROB, 
                        learning_rate_ : LEARNING_RATE}
                # Loss
                loss, _ , acc = sess.run([cost, optimizer, accuracy], 
                                         feed_dict = feed)
                train_acc.append(acc)
                train_loss.append(loss)

                # Print at each 10 iters
                if (iteration % 10 == 0):
                    print("Epoch: {}/{}".format(e, NUM_EPOCHS),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc))
                iteration += 1

            # Compute validation loss at the end of every CV epoch
            val_acc_ = []
            val_loss_ = []

            for x_v, y_v in fetch_batches(X_val, y_val, BATCH_SIZE):
                feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                val_acc_.append(acc_v)
                val_loss_.append(loss_v)

            print("Epoch: {}/{}".format(e, NUM_EPOCHS),
                  "Learning rate: {:.6f}".format(LEARNING_RATE),
                  "Iteration: {:d}".format(iteration),
                  "Validation loss: {:6f}".format(np.mean(val_loss_)),
                  "Validation acc: {:.6f}".format(np.mean(val_acc_)))
            # Store
            cv_valacc.append(np.mean(val_acc_))
            cv_valloss.append(np.mean(val_loss_))
        print('CV mean accuracy: {:.6}, CV std: {:.6}'.format(np.mean(cv_valacc), np.std(cv_valacc)))
        print('CV mean loss: {:.6}'.format(np.mean(cv_valloss)))


    saver.save(sess,"checkpoints-cnn/catalog.ckpt")
