# Thanks to Burak Himmetoglu at
# https://github.com/healthDataScience/deep-learning-HAR
# for providing the framework

from __future__ import print_function
import numpy as np
import pandas as pd
import gc
import os, sys
from sklearn.model_selection import train_test_split
data_path = os.path.join(os.environ['DEEPQSODIR'], 'data')
sys.path.insert(0, data_path)
from data_utils import *
import tensorflow as tf
import time

features_path = os.path.join(data_path, 'features.csv')
label_path = os.path.join(data_path, 'labels.csv')

#NUM_POSITIVES = 15631
NUM_POSITIVES = 156
NUM_TIMES = 60
NUM_ATTRIBUTES = 13
NUM_FILTERS = 5
NUM_CHANNELS = NUM_ATTRIBUTES + NUM_FILTERS

# Training hyperparameters
BATCH_SIZE = 500
LEARNING_RATE = 1.e-2
NUM_EPOCHS = 1200
KEEP_PROB = 1.0
NUM_CLASSES = 2

start_data = time.time()
X = np.loadtxt(features_path, delimiter=',').reshape(NUM_POSITIVES*2, NUM_TIMES, NUM_CHANNELS)
y = np.loadtxt(label_path, delimiter=',').astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, stratify=y, random_state=123)

y_train = to_onehot(y_train, num_classes=NUM_CLASSES)
y_val = to_onehot(y_val, num_classes=NUM_CLASSES)
end_data = time.time()

NUM_TRAIN = X_train.shape[0]
NUM_VAL = X_val.shape[0]

print("Finished reading in data... in %0.2f seconds" %(end_data - start_data))

graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, NUM_TIMES, NUM_CHANNELS], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

print("Setting up network graph...")
with graph.as_default():
    # (batch, 64, 65) --> (batch, 64, 65)
    conv0 = tf.layers.conv1d(inputs=inputs_, filters=65, kernel_size=1, strides=1,
                             padding='same', activation = tf.nn.relu)
    #max_pool_0 = tf.layers.max_pooling1d(inputs=conv0, pool_size=2, strides=2, padding='same')

    conv00 = tf.layers.conv1d(inputs=conv0, filters=65*2, kernel_size=1, strides=1,
                              padding='same', activation = tf.nn.relu)

    # (batch, 64, #65 130) --> (batch, 32, #130 260)
    conv1 = tf.layers.conv1d(inputs=conv00, filters=65*4, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    
    # (batch, 32, 130) --> (batch, 16, 260)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=65*8, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    
    # (batch, 16, 260) --> (batch, 8, 520)
    #conv3 = tf.layers.conv1d(inputs=conv2, filters=65*16, kernel_size=2, strides=1, 
    #                         padding='same', activation = tf.nn.relu)
    #max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
    
    # (batch, 8, 520) --> (batch, 4, 1040)
    #conv4 = tf.layers.conv1d(inputs=conv3, filters=960, kernel_size=2, strides=2, 
    #                         padding='same', activation = tf.nn.relu)
    #max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

    # (batch, 8, 520) --> (batch, 4, 1040)
    #conv5 = tf.layers.conv1d(inputs=conv4, filters=1920, kernel_size=2, strides=2, 
    #                         padding='same', activation = tf.nn.relu)
    #max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')

with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_2, (-1, 15*65*8))
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

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

print("Began training:")
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
   
    # Loop over epochs
    for e in range(NUM_EPOCHS):
        
        p = np.random.permutation(NUM_TRAIN)
        X_train = X_train[p, :, :]
        y_train = y_train[p, ]
        batches = fetch_batches(X_train, y_train, batch_size=BATCH_SIZE)

        #print("learning rate: %0.7f" %LEARNING_RATE)

        if (e % 50 == 0) and (e != 0):
            LEARNING_RATE /= (1.0 + e*0.0005)
        
        # Loop over batches
        for x, y in batches:
            
            # Feed dictionary
            feed = {inputs_ : x,
                    labels_ : y, 
                    keep_prob_ : KEEP_PROB, 
                    learning_rate_ : LEARNING_RATE}
            
            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], 
                                     feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 10 == 0):
                print("Epoch: {}/{}".format(e, NUM_EPOCHS),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration % 100 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in fetch_batches(X_val, y_val, 300):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, NUM_EPOCHS),
                      "Learning rate: {:.6f}".format(LEARNING_RATE),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            
            # Iterate 
            iteration += 1
    
    saver.save(sess,"checkpoints-cnn/catalog.ckpt")
