# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:16:51 2019

@author: ALLEN
"""
    
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import copy
import time
import os

'''
data: collections of videos[type:list(dict)]
seq_len: 300(frames/video)
batch_size: how many videos per epoch
'''
def batch_generator(data, batch_size, seq_len):
    data = copy.copy(data)
    num_batches = int(len(data) / batch_size)
    data = data[:batch_size * num_batches]
    while True:
        np.random.shuffle(data)
        for l in range(0, len(data), batch_size):
            X_list = []
            y_list = []
            for i in range(batch_size):
               x = data[l+i]['video_matrix']
               X_list.append(x)
               y = tf.broadcast_to(data[l+i]['labels'], shape=[300, 3862])
               y_list.append(tf.cast(y, dtype=tf.float32))
            X = tf.concat(X_list, axis=0)
            Y = tf.reshape(tf.stack(y_list), shape=[-1, 3862])
            yield X, Y


class Model():
    '''
       rnn_size is a list where each element is hidden_size of a layer
    '''
    def __init__(self, batch_size, seq_len, output_size, inp_size = 1024, rnn_size=[128, 128], num_layers=2, training=True, use_dropout=True, lr=0.001, dropout_rate=0.5):
        self.batch_size =  batch_size
        self.seq_len = seq_len
        self.inp_size = inp_size
        if training == False:
            self.batch_size = 1
            self.seq_len = 300
        
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.training = training
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.lr = lr
       
        tf.reset_default_graph()
        self.X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_len, self.inp_size])
        self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size * self.seq_len, self.output_size])
        self.predict, self.loss = self.model(self.X, self.y)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        
    def rnn_cell(self, layer_num):
        return tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size[layer_num])
    
    def multi_rnn(self):
        cell = []
        for layer_i in range(self.num_layers):
            if self.training and self.use_dropout:
                cell.append(tf.contrib.rnn.DropoutWrapper(self.rnn_cell(layer_i), output_keep_prob=1.0 - self.dropout_rate))
                continue
            cell.append(self.rnn_cell(layer_i))
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
        return multi_rnn_cell
    
    '''
       inp: (batch_size, seq_len, inp_size)
       @return: (batch_size, seq_len, hidden_size)
    '''
    def Recurrent_layers(self, inp):
        multi_rnn_cell = self.multi_rnn()
        self.initial_state = multi_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(multi_rnn_cell, inp, initial_state=self.initial_state)
        return outputs, state
    
    '''
       @return: (batch_size * seq_len, output_size) x 2
    '''
    def Output_layers(self, outputs, hidden_size):
        flatten_out = tf.reshape(outputs, shape=[-1, hidden_size]) # shape: batch_size * seq_len, hidden_size
        softmax_W = tf.get_variable(name="out_W", shape=[hidden_size, self.output_size], dtype=tf.float32, initializer=tf.initializers.variance_scaling)
        softmax_b = tf.get_variable(name="out_b", shape=[self.output_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        logits = tf.matmul(flatten_out, softmax_W) + softmax_b
        prob = tf.nn.softmax(logits)
        return logits, prob
    
    def compute_loss(self, logits, y):
        mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
        return mean_loss
    
    def model(self, X, y):
        rnn_out, self.final_state = self.Recurrent_layers(X)
        logits, self.prob = self.Output_layers(rnn_out, self.rnn_size[-1]) # logits(batch_size * seq_len, output_size)
        loss = self.compute_loss(logits, y)
        pred = tf.reshape(tf.argmax(logits, axis=1), [self.batch_size, self.seq_len])
        return pred, loss
    
    def train(self,batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        with self.session as sess:
            sess.run(init)
            step = 0    
            new_state = sess.run(self.initial_state)
            for X_batch, y_batch in batch_generator:
                 step += 1
                 start = time.time()
                 new_state, loss_val, _ = sess.run([self.final_state, self.loss, self.train_op], feed_dict={self.X: X_batch, self.y: y_batch, self.initial_state: new_state})
                 end = time.time()
                 if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(loss_val),
                          '{:.4f} sec/batch'.format((end - start)))
                 if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                 if step >= max_steps:
                    break
        self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
        
    def predict(self, inp):
         sess = self.session
         new_state = sess.run(self.initial_state)
         pred = sess.run(self.prob, feed_dict={self.X: inp, self.initial_state: new_state})
         return np.argsort(pred, axis=-1)[-5:]
     
    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))