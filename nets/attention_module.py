from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def se_block(input_feature, name, ratio=8):
  """Contains the implementation of Squeeze-and-Excitation block.
  As described in https://arxiv.org/abs/1709.01507.
  """

  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  bias_initializer = tf.constant_initializer(value=0.0)

  with tf.variable_scope(name):
    channel = input_feature.get_shape()[-1]
    # Global average pooling
    squeeze = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)    
    excitation = tf.layers.dense(inputs=squeeze,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='bottleneck_fc')    
    excitation = tf.layers.dense(inputs=excitation,
                                 units=channel,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='recover_fc')    
    scale = input_feature * excitation    
  return scale