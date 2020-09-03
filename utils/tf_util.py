""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2016

Upadted by Yue Wang and Yongbin Sun
"""

import numpy as np
import tensorflow as tf
#import lorentz
from math import *
from itertools import combinations
# from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
# import tensorflow.contrib.seq2seq as seq2seq

def _variable_on_cpu(name, shape, initializer, use_fp16=False, trainable=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn', is_dist=is_dist)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def conv2d_nobias(inputs,
                  num_output_channels,
                  kernel_size,
                  scope,
                  stride=[1, 1],
                  padding='SAME',
                  use_xavier=True,
                  stddev=1e-3,
                  weight_decay=0.0,
                  activation_fn=tf.nn.relu,
                  bn=False,
                  bn_decay=None,
                  is_training=None,
                  is_dist=False):
  """ 2D convolution with non-linear operation.
      Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable
      Returns:
        Variable tensor
      """


  with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn', is_dist=is_dist)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def get_neighbors(point_cloud, nn_idx, k=20):
    """Construct neighbors feature for each point
      Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int
      Returns:
        neighbors features: (batch_size, num_points, k, num_dims)
      """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    og_num_dims = point_cloud.get_shape().as_list()[-1]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)
    if og_num_dims == 1:
        point_cloud = tf.expand_dims(point_cloud, -1)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

    return point_cloud_neighbors

      
def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', is_dist=is_dist)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None,
                     is_dist=False):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', is_dist=is_dist)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

   

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn', is_dist=is_dist)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    is_dist=False):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
      num_input_units = inputs.get_shape()[-1].value
      weights = _variable_with_weight_decay('weights',
                                            shape=[num_input_units, num_outputs],
                                            use_xavier=use_xavier,
                                            stddev=stddev,
                                            wd=weight_decay)
      outputs = tf.matmul(inputs, weights)
      biases = _variable_on_cpu('biases', [num_outputs],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)
     
      if bn:
          outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', is_dist=is_dist)
          
      if activation_fn is not None:
          if activation_fn == tf.nn.softmax:         
              outputs = activation_fn(outputs-tf.reduce_max(outputs,axis=1, keep_dims=True))
          else:
              outputs = activation_fn(outputs)
      return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs





def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    #print('types again', inputs.dtype, mean.dtype, var.dtype, beta.dtype, gamma.dtype)
    #normed = tf.nn.batch_normalization(inputs, mean, var, tf.cast(beta,tf.float64), tf.cast(gamma,tf.float64), 1e-3)
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed


def batch_norm_dist_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ The batch normalization for distributed training.
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = _variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
    gamma = _variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

    pop_mean = _variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer(), trainable=False)
    pop_var = _variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer(), trainable=False)

    def train_bn_op():
      batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
      decay = bn_decay if bn_decay is not None else 0.9
      #decay = tf.cast(decay,tf.float64)
      #print('types:', pop_mean.dtype,  decay.dtype,batch_mean.dtype)
      train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)) 
      train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
      with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

    def test_bn_op():
      return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

    normed = tf.cond(is_training,
                     train_bn_op,
                     test_bn_op)
    return normed



def batch_norm_for_fc(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,1], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)



  
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,1,2], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)



def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,1,2,3], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs


def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
      point_cloud = tf.expand_dims(point_cloud, 0) # first dim is batch size

  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1]) 
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose) # x.x + y.y + z.z shape: NxN
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True) # from x.x, y.y, z.z to x.x + y.y + z.z
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def pairwise_distanceR(point_cloud):
  """Compute pairwise distance in the eta-phi plane for the point cloud.
  Uses the third dimension to find the zero-padded terms
  Args:
    point_cloud: tensor (batch_size, num_points, 2)
    IMPORTANT: The order should be (eta, phi) 
  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
      point_cloud = tf.expand_dims(point_cloud, 0) # first dim is batch size
  
  pt = point_cloud[:,:,2]
  pt = tf.expand_dims(pt,-1)
  is_zero = point_cloud[:,:,] 
  point_shift = 1000*tf.where(tf.equal(pt,0),tf.ones_like(pt),tf.fill(tf.shape(pt), tf.constant(0.0, dtype=pt.dtype)))
  point_shift_transpose = tf.transpose(point_shift,perm=[0, 2, 1])
  #pt = tf.exp(pt)
  point_cloud = point_cloud[:,:,:2] 
  
  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
  point_cloud_phi = point_cloud_transpose[:,1:,:]
  point_cloud_phi = tf.tile(point_cloud_phi,[1,point_cloud_phi.get_shape()[2].value,1])
  point_cloud_phi_transpose = tf.transpose(point_cloud_phi,perm=[0, 2, 1])
  point_cloud_phi = tf.abs(point_cloud_phi - point_cloud_phi_transpose)
  is_bigger2pi = tf.greater_equal(tf.abs(point_cloud_phi),2*np.pi)
  point_cloud_phi_corr = tf.where(is_bigger2pi,4*np.pi**2-4*np.pi*point_cloud_phi,point_cloud_phi-point_cloud_phi)
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose) # x.x + y.y + z.z shape: NxN
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True) # from x.x, y.y, z.z to x.x + y.y + z.z
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])

  #print("shape",point_cloud_square.shape)
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose+point_cloud_phi_corr +point_shift+point_shift_transpose
  #return point_shift



def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k) #values, indices
  return nn_idx


  

def get_edge_feature(point_cloud, nn_idx, k=20,edge_type='dgcnn'):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims) 
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, 2*num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)

  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  if edge_type == "dgcnn":
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  elif edge_type == "sub":
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    #4 vector difference with invariant mass sum
    edge_feature = Sub_Cloud(point_cloud_central, point_cloud_neighbors)
    edge_feature = tf.concat([point_cloud_central, edge_feature], axis=-1)
  elif edge_type == "add":
    #4 vector difference with invariant mass sum
    edge_feature = Add_Cloud(point_cloud_central, point_cloud_neighbors)
    edge_feature = tf.concat([point_cloud_central, edge_feature], axis=-1) 
  #edge_feature= tf.reduce_max(edge_feature, axis=-2, keep_dims=True)
  return edge_feature



def Sub_Cloud(central,neighbors):
    """ Input: BxPxKxF for central and K-neighbors
    Returns: BxPxKx(F+7), 7 = eta, phi, pt, px, py, pz differences + invariant mass sum """
    num_batch = central.get_shape()[0].value
    num_point = central.get_shape()[1].value
    num_k = central.get_shape()[2].value
    num_dims = central.get_shape()[3].value
    point_diff = central - neighbors
        
    identity = -np.identity(4,dtype=np.float32) #4vector
    identity[0][0] = 1
    identity = tf.expand_dims(identity, 0)
    identity = tf.expand_dims(identity, 0)
    identity = tf.expand_dims(identity, 0)
    identity = tf.tile(identity,[num_batch,num_point,num_k,1,1])
    
    sum_vec = neighbors + central
    sum_vec = tf.concat([point_diff[:,:,0:1,:],sum_vec[:,:,1:,:]],-2) # first neighbor is the point itself
    sum_vec = sum_vec[:,:,:,4:8]
    sum_vec = tf.expand_dims(sum_vec, -2)
    sum_vec_T = tf.transpose(sum_vec,perm=[0, 1,2, 4,3])
    mult = tf.matmul(sum_vec,identity)
    mult = tf.matmul(mult, sum_vec_T)
    mult = tf.sqrt(tf.abs(mult))
    #return tf.squeeze(mult,axis=-2)
    phi = point_diff[:,:,:,1:2] #Correct phi for 2pi bound
    is_bigger2pi = tf.greater_equal(tf.abs(phi),2*np.pi)
    phi_corr = tf.where(is_bigger2pi,phi - 2*np.pi,phi)
    
    diff_update = point_diff[:,:,:,0:1]
    diff_update = tf.concat([diff_update, phi_corr], axis=-1)
    diff_update = tf.concat([diff_update, point_diff[:,:,:,2:3]], axis=-1)
    diff_update = tf.concat([diff_update, tf.squeeze(mult,axis=-2)], axis=-1)
    diff_update = tf.concat([diff_update, point_diff[:,:,:,4:]], axis=-1)
    return diff_update
    


def Add_Cloud(central,neighbors):
    """ Input: BxPxKxF for central and K-neighbors
    Returns: BxPxKx(F+7), 5 = E, px, py, pz sum + invariant mass sum """
    num_batch = central.get_shape()[0].value
    num_point = central.get_shape()[1].value
    num_k = central.get_shape()[2].value
    num_dims = central.get_shape()[3].value

        
    identity = -np.identity(4,dtype=np.float32) #4vector
    identity[0][0] = 1
    identity = tf.expand_dims(identity, 0)
    identity = tf.expand_dims(identity, 0)
    identity = tf.expand_dims(identity, 0)
    identity = tf.tile(identity,[num_batch,num_point,num_k,1,1])
    
    sum_vec = neighbors + central
    sum_vec = tf.concat([point_diff[:,:,0:1,:],sum_vec[:,:,1:,:]],-2) # first neighbor is the point itself
    sum_vec = sum_vec[:,:,:,4:8]
    point_sum = sum_vec
    sum_vec = tf.expand_dims(sum_vec, -2)
    sum_vec_T = tf.transpose(sum_vec,perm=[0, 1,2, 4,3])
    mult = tf.matmul(sum_vec,identity)
    mult = tf.matmul(mult, sum_vec_T)
    mult = tf.sqrt(tf.abs(mult))
        
    diff_update = tf.squeeze(mult,axis=-2)
    diff_update = tf.concat([diff_update, point_sum], axis=-1)
    return diff_update

def Add_3VecCloud(central,neighbors):
    """ Will add the 4vectors for the sum of 3 4-vectors with the invariant mass sum """
    num_batch = central.get_shape()[0].value
    num_point = central.get_shape()[1].value
    num_k = central.get_shape()[2].value
    num_dims = central.get_shape()[3].value
    point_diff = central - neighbors
    
    identity = -np.identity(4,dtype=np.float32) #4vector
    identity[0][0] = 1
    identity = tf.expand_dims(identity, 0)
    identity = tf.expand_dims(identity, 0)
    identity = tf.expand_dims(identity, 0)
    identity = tf.tile(identity,[num_batch,num_point,num_k,1,1])
    
    sum_vec = neighbors + central
    sum_vec = tf.concat([point_diff[:,:,0:1,:],sum_vec[:,:,1:,:]],-2)
    sum_vec = sum_vec[:,:,:,4:8]
    point_sum = sum_vec
    sum_vec = tf.expand_dims(sum_vec, -2)
    sum_vec_T = tf.transpose(sum_vec,perm=[0, 1,2, 4,3])
    mult = tf.matmul(sum_vec,identity)
    mult = tf.matmul(mult, sum_vec_T)
    mult = tf.sqrt(tf.abs(mult))
    
    diff_update = tf.squeeze(mult,axis=-2)
    diff_update = tf.concat([diff_update, point_sum], axis=-1)
    return diff_update

  
# def seq2seq_with_attention(inputs,
#         hidden_size,
#         scope,
#         activation_fn=tf.nn.relu,
#         bn=False,
#         bn_decay=None,
#         is_training=None):
#     """ sequence model with attention.
#        Args:
#          inputs: 4-D tensor variable BxNxTxD
#          hidden_size: int
#          scope: encoder
#          activation_fn: function
#          bn: bool, whether to use batch norm
#          bn_decay: float or float tensor variable in [0,1]
#          is_training: bool Tensor variable
#        Return:
#          Variable Tensor BxNxD
#        """
#     with tf.variable_scope(scope) as sc:
#         batch_size = inputs.get_shape()[0].value
#         npoint = inputs.get_shape()[1].value
#         nstep = inputs.get_shape()[2].value
#         in_size = inputs.get_shape()[3].value
#         reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))

#         with tf.variable_scope('encoder'):
#             #build encoder
#             encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
#             encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, reshaped_inputs,
#                                                                sequence_length=tf.fill([batch_size*npoint], 4),
#                                                                dtype=tf.float64, time_major=False)
#         with tf.variable_scope('decoder'):
#             #build decoder
#             decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
#             decoder_inputs = tf.reshape(encoder_state.h, [batch_size*npoint, 1, hidden_size])

#             # building attention mechanism: default Bahdanau
#             # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
#             attention_mechanism = seq2seq.BahdanauAttention(num_units=hidden_size, memory=encoder_outputs)
#             # 'Luong' style attention: https://arxiv.org/abs/1508.04025
#             # attention_mechanism = seq2seq.LuongAttention(num_units=hidden_size, memory=encoder_outputs)

#             # AttentionWrapper wraps RNNCell with the attention_mechanism
#             decoder_cell = seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
#                                                               attention_layer_size=hidden_size)

#             # Helper to feed inputs for training: read inputs from dense ground truth vectors
#             train_helper = seq2seq.TrainingHelper(inputs=decoder_inputs, sequence_length=tf.fill([batch_size*npoint], 1),
#                                                   time_major=False)
#             decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size*npoint, dtype=tf.float64)
#             train_decoder = seq2seq.BasicDecoder(cell=decoder_cell, helper=train_helper, initial_state=decoder_initial_state, output_layer=None)
#             decoder_outputs_train, decoder_last_state_train, decoder_outputs_length_train = seq2seq.dynamic_decode(
#                 decoder=train_decoder, output_time_major=False, impute_finished=True)

#         outputs = tf.reshape(decoder_last_state_train[0].h, (-1, npoint, hidden_size))
#         if bn:
#           outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

#         if activation_fn is not None:
#           outputs = activation_fn(outputs)
#         return outputs


if __name__ == "__main__":
    import provider
    import numpy as np
    batch_size = 1
    num_pt = 3
    pos_dim = 5
    k = 2
    #wmass = ROOT.TH1D("wmass","wmass",100,0,6)
    #test_feed = np.random.rand(batch_size, num_pt, pos_dim)
    #pairwise_distanceR(pointclouds_pl)
    a =  np.array(
      [
        [
          [-5.0, 5,1,2,3],
          [ 0,0,0,0,0],
          [ 3, -3,10,11,12],

        ]
      ]
)

    #a, b, c = provider.load_h5_data_label_seg("../data/ttbb/h5/test_files_ttbar.h5")

    batch_size = a.shape[0]
    with tf.Graph().as_default():
      pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_pt, pos_dim))
      #nn_idx = tf.placeholder(tf.int32, shape=(batch_size, num_pt, k))
      pair = pairwise_distanceR(pointclouds_pl[:,:,:3])
      nn_idx = knn(pair, k=k)
      
      #edge = get_edge_feature(pointclouds_pl, nn_idx, k=k,edge_type='sub')
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
          pointclouds_pl: a,
          #nn_idx: idx
        }
        #edges = sess.run([pair], feed_dict=feed_dict)
        idxs,pairs = sess.run([nn_idx,pair], feed_dict=feed_dict)
        print (idxs, pairs)
        #for batch in edges:
        #  for point in batch:
        #    for mass in point:
        #       wmass.Fill(mass[1][0])

    #wmass.Draw()
    #raw_input()
    #pair = pairwise_distanceR(a)    
    #r = tf.greater_equal(tf.abs(a),2*np.pi)
    #re = tf.where(r,4*np.pi**2-4*np.pi*tf.abs(a),a-a)
    #print(a.get_shape())
    #print(a.get_shape(),'a')
    #at = tf.transpose(a,[0,2,1])
    #print(at.get_shape(),'at')
    #phi = at[:,1:,:]
    #print(phi.get_shape()[2].value,'phi')
    #phi5 = tf.tile(phi,[1,5,1])
    #b =  tf.constant([[1],[2]])
    #c = a + b
    #print(pair.eval())
    
