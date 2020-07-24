import tensorflow as tf
import tf_util
import numpy as np


def attn_feature(input_feature, output_dim, neighbors_idx, activation, in_dropout=0.0, coef_dropout=0.0, is_training=None, bn_decay=None, layer='', k=20, i=0, is_dist=False):
    batch_size = input_feature.get_shape()[0].value
    num_dim = input_feature.get_shape()[-1].value

    input_feature = tf.squeeze(input_feature)
    if batch_size == 1:
        input_feature = tf.expand_dims(input_feature, 0)

    input_feature = tf.expand_dims(input_feature, axis=-2)


    # if in_dropout != 0.0:
    #     input = tf.nn.dropout(input, 1.0 - in_dropout)

    new_feature = tf_util.conv2d_nobias(input_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                        is_training=is_training, scope=layer + '_newfea_conv_head_' + str(i),
                                        bn_decay=bn_decay, is_dist=is_dist)
    #Encode the high level features in a 1 layer CNN, the weights are learnable parameters of this filter

    neighbors = tf_util.get_neighbors(input_feature, nn_idx=neighbors_idx, k=k) #Group up the neighbors using the index passed on the arguments
    input_feature_tiled = tf.tile(input_feature, [1, 1, k, 1])
    edge_feature = input_feature_tiled - neighbors #Make the edge features yij
    #edge_feature = tf.concat([input_feature_tiled, input_feature_tiled-neighbors], axis=-1)
    edge_feature = tf_util.conv2d(edge_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1],
                               bn=True, is_training=is_training, scope=layer + '_edgefea_' + str(i), bn_decay=bn_decay, is_dist=is_dist)
    #Enconde that as well

    self_attention = tf_util.conv2d(new_feature, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                  is_training=is_training, scope=layer+'_self_att_conv_head_'+str(i), bn_decay=bn_decay, is_dist=is_dist)
    
    neibor_attention = tf_util.conv2d(edge_feature, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                    is_training=is_training, scope=layer+'_neib_att_conv_head_'+str(i), bn_decay=bn_decay, is_dist=is_dist)
    #To merge both contributions, pass them to a 1 layer, size 1 output

    logits = self_attention + neibor_attention
    logits = tf.transpose(logits, [0, 1, 3, 2])

    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
    # zero_tf =  tf.fill(tf.shape(coefs), tf.constant(0.0, dtype=coefs.dtype))
    # coefs = tf.where(tf.less_equal(coefs,0.1),zero_tf,coefs ) #Keep only att. > 0.1
    #coefs = tf.nn.softmax(tf.nn.relu(logits))
    # coefs = tf.ones_like(coefs)
    #
    # if coef_dropout != 0.0:
    #     coefs = tf.nn.dropout(coefs, 1.0 - coef_dropout)


    vals = tf.matmul(coefs, edge_feature)

    if is_dist:
        ret = activation(vals)
    else:
        ret = tf.contrib.layers.bias_add(vals)
        ret = activation(ret)


    return ret, self_attention, edge_feature
    #return ret, coefs, edge_feature
