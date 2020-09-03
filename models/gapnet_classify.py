import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../nn_distance/tf_ops/nn_distance/'))
import tf_util
# import tf_nndistance #Only for EMD and CD

from gat_layers import attn_feature


def placeholder_inputs(batch_size, num_point, num_features,num_glob):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    global_pl = tf.placeholder(tf.float32, shape=(batch_size,num_glob)) 
    return pointclouds_pl,  labels_pl, global_pl


def gap_block(k,n_heads,nn_idx,net,point_cloud,edge_size,bn_decay,weight_decay,is_training,scname):
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(net, edge_size[1], nn_idx, activation=tf.nn.relu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer{0}'.format(edge_size[0])+scname, k=k, i=i)
        attns.append(edge_feature)# This is the edge feature * att. coeff. activated by RELU, one per particle
        local_features.append(locals) #Those are the yij


    neighbors_features = tf.concat(attns, axis=-1)
    net = tf.squeeze(net)
    #neighbors_features = tf.concat([tf.expand_dims(net, -2), neighbors_features], axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), neighbors_features], axis=-1)

    locals_transform = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)

    return neighbors_features, locals_transform, coefs


def get_model(point_cloud, is_training, num_class,params, global_pl,
                weight_decay=None, bn_decay=None,scname=''):
    ''' input: BxNxF
    Use https://arxiv.org/pdf/1902.08570 as baseline
    output:BxNx(cats*segms)  '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    num_feat = point_cloud.get_shape()[2].value


  
    k = params[0]
    adj = tf_util.pairwise_distanceR(point_cloud[:,:,:3])
    n_heads = params[1]
    nn_idx = tf_util.knn(adj, k=k)

    
    net, locals_transform, coefs= gap_block(k,n_heads,nn_idx,point_cloud,point_cloud,('filter0',params[2]),bn_decay,weight_decay,is_training,scname)


    net = tf_util.conv2d(net, params[3], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet01'+scname, bn_decay=bn_decay)
    net01 = net


    net = tf_util.conv2d(net, params[4], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet02'+scname, bn_decay=bn_decay)

    net02 = net
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)    
    adj_conv = nn_idx
    n_heads = params[5]

    net, locals_transform1, coefs2= gap_block(k,n_heads,nn_idx,net,point_cloud,('filter1',params[6]),bn_decay,weight_decay,is_training,scname)

    net = tf_util.conv2d(net, params[7], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet11'+scname, bn_decay=bn_decay)
    net11 = net



    net = tf_util.conv2d(net, params[8], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet12'+scname, bn_decay=bn_decay)

    net12= net
    

    # global_expand = tf.reshape(global_pl, [batch_size, 1, 1, -1])
    # global_expand = tf.tile(global_expand, [1, num_point, 1, 1])
    # global_expand = tf_util.conv2d(global_expand, 16, [1, 1],
    #                                padding='VALID', stride=[1, 1],
    #                                bn=True, is_training=is_training,
    #                                scope='global_expand'+scname, bn_decay=bn_decay)


    net = tf.concat([
        net01,
        net02,
        net11,
        net12,
        #global_expand,
        locals_transform,
        locals_transform1
    ], axis=-1)


    net = tf_util.conv2d(net, params[9], [1, 1], padding='VALID', stride=[1, 1], 
                         activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='agg'+scname, bn_decay=bn_decay)
    
    net = tf_util.avg_pool2d(net, [num_point, 1], padding='VALID', scope='avgpool'+scname)
    max_pool = net
    
    
    net = tf.reshape(net, [batch_size, -1]) 
    net = tf_util.fully_connected(net, params[10], bn=True, is_training=is_training, activation_fn=tf.nn.relu,
                                 scope='fc1'+scname, bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training,
                          scope='dp1'+scname)
    net = tf_util.fully_connected(net, params[11], bn=True, is_training=is_training, activation_fn=tf.nn.relu,
                                  scope='fc2'+scname, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_class,activation_fn=None, scope='fc3'+scname)


        
    net = tf.squeeze(net)
    

    return net,max_pool


            
def get_focal_loss(y_pred,label,num_class,gamma=4., alpha=10):
    gamma = float(gamma)
    alpha = float(alpha)
    epsilon = 1.e-9

    labels = tf.one_hot(indices=label, depth=num_class)
    y_true = tf.convert_to_tensor(labels, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    
    y_pred = tf.nn.softmax(y_pred-tf.reduce_max(y_pred,axis=1, keep_dims=True))
    
    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)

def get_loss_kmeans(max_pool,mu,  max_dim,n_clusters,alpha=100):
    # batch_size = max_pool.get_shape()[0].value
    
    list_dist = []
    #list_dist_centers = []
    for i in range(0, n_clusters):
        dist = f_func(tf.squeeze(max_pool), tf.reshape(mu[i, :], (1, max_dim)))
        #dist_centers = f_func(tf.squeeze(mu), tf.reshape(mu[i, :], (1, max_dim)))
        list_dist.append(dist)
        #list_dist_centers.append(dist_centers)
    stack_dist = tf.stack(list_dist)
    min_dist = tf.reduce_min(list_dist, axis=0)            
    #stack_centers = tf.stack(list_dist_centers)
    #sum_dist = tf.reduce_sum(stack_centers)
    #reg = tf.exp(-sum_dist)


    list_exp = []            
    for i in range(n_clusters):
        exp = tf.exp(-alpha*(stack_dist[i] - min_dist))
        list_exp.append(exp)
    
    stack_exp = tf.stack(list_exp)
    sum_exponentials = tf.reduce_sum(stack_exp, axis=0)
    
    list_weighted_dist = []
    for j in range(n_clusters):
        softmax = stack_exp[j] / sum_exponentials
        weighted_dist = stack_dist[j] * softmax
        
        list_weighted_dist.append(weighted_dist)

    stack_weighted_dist = tf.stack(list_weighted_dist)
    kmeans_loss = tf.reduce_mean(tf.reduce_sum(stack_weighted_dist, axis=0)) 
    #+ 10*reg


    return kmeans_loss, stack_dist
    



def get_loss(pred, label,num_class):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=num_class)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred)
  #return loss

  classify_loss = tf.reduce_mean(loss)
  return classify_loss





def f_func(x, y,k=256):
    dists = tf.square(x - y)
    return tf.reduce_sum(dists, axis=1)
    dists = tf.reduce_sum(dists, axis=1)
    #dists_transpose = tf.reshape(dists,(1,dists.get_shape()[0].value))
    neg_dist = -dists
    #print(neg_dist.get_shape())
    top_k,_ = tf.nn.top_k(neg_dist, k=k) #values, indices
    
    return tf.reshape(top_k, (k,1))
    



