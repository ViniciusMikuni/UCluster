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
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size,num_point))
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
    #coefs = tf.reduce_sum(neighbors_features,axis=-1)
    #print("shapes: ",point_cloud.shape, net.shape)
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
    #adj = tf_util.pairwise_distance(point_cloud[:,:,:3])
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
    

    global_expand = tf.reshape(global_pl, [batch_size, 1, 1, -1])
    global_expand = tf.tile(global_expand, [1, num_point, 1, 1])
    global_expand = tf_util.conv2d(global_expand, 16, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=True, is_training=is_training,
                                   scope='global_expand'+scname, bn_decay=bn_decay)


    net = tf.concat([
        net01,
        net02,
        net11,
        net12,
        global_expand,
        locals_transform,
        locals_transform1
    ], axis=-1)


    net = tf_util.conv2d(net, params[9], [1, 1], padding='VALID', stride=[1, 1], 
                         activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='agg'+scname, bn_decay=bn_decay)
    
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='avgpool'+scname)
    max_pool = net
    expand = tf.tile(net, [1, num_point, 1, 1])
    net = tf.concat(axis=3, values=[expand, 
                                    net01,
                                    net11,
                                ])
    net = tf_util.conv2d(net, params[10], [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net = tf_util.conv2d(net, params[11], [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)
    #tf.nn.softmax
    net = tf_util.conv2d(net, num_class, [1,1], padding='VALID', stride=[1,1], activation_fn=tf.nn.softmax, 
                         bn=False, scope='seg/conv5', weight_decay=weight_decay, is_dist=True)

    net = tf.reshape(net, [batch_size, num_point, num_class])
    

    return net,max_pool


def SinkHorn(stack_dist,min_dist,r,epsilon=1e-2,n=4000):
    list_exp = []
    n, m = stack_dist.shape
    for i in range(n):
        exp = tf.exp(-(stack_dist[i]-min_dist)/epsilon)
        list_exp.append(exp)
    stack_exp = tf.stack(list_exp)
    
    P = stack_exp
    P /=tf.reduce_sum(P)
    

    c = tf.ones(m)/int(m)

    
    u = tf.zeros(n)
    
    for j in range(n):
        u = tf.reduce_sum(P,axis=1)
        P *= tf.reshape((r / u),(-1, 1))
        P *= tf.reshape((c / tf.reduce_sum(P,axis=0)),(1, -1))

    return P, tf.reduce_sum(P * stack_dist)
    
    


    # stack_exp_transpose = tf.transpose(stack_exp,perm=[1,0])#cik
    # b = tf.ones([n_clusters,1],dtype=tf.float32)
    # ones = tf.ones([batch_size,1],dtype=tf.float32)
    # fracs = tf.expand_dims(fracs,axis=-1)

    # for j in range(n):
    #     a = ones/(batch_size*1.0*tf.matmul(stack_exp_transpose,b))#size: batch_size x 1        
    #     b = tf.div(fracs,(tf.matmul(stack_exp,a))) #size: k x 1
        
    
    # pi = tf.multiply(a,stack_exp_transpose)
    # pi = tf.multiply(pi,tf.transpose(b,[1,0]))

    # return  tf.transpose(pi,[1,0])
            
    

def get_loss_kmeans(max_pool,mu, max_dim,n_clusters,alpha=100):
    # batch_size = max_pool.get_shape()[0].value
    
    list_dist = []
    for i in range(0, n_clusters):
        dist = f_func(tf.squeeze(max_pool), tf.reshape(mu[i, :], (1, max_dim)))
        list_dist.append(dist)
    stack_dist = tf.stack(list_dist)
    min_dist = tf.reduce_min(list_dist, axis=0)            


    # pi, kmeans_loss = SinkHorn(stack_dist,min_dist,fracs,epsilon=epsilon)


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
    #kmeans_loss = tf.reduce_sum(stack_weighted_dist, axis=0)


    return kmeans_loss, stack_dist
    
    

  
# def soft_assignment(embeddings, cluster_centers):
#     """Implemented a soft assignment as the  probability of assigning sample i to cluster j.        
#     Args:
#     embeddings: (num_points, dim)
#     cluster_centers: (num_cluster, dim)
    
#     Return:
#     q_i_j: (num_points, num_cluster)
#     """
    
#     batch_size = embeddings.get_shape()[0].value
#     embeddings = tf.squeeze(embeddings)
#     n_clusters = cluster_centers.get_shape()[0].value
#     def _pairwise_euclidean_distance(a,b):
#         p1 = tf.matmul(
#             tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
#             tf.ones(shape=(1, n_clusters))
#         )
#         p2 = tf.transpose(tf.matmul(
#             tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
#             tf.ones(shape=(batch_size, 1)),
#             transpose_b=True
#         ))
        
        
#         res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
#         return res

#     dist= _pairwise_euclidean_distance(embeddings, cluster_centers)
#     q = 1.0/(1.0+dist**2)
#     q = (q/tf.reduce_sum(q, axis=1, keepdims=True))
#     return q

# def target_distribution(q):
#     p = tf.square(q)/tf.reduce_sum(q,axis=0)
#     p = p / tf.reduce_sum(p,axis=1, keepdims=True)
#     return p

# def target_distribution(q):
#     p = q**2/np.sum(q,axis=0)
#     p = p / np.sum(p,axis=1, keepdims=True)
#     return p

# def get_kl(target, pred):
#     return tf.reduce_mean(tf.reduce_sum(target*tf.log(target/(pred)), axis=1))

def get_focal_loss(label,y_pred,num_class,gamma=2., alpha=10):
    gamma = float(gamma)
    alpha = float(alpha)
    epsilon = 1.e-9

    labels = tf.one_hot(indices=label, depth=num_class)
    y_true = tf.convert_to_tensor(labels, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    
    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)
    


def get_loss(pred, label,num_class):
  """ pred: B*NUM_CLASSES,
      label: B, """
  loss_per_part = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  per_instance_seg_loss = tf.reduce_mean(loss_per_part, axis=1)
  seg_loss = tf.reduce_mean(per_instance_seg_loss)
  return seg_loss
  labels = tf.one_hot(indices=label, depth=num_class)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred)
  
  classify_loss = tf.reduce_mean(loss)
  return classify_loss



# def get_loss_AE(pred, label):
#     dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
#     loss = tf.reduce_mean(dists_forward+dists_backward)
#     return loss*100


def f_func(x, y,k=20):
    dists = tf.square(x - y)
    # neg_adj = -dists
    # top_k,_ = tf.nn.top_k(neg_adj, k=k) #values, indices
    # return tf.reduce_sum(-top_k, axis=1)
    return tf.reduce_sum(dists, axis=1)


# def get_loss_MSE(pred,label):
#     loss = tf.compat.v1.losses.mean_squared_error(label, pred)
#     loss = tf.reduce_mean(loss)
#     return loss



if __name__=='__main__':
    dists_arr = np.array(
        [[1,2,3,8],
         [3,1,5,2],
         [5,6,1,7],]
    )

    batch_size = 4
    clusters = 3
    fracs = np.array([0.33,0.33,0.33])
    epsilon = 1e-2

    with tf.Graph().as_default():
        dists = tf.placeholder(tf.float32, shape=(clusters,batch_size))
        min_dist = tf.reduce_min(dists, axis=0)
        pi, kmeans_loss,exp_dist = SinkHorn(dists,min_dist,fracs,epsilon=epsilon,n=200)
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {dists:dists_arr}
            pi_arr,loss,mindist = sess.run([pi,kmeans_loss,min_dist],feed_dict=feed_dict) 
            pi_arr = np.array(pi_arr)
            print(loss,)
            #print(dists)
