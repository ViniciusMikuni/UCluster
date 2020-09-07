import argparse
import math
import subprocess
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os,ast
import sys
from sklearn.cluster import KMeans
import h5py
#np.set_printoptions(edgeitems=1000)

from scipy.optimize import linear_sum_assignment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
import provider
import gapnet_seg as MODEL

parser = argparse.ArgumentParser()

parser.add_argument('--max_dim', type=int, default=2, help='Dimension of the encoding layer [Default: 512]')
parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters [Default: 2]')
parser.add_argument('--nglob', type=int, default=4, help='Number of global features [default: 2]')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='gapnet_seg', help='Model name [default: dgcnn]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=100, help='Point Number [default: 500]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size during training [default: 1024]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.01]')

parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=1000000, help='Decay step for lr decay [default: 1000000]')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--data_dir', default='../h5', help='directory with data [default: hdf5_data]')
parser.add_argument('--nfeat', type=int, default=7, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=21, help='Number of categories [default: 2]')
parser.add_argument('--nbatches', type=int, default=-1, help='Max number of batches to use for training [default: -1]')
parser.add_argument('--box', type=int, default=1, help='Black Box number, ignored if RD dataset [default: 1]')
parser.add_argument('--min', default='loss', help='Condition for early stopping loss or acc [default: loss]')
parser.add_argument('--RD',  default=False, action='store_true',help='Use RD data set [default: False')



FLAGS = parser.parse_args()
H5_DIR = FLAGS.data_dir
RD = FLAGS.RD
EPOCH_CNT = 0
MAX_PRETRAIN = 10
BATCH_SIZE = FLAGS.batch_size
NUM_GLOB = FLAGS.nglob
NUM_POINT = FLAGS.num_point
NUM_FEAT = FLAGS.nfeat
NUM_CLASSES = FLAGS.ncat
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


MODEL_FILE = os.path.join(BASE_DIR, '../models', FLAGS.model+'.py')
LOG_DIR = os.path.join('..','logs',FLAGS.log_dir)

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_kmeans_seg.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_dkm.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LEARNING_RATE_CLIP = 1e-5
HOSTNAME = socket.gethostname()
EARLY_TOLERANCE=200


if RD:
    TRAIN_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'train_files_RD.txt'))
    TEST_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'test_files_RD.txt'))
else:
    TRAIN_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'train_files_voxel_b{}.txt'.format(FLAGS.box)))
    TEST_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'test_files_voxel_b{}.txt'.format(FLAGS.box)))
                                                                   
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP) # CLIP THE LEARNING RATE!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

    


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl,  labels_pl, global_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_FEAT,NUM_GLOB) 

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            alpha = tf.placeholder(dtype=tf.float32, shape=())
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            print("--- Get model and loss")

            pred , max_pool = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,global_pl = global_pl,
                                              bn_decay=bn_decay,
                                              num_class=NUM_CLASSES, weight_decay=FLAGS.wd,
            )
            
            
            mu = tf.Variable(tf.zeros(shape=(FLAGS.n_clusters,FLAGS.max_dim)),name="mu",trainable=True) #k centroids
            
            classify_loss = MODEL.get_focal_loss(pred,labels_pl,NUM_CLASSES)
            kmeans_loss, stack_dist= MODEL.get_loss_kmeans(max_pool,mu, FLAGS.max_dim,
                                                            FLAGS.n_clusters,alpha)
                    
            full_loss = kmeans_loss + classify_loss

            
            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(classify_loss, global_step=batch)
            train_op_full = optimizer.minimize(full_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        
        
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        print("Total number of weights for the model: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl':labels_pl,
               'global_pl':global_pl,
               'is_training_pl': is_training_pl,
               'max_pool':max_pool,
               'pred': pred,
               'alpha': alpha,
               'stack_dist':stack_dist,
               'classify_loss': classify_loss,
               'kmeans_loss': kmeans_loss,
               'train_op': train_op,
               'train_op_full': train_op_full,
               'merged': merged,
               'step': batch,
        }

        best_acc = -1
        
        
        if FLAGS.min == 'loss':early_stop = np.inf
        else:early_stop = 0
        earlytol = 0


        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            is_full_training = epoch > MAX_PRETRAIN
            
            
            lss = eval_one_epoch(sess, ops, test_writer,is_full_training)
            
            if is_full_training:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'cluster.ckpt'))
            else:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
            log_string("Model saved in file: %s" % save_path)

            max_pool = train_one_epoch(sess, ops, train_writer,is_full_training)
            if epoch == MAX_PRETRAIN:
                centers  = KMeans(n_clusters=FLAGS.n_clusters).fit(np.squeeze(max_pool))
                centers = centers.cluster_centers_
                sess.run(tf.assign(mu,centers))
            
def get_batch(data,label,global_pl,  start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_global = global_pl[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:,:]
    return batch_data, batch_label, batch_global

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
    y: true labels, numpy.array with shape `(n_samples,)`
    y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
    accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    #print(y_pred.shape,y_true.shape)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
        

def train_one_epoch(sess, ops, train_writer,is_full_training):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    
    train_idxs = np.arange(0, len(TRAIN_FILES))
    
    acc =  loss_sum = 0
    y_pool = []
    y_assign = []
    for fn in range(len(TRAIN_FILES)):
        #log_string('----' + str(fn) + '-----')
        current_file = os.path.join(H5_DIR,TRAIN_FILES[train_idxs[fn]])
        if RD:
            current_data,  current_cluster,current_label = provider.load_h5_data_label_seg(current_file)
        else:
            current_data, current_label = provider.load_h5(current_file,'seg')

        adds = provider.load_add(current_file,['global'])
        if NUM_GLOB < adds['global'].shape[1]:
            log_string("Using less global variables than possible")
            adds['global'] = adds['global'][:,:NUM_GLOB]
            
        
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        if FLAGS.nbatches>0:
            num_batches = FLAGS.nbatches
                                                                
        log_string(str(datetime.now()))     
   
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            batch_data, batch_label, batch_global = get_batch(current_data, current_label, adds['global'],start_idx, end_idx)
            cur_batch_size = end_idx-start_idx
                

            #print(batch_weight) 
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['global_pl']: batch_global,
                         ops['is_training_pl']: is_training,
                         ops['alpha']: 10*(EPOCH_CNT-MAX_PRETRAIN+1),
                         
                        
            }
            if is_full_training:
                summary, step, _, loss_val, pred_val,max_pool,dist = sess.run([ops['merged'], ops['step'],
                                                                               ops['train_op_full'], ops['kmeans_loss'],
                                                                               ops['pred'],ops['max_pool'],
                                                                               ops['stack_dist']
                                                                           ],
                                                                              
                                                                              feed_dict=feed_dict)


                cluster_assign = np.zeros((cur_batch_size), dtype=int)
                for i in range(cur_batch_size):
                    index_closest_cluster = np.argmin(dist[:, i])
                    cluster_assign[i] = index_closest_cluster
                if RD:
                    batch_cluster = current_cluster[start_idx:end_idx]
                    if batch_cluster.size == cluster_assign.size:    
                        acc+=cluster_acc(batch_cluster,cluster_assign)


            else:
                summary, step, _, loss_val, pred_val,max_pool = sess.run([ops['merged'], ops['step'],
                                                                          ops['train_op'], ops['classify_loss'],
                                                                          ops['pred'],ops['max_pool']
                                                                      ],
                                                                              
                                                                         feed_dict=feed_dict)


            loss_sum += np.mean(loss_val)
            if len(y_pool)==0:
                y_pool=np.squeeze(max_pool)                
                
            else:
                y_pool=np.concatenate((y_pool,np.squeeze(max_pool)),axis=0)


            train_writer.add_summary(summary, step)
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('train clustering accuracy: %f' % (acc/ float(num_batches)))
    return y_pool
        
def eval_one_epoch(sess, ops, test_writer,is_full_training):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_FILES))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    loss_sum = acc =0
    acc_seg = 0


    for fn in range(len(TEST_FILES)):
        #log_string('----' + str(fn) + '-----')
        current_file = os.path.join(H5_DIR,TEST_FILES[test_idxs[fn]])
        if RD:
            current_data,  current_cluster,current_label = provider.load_h5_data_label_seg(current_file)
        else:
            current_data, current_label = provider.load_h5(current_file,'seg')
        adds = provider.load_add(current_file,['global'])
        if NUM_GLOB < adds['global'].shape[1]:
            log_string("Using less global variables than possible")
            adds['global'] = adds['global'][:,:NUM_GLOB]

        current_label = np.squeeze(current_label)

            
            
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            batch_data, batch_label, batch_global = get_batch(current_data, current_label, adds['global'],start_idx, end_idx)
            cur_batch_size = end_idx-start_idx
            
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['is_training_pl']: is_training,
                         ops['global_pl']: batch_global,
                         ops['labels_pl']: batch_label,
                         ops['alpha']: 10*(EPOCH_CNT-MAX_PRETRAIN+1),
            }
            if is_full_training:
                summary, step, loss_val, pred_val,max_pool,dist= sess.run([ops['merged'], ops['step'],
                                                                           ops['kmeans_loss'], ops['pred'],
                                                                           ops['max_pool'],ops['stack_dist'],
                                                                           
                                                                           #ops['pi']
                                                                           
                                                                           ],
                                                                              feed_dict=feed_dict)

                cluster_assign = np.zeros((cur_batch_size), dtype=int)
                for i in range(cur_batch_size):
                    index_closest_cluster = np.argmin(dist[:, i])
                    cluster_assign[i] = index_closest_cluster
                if RD:
                    batch_cluster = current_cluster[start_idx:end_idx]
                    
                    if batch_cluster.size == cluster_assign.size:    
                        acc+=cluster_acc(batch_cluster,cluster_assign)

            else:
                summary, step, loss_val, pred_val,max_pool= sess.run([ops['merged'], ops['step'],
                                                                      ops['classify_loss'], ops['pred'],
                                                                      ops['max_pool'],
                                                                  ],
                                                                     feed_dict=feed_dict)



            test_writer.add_summary(summary, step)
            


            loss_sum += np.mean(loss_val)
        
    total_loss = loss_sum*1.0 / float(num_batches)
    log_string('mean loss: %f' % (total_loss))
    log_string('testing clustering accuracy: %f' % (acc / float(num_batches)))

    EPOCH_CNT += 1
    if FLAGS.min == 'acc':
        return total_correct / float(total_seen)
    else:
        return total_loss
    


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
