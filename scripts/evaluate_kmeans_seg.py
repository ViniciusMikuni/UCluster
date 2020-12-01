import argparse
import h5py
from math import *
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os, ast
import sys


#np.set_printoptions(threshold=sys.maxsize)
from scipy.optimize import linear_sum_assignment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
#from MVA_cfg import *
import provider
import gapnet_seg as MODEL


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters [Default: 2]')
parser.add_argument('--max_dim', type=int, default=2, help='Dimension of the encoding layer [Default: 512]')
parser.add_argument('--log_dir', default='anomaly', help='Log dir [default: log]')
parser.add_argument('--nglob', type=int, default=4, help='Number of global features [default: 4]')
parser.add_argument('--batch', type=int, default=1024, help='Batch Size  during training [default: 1024]')
parser.add_argument('--num_point', type=int, default=100, help='Point Number [default: 500]')
parser.add_argument('--data_dir', default='../h5', help='directory with data [default: ../h5]')
parser.add_argument('--nfeat', type=int, default=7, help='Number of features [default: 7]')
parser.add_argument('--ncat', type=int, default=21, help='Number of categories [default: 21]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')
parser.add_argument('--box', type=int, default=1, help='Black Box number, ignored if RD dataset [default: 1]')
parser.add_argument('--RD',  default=False, action='store_true',help='Use RD data set [default: False')
parser.add_argument('--full_train',  default=False, action='store_true',help='Use full training [default: False')

FLAGS = parser.parse_args()
LOG_DIR = os.path.join('..','logs',FLAGS.log_dir)
NUM_GLOB = FLAGS.nglob
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)
H5_OUT = FLAGS.h5_folder
if not os.path.exists(H5_OUT): os.mkdir(H5_OUT)  
FULL_TRAINING = FLAGS.full_train
RD = FLAGS.RD
# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch
NFEATURES = FLAGS.nfeat



NUM_CATEGORIES = FLAGS.ncat
#Only used to get how many parts per category

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(NUM_POINT))
print('#### Using GPUs: {0}'.format(FLAGS.gpu))
    
print('### Starting evaluation')



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
        




if RD:
    EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'evaluate_files_RD.txt')) # Need to create those
else:
    EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'evaluate_files_b{}.txt'.format(FLAGS.box))) 

  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl,  labels_pl, global_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES,NUM_GLOB) 
            batch = tf.Variable(0, trainable=False)
            alpha = tf.placeholder(tf.float32, shape=())
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred,max_pool = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,global_pl = global_pl,num_class=NUM_CATEGORIES)
            mu = tf.Variable(tf.zeros(shape=(FLAGS.n_clusters,FLAGS.max_dim)),name="mu",trainable=False) #k centroids

            classify_loss = MODEL.get_focal_loss(pred,labels_pl,NUM_CATEGORIES)
            kmeans_loss, stack_dist= MODEL.get_loss_kmeans(max_pool,mu, FLAGS.max_dim,
                                                           FLAGS.n_clusters,alpha)


            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = False
        sess = tf.Session(config=config)
        if FULL_TRAINING:
            saver.restore(sess,os.path.join(LOG_DIR,'cluster.ckpt'))
        else:
            saver.restore(sess,os.path.join(LOG_DIR,'model.ckpt'))


        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'global_pl':global_pl,
               'mu':mu,
               'stack_dist':stack_dist,
               'kmeans_loss':kmeans_loss,
               'pred': pred,
               'alpha': alpha,
               'max_pool': max_pool,
               'classify_loss': classify_loss,}
            
        eval_one_epoch(sess,ops)

def get_batch(data,label,global_pl,  start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_global = global_pl[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:]
    return batch_data, batch_label, batch_global

        
def eval_one_epoch(sess,ops):
    is_training = False
    eval_idxs = np.arange(0, len(EVALUATE_FILES))

    y_assign = []
    y_glob =[]
    acc = 0

    for fn in range(len(EVALUATE_FILES)):
        current_file = os.path.join(H5_DIR,EVALUATE_FILES[eval_idxs[fn]])        
        if RD:
            current_data,  current_cluster,current_label = provider.load_h5_data_label_seg(current_file)
        else:
            current_data, current_label = provider.load_h5(current_file,'seg')

        adds = provider.load_add(current_file,['global','masses'])
        
        if NUM_GLOB < adds['global'].shape[1]:
            print("Using less global variables than possible")
            current_glob = adds['global'][:,:NUM_GLOB]
        else:
            current_glob = adds['global']

        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data, batch_label, batch_global = get_batch(current_data, current_label, current_glob,start_idx, end_idx)

            cur_batch_size = end_idx-start_idx


            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['global_pl']: batch_global,                         
                         ops['labels_pl']: batch_label,
                         ops['alpha']: 1, #No impact during evaluation
                         ops['is_training_pl']: is_training,
            }

            dist,mu,max_pool = sess.run([ops['stack_dist'],ops['mu'],
                                         ops['max_pool']],feed_dict=feed_dict)

            cluster_assign = np.zeros((cur_batch_size), dtype=int)
            if RD:
                batch_cluster = current_cluster[start_idx:end_idx]


            for i in range(cur_batch_size):
                index_closest_cluster = np.argmin(dist[:, i])
                cluster_assign[i] = index_closest_cluster
                if RD:
                    acc+=cluster_acc(batch_cluster,cluster_assign)
                    
            if len(y_assign)==0:                
                if RD:
                    y_val=batch_cluster
                y_assign=cluster_assign
                y_pool=np.squeeze(max_pool)
            else:
                y_assign=np.concatenate((y_assign,cluster_assign),axis=0)
                y_pool=np.concatenate((y_pool,np.squeeze(max_pool)),axis=0)
            
                if RD:
                    y_val=np.concatenate((y_val,batch_cluster),axis=0)
                
                

        if len(y_glob)==0:
            y_glob = adds['global'][:num_batches*BATCH_SIZE]
            y_mass = adds['masses'][:num_batches*BATCH_SIZE]
        else:
            y_glob=np.concatenate((y_glob,adds['global'][:num_batches*BATCH_SIZE]),axis=0)
            y_mass=np.concatenate((y_mass,adds['masses'][:num_batches*BATCH_SIZE]),axis=0)
            

    with h5py.File(os.path.join(H5_OUT,'{0}.h5'.format(FLAGS.name)), "w") as fh5:        
        if RD:
            dset = fh5.create_dataset("label", data=y_val)
        dset = fh5.create_dataset("pid", data=y_assign)
        dset = fh5.create_dataset("max_pool", data=y_pool)
        dset = fh5.create_dataset("global", data=y_glob)
        dset = fh5.create_dataset("masses", data=y_mass)

        


################################################          
    

if __name__=='__main__':
  eval()
