import argparse
import h5py
from math import *
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os, ast
import sys


np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
#from MVA_cfg import *
import provider
import gapnet_classify as MODEL


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters [Default: 3]')
parser.add_argument('--max_dim', type=int, default=3, help='Dimension of the encoding layer [Default: 3]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch', type=int, default=512, help='Batch Size  during training [default: 512]')
parser.add_argument('--num_point', type=int, default=100, help='Point Number [default: 100]')
parser.add_argument('--data_dir', default='../h5', help='directory with data [default: ../h5]')
parser.add_argument('--nfeat', type=int, default=8, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=20, help='Number of categories [default: 20]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')
parser.add_argument('--full_train',  default=False, action='store_true',help='load full training results [default: False]')

FLAGS = parser.parse_args()
LOG_DIR = os.path.join('..','logs',FLAGS.log_dir)
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)
H5_OUT = FLAGS.h5_folder
if not os.path.exists(H5_OUT): os.mkdir(H5_OUT)  

# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch
NFEATURES = FLAGS.nfeat
FULL_TRAINING = FLAGS.full_train


NUM_CATEGORIES = FLAGS.ncat
#Only used to get how many parts per category

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(NUM_POINT))
print('#### Using GPUs: {0}'.format(FLAGS.gpu))



    
print('### Starting evaluation')


EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'evaluate_files_wztop.txt')) 

  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl,  labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES) 
            batch = tf.Variable(0, trainable=False)
            alpha = tf.placeholder(tf.float32, shape=())
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred,max_pool = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,num_class=NUM_CATEGORIES)
            mu = tf.Variable(tf.zeros(shape=(FLAGS.n_clusters,FLAGS.max_dim)),name="mu",trainable=False) #k centroids
            

            classify_loss = MODEL.get_focal_loss(pred, labels_pl,NUM_CATEGORIES)
            kmeans_loss, stack_dist= MODEL.get_loss_kmeans(max_pool,mu, FLAGS.max_dim,
                                                           FLAGS.n_clusters,alpha)

            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        if FULL_TRAINING:
            saver.restore(sess,os.path.join(LOG_DIR,'cluster.ckpt'))
        else:
            saver.restore(sess,os.path.join(LOG_DIR,'model.ckpt'))
        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'stack_dist':stack_dist,
               'kmeans_loss':kmeans_loss,
               'pred': pred,
               'alpha': alpha,
               'max_pool': max_pool,
               'is_training_pl':is_training_pl,
               'classify_loss': classify_loss,}
            
        eval_one_epoch(sess,ops)

def get_batch(data,label,start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_data = data[start_idx:end_idx,:,:]
    return batch_data, batch_label

        
def eval_one_epoch(sess,ops):
    is_training = False

    eval_idxs = np.arange(0, len(EVALUATE_FILES))
    y_val = []
    for fn in range(len(EVALUATE_FILES)):
        current_file = os.path.join(H5_DIR,EVALUATE_FILES[eval_idxs[fn]])
        current_data, current_label, current_cluster = provider.load_h5_data_label_seg(current_file)
        adds = provider.load_add(current_file,['masses'])

        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        num_batches = 5

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data, batch_label = get_batch(current_data, current_label,start_idx, end_idx)
            batch_cluster = current_cluster[start_idx:end_idx]
            cur_batch_size = end_idx-start_idx


            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['alpha']: 1, #No impact on evaluation,
                         ops['is_training_pl']: is_training,
            }

            loss, dist,max_pool = sess.run([ops['kmeans_loss'], ops['stack_dist'],
                                            ops['max_pool']],feed_dict=feed_dict)
            cluster_assign = np.zeros((cur_batch_size), dtype=int)
            for i in range(cur_batch_size):
                index_closest_cluster = np.argmin(dist[:, i])
                cluster_assign[i] = index_closest_cluster

            batch_cluster = np.array([np.where(r==1)[0][0] for r in current_cluster[start_idx:end_idx]])

            if len(y_val)==0:      
                y_val=batch_cluster
                y_assign=cluster_assign
                y_pool=np.squeeze(max_pool)
                y_mass = adds['masses'][start_idx:end_idx]
            else:
                y_val=np.concatenate((y_val,batch_cluster),axis=0)
                y_assign=np.concatenate((y_assign,cluster_assign),axis=0)
                y_pool=np.concatenate((y_pool,np.squeeze(max_pool)),axis=0)
                y_mass=np.concatenate((y_mass,adds['masses'][start_idx:end_idx]),axis=0)
                            
    with h5py.File(os.path.join(H5_OUT,'{0}.h5'.format(FLAGS.name)), "w") as fh5:
        dset = fh5.create_dataset("pid", data=y_val) #Real jet categories
        dset = fh5.create_dataset("label", data=y_assign) #Cluster labeling
        dset = fh5.create_dataset("max_pool", data=y_pool)        
        dset = fh5.create_dataset("masses", data=y_mass)

################################################          
    

if __name__=='__main__':
  eval()
