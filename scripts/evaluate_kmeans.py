import argparse
import h5py
from math import *
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn import metrics
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
parser.add_argument('--params', default='[50,1,32,64,128,128,2,64,128,128,256,256,256]', help='DNN parameters[[k,H,A,F,F,F,H,A,F,C,F]]')
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters [Default: 2]')
parser.add_argument('--max_dim', type=int, default=512, help='Dimension of the encoding layer [Default: 512]')
parser.add_argument('--model_path', default='../logs/PU/model.ckpt', help='Model checkpoint path')
parser.add_argument('--modeln', type=int,default=-1, help='Model number')
parser.add_argument('--nglob', type=int, default=2, help='Number of global features [default: 2]')
parser.add_argument('--batch', type=int, default=64, help='Batch Size  during training [default: 64]')
parser.add_argument('--num_point', type=int, default=500, help='Point Number [default: 500]')
parser.add_argument('--data_dir', default='../h5', help='directory with data [default: ../data/PU]')
parser.add_argument('--nfeat', type=int, default=8, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')
parser.add_argument('--gwztop',  default=False, action='store_true',help='use the set with g/w/z/top [default: False]')
parser.add_argument('--full_train',  default=False, action='store_true',help='load full training results [default: False]')

FLAGS = parser.parse_args()
MODEL_PATH = FLAGS.model_path
NUM_GLOB = FLAGS.nglob
params = ast.literal_eval(FLAGS.params)
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

if FLAGS.gwztop:
    EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'evaluate_files_gwztop.txt')) 
else:
    EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'evaluate_files_wztop.txt')) 

  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl,  labels_pl, global_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES,NUM_GLOB) 
            batch = tf.Variable(0, trainable=False)
            alpha = tf.placeholder(tf.float32, shape=())
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred,max_pool = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,params=params,global_pl = global_pl,num_class=NUM_CATEGORIES)
            mu = tf.Variable(tf.zeros(shape=(FLAGS.n_clusters,FLAGS.max_dim)),name="mu",trainable=False) #k centroids
            
            #loss  = MODEL.get_loss(pred,labels_pl,NUM_CATEGORIES)
            loss = MODEL.get_focal_loss(pred, labels_pl,NUM_CATEGORIES)
            kmeans_loss, stack_dist= MODEL.get_loss_kmeans(max_pool,mu, FLAGS.max_dim,
                                                           FLAGS.n_clusters,alpha)

            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        if FULL_TRAINING:
            saver.restore(sess,os.path.join(MODEL_PATH,'cluster_dkm.ckpt'))
        else:
            saver.restore(sess,os.path.join(MODEL_PATH,'model.ckpt'))
        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'global_pl':global_pl,
               'stack_dist':stack_dist,
               'kmeans_loss':kmeans_loss,
               'pred': pred,
               'alpha': alpha,
               'max_pool': max_pool,
               'is_training_pl':is_training_pl,
               'loss': loss,}
            
        eval_one_epoch(sess,ops)

def get_batch(data,label,global_pl,  start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_global = global_pl[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:,:]
    return batch_data, batch_label, batch_global

        
def eval_one_epoch(sess,ops):
    is_training = False

    total_correct = total_correct_ones =  total_seen =total_seen_ones= loss_sum =0    
    eval_idxs = np.arange(0, len(EVALUATE_FILES))
    y_val = []
    for fn in range(len(EVALUATE_FILES)):
        current_file = os.path.join(H5_DIR,EVALUATE_FILES[eval_idxs[fn]])
        current_truth = []
        current_mass = []
        current_data, current_label, current_cluster = provider.load_h5_data_label_seg(current_file)
        adds = provider.load_add(current_file,['global','masses'])

        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        #num_batches = 10

        

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data, batch_label, batch_global = get_batch(current_data, current_label, adds['global'],start_idx, end_idx)
            batch_cluster = current_cluster[start_idx:end_idx]
            cur_batch_size = end_idx-start_idx


            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['global_pl']: batch_global,
                         ops['labels_pl']: batch_label,
                         ops['alpha']: 100,
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
                #y_loss = loss
                y_assign=cluster_assign
                y_pool=np.squeeze(max_pool)
                y_mass = adds['masses'][start_idx:end_idx]
                #y_data = batch_data[:,:,:3]
            else:
                y_val=np.concatenate((y_val,batch_cluster),axis=0)
                #y_loss=np.concatenate((y_loss,loss),axis=0)
                y_assign=np.concatenate((y_assign,cluster_assign),axis=0)
                y_pool=np.concatenate((y_pool,np.squeeze(max_pool)),axis=0)
                y_mass=np.concatenate((y_mass,adds['masses'][start_idx:end_idx]),axis=0)
                #y_data=np.concatenate((y_data,batch_data[:,:,:3]),axis=0)
                            
    pos_label = 1
    total_loss = loss_sum*1.0 / float(num_batches)    


    with h5py.File(os.path.join(H5_OUT,'{0}.h5'.format(FLAGS.name)), "w") as fh5:
        dset = fh5.create_dataset("pid", data=y_val)
        #dset = fh5.create_dataset("data", data=y_data)
        dset = fh5.create_dataset("label", data=y_assign)
        dset = fh5.create_dataset("max_pool", data=y_pool)        
        dset = fh5.create_dataset("masses", data=y_mass)

################################################          
    

if __name__=='__main__':
  eval()
