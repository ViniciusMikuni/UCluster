import os
import sys
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification


def shuffle_data(data, labels,global_pl=[],weights=[]):
  """ Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,N, numpy array
    Return:
      shuffled data, label and shuffle indices
  """
  idx = np.arange(len(labels))
  np.random.shuffle(idx)
  #return data[idx,:], labels[idx,:], idx
  if global_pl != []:
    return data[idx,:], labels[idx], global_pl[idx,:], idx
  elif weights == []:
    return data[idx,:], labels[idx],idx
  else:
    return data[idx,:], labels[idx], weights[idx],idx


def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
  """ Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
  """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
  return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
  """ Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
  """
  B, N, C = batch_data.shape
  assert(clip > 0)
  jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
  jittered_data += batch_data
  return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
  """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
  """
  B, N, C = batch_data.shape
  shifts = np.random.uniform(-shift_range, shift_range, (B,3))
  for batch_index in range(B):
    batch_data[batch_index,:,:] += shifts[batch_index,:]
  return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
  """
  B, N, C = batch_data.shape
  scales = np.random.uniform(scale_low, scale_high, B)
  for batch_index in range(B):
    batch_data[batch_index,:,:] *= scales[batch_index]
  return batch_data

def norm_inputs_point_cloud(data,cloud=True):
  """ Normalize the input data by the mean of the distribution"""
  if cloud:
    NPOINTS = data.shape[1]
    NFEATURES = data.shape[2]
  else:
    NPOINTS = data.shape[0]
    NFEATURES = data.shape[1]
  reshape = np.reshape(data,(-1,NFEATURES))
  #scaler = StandardScaler()
  scaler = MinMaxScaler()
  scaler.fit(reshape[0::NPOINTS])
  #print(scaler.mean_)
  zero_arr = [0]*NFEATURES
  zero_arr = scaler.transform([zero_arr])
  reshape = scaler.transform(reshape)
  for i in range(NFEATURES):
    reshape[reshape==zero_arr[0][i]]=0
  if cloud:
    reshape = np.reshape(reshape,(-1,NPOINTS,NFEATURES))
    #print(reshape)
  else:
    reshape = np.reshape(reshape,(-1,NFEATURES))

  print("Normalized the data")
  return reshape



def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]

def load_add(h5_filename,names=[]):
  f = h5py.File(h5_filename,'r')
  if len(names) ==0:
    names = list(f.keys())
    print (names)
    names.remove('data')
    names.remove('pid')
    names.remove('label')

  datasets = {}
  for data in names:
    datasets[data] = f[data][:]

  return datasets

def load_h5(h5_filename,mode='seg',unsup=False,glob=False):
  f = h5py.File(h5_filename,'r')
  data = f['data'][:]
  #data = norm_inputs_point_cloud(data)
  if mode == 'class':
    label = f['pid'][:].astype(int)
  elif mode == 'seg':
    label = f['label'][:].astype(int)
  else:
    print('No mode found')
  print("loaded {0} events".format(len(data)))
  if glob:
    global_pl = f['global'][:]
    return (data, label,global_pl)
  else:
    return (data, label)
    #global_pl = norm_inputs_point_cloud(global_pl,cloud=False)    




def load_h5_weights(h5_filename):
  f = h5py.File(h5_filename,'r')
  data = f['data'][:]
  label = f['pid'][:]
  weights = {}
  for var in f.keys():
    if 'w' in var:
      weights[var]=f[var][:]
    
  
  return (data, label,weights)

def load_h5_eval(h5_filename):
  f = h5py.File(h5_filename,'r')
  data = f['data'][:]
  label = f['pid'][:]
  weight_nom = np.abs(f['weight_nom'][:])
  weight_up = np.abs(f['weight_up'][:])
  weight_down = np.abs(f['weight_down'][:])
  return (data, label,weight_nom,weight_up,weight_down)




def loadDataFile(filename):
  return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
  f = h5py.File(h5_filename,'r')
  data = f['data'][:] # (2048, 2048, 3)
  #data = norm_inputs_point_cloud(data)
  label = f['pid'][:] # (2048, 1)
  seg = f['label'][:] # (2048, 2048)
  print("loaded {0} events".format(len(data)))
  
  return (data, label, seg)
