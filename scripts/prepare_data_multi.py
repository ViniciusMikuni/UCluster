import h5py
import os, sys
import numpy as np
from optparse import OptionParser



def save_voxels(sample,NPARTS=100,NVOXELS=3):
    ''' Prepare the samples to be used during training. 
    sample: raw h5 file
    NPARTS: Number of particles to save per event
    NVOXELS: Number of histogram bins
    '''
    pid = []

    MASSRANGE = np.linspace(0,200,NVOXELS)
    MASSRANGE = np.append(MASSRANGE, [10000])


    data = sample['jetConstituentList'][:]
    labels = sample['jets'][:,53:]
    masses = sample['jets'][:,3]

    glob = np.concatenate( #In the end these features are not used in the main model
        (
            np.expand_dims(sample['jets'][:,14],axis=-1), #c11 
            np.expand_dims(sample['jets'][:,15],axis=-1), #c12
            np.expand_dims(sample['jets'][:,16],axis=-1), #c21
            np.expand_dims(sample['jets'][:,12],axis=-1), #zlogz            
     ),axis=-1)

    keep_mask = ((labels[:,3]==1)|(labels[:,2]==1)|(labels[:,4]==1)) &(masses[:]>10) &(masses[:]<200) # w,z,top
    data=data[keep_mask]
    labels=labels[keep_mask]
    masses = masses[keep_mask]
    glob = glob[keep_mask]
    pid = np.zeros(masses.shape)



    features=np.concatenate(
        (
            np.expand_dims(data[:,:,8],axis=-1), 
            np.expand_dims(data[:,:,11],axis=-1),             
            np.expand_dims(np.ma.log(data[:,:,6]).filled(0),axis=-1), 
            np.expand_dims(data[:,:,13],axis=-1), 
            np.expand_dims(np.ma.log(data[:,:,3]).filled(0),axis=-1), 
            np.expand_dims(np.ma.log(data[:,:,4]).filled(0),axis=-1), 
            np.expand_dims(np.ma.log(data[:,:,5]).filled(0),axis=-1),             
            np.expand_dims(data[:,:,15],axis=-1), 
     ),axis=-1)
    features[np.abs(features)==np.inf] = 0
    labels = np.concatenate(
        (
            np.expand_dims(labels[:,2],axis=-1), 
            np.expand_dims(labels[:,3],axis=-1), 
            np.expand_dims(labels[:,4],axis=-1), 
        ),axis=-1)

    ivoxel = 0

    for imass in range(NVOXELS):
        mask_mass = (masses[:] >= MASSRANGE[imass]) & (masses[:] < MASSRANGE[imass+1]) 
        if len(pid[mask_mass]) >0:
            pid[mask_mass] = ivoxel
            ivoxel+=1


    
    return pid,features,labels, masses, glob
    
    
   


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--nparts", type=int, default=100, help="Number of particles per event")
    parser.add_option("--nvoxels", type=int, default=20, help="Number of mass divisions to use")
    parser.add_option("--make_eval", action="store_true", default=False, help="Only produce evaluation sample. Otherwise will produce train/test samples")
    parser.add_option("--dir", type="string", default="../samples/", help="Folder containing the input files")
    parser.add_option("--out", type="string", default="../h5/", help="Folder to save output files")

    (flags, args) = parser.parse_args()


    NPARTS=flags.nparts
    NVOXELS=flags.nvoxels
    make_eval = flags.make_eval
    samples_path = flags.dir
    save_path = flags.out

    samples_path = '/scratch/vmikuni/ML/'
    
    #Assuming that the 2 samples were saved under these respective folders
    if make_eval:
        samples_path = os.path.join(samples_path,'val')
    else:
        samples_path = os.path.join(samples_path,'train')

    files = os.listdir(samples_path)
    files = [f for f in files if f.endswith(".h5")]
    files = [f for f in files if '100p' in f] #I'm using the 100p samples, but could be changed if a different one was used

    pids = np.array([])
    ncount = 0

    for f in files:
        data = h5py.File(os.path.join(samples_path,f),"r")
        if 'jetConstituentList' in data.keys():
            pid,feat,lab,mass,glob = save_voxels(data,NPARTS,NVOXELS)
            if pids.size==0:
                pids = pid
                feats = feat
                labs = lab
                masses = mass
                globs = glob
            else:
                pids = np.concatenate((pids,pid),axis=0)
                feats = np.concatenate((feats,feat),axis=0)
                labs = np.concatenate((labs,lab),axis=0)
                masses = np.concatenate((masses,mass),axis=0)
                globs = np.concatenate((globs,glob),axis=0)



    NTRAIN = int(0.8*len(labs)) #80% of the data is used for training
    if make_eval:
        with h5py.File(os.path.join(save_path,"eval_multi_{}v_{}P.h5".format(NVOXELS,NPARTS)), "w") as fh5: 
            dset = fh5.create_dataset("data", data=feats)
            dset = fh5.create_dataset("label", data=labs)
            dset = fh5.create_dataset("pid", data=pids)
            dset = fh5.create_dataset("masses", data=masses)
            dset = fh5.create_dataset("global", data=globs)

    else:

        with h5py.File(os.path.join(save_path,"train_multi_{}v_{}P.h5".format(NVOXELS,NPARTS)), "w") as fh5:#         
            dset = fh5.create_dataset("data", data=feats[:NTRAIN])
            dset = fh5.create_dataset("label", data=labs[:NTRAIN])
            dset = fh5.create_dataset("pid", data=pids[:NTRAIN])
            dset = fh5.create_dataset("masses", data=masses[:NTRAIN])
            dset = fh5.create_dataset("global", data=globs[:NTRAIN])
            
        with h5py.File(os.path.join(save_path,"test_multi_{}v_{}P.h5".format(NVOXELS,NPARTS)), "w") as fh5: #        
            dset = fh5.create_dataset("data", data=feats[NTRAIN:])
            dset = fh5.create_dataset("label", data=labs[NTRAIN:])
            dset = fh5.create_dataset("pid", data=pids[NTRAIN:])
            dset = fh5.create_dataset("masses", data=masses[NTRAIN:])
            dset = fh5.create_dataset("global", data=globs[NTRAIN:])

            

