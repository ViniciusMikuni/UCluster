import pandas as pd
import h5py
import os, sys
import numpy as np
import random
import pyjet as fj
from optparse import OptionParser

np.random.seed(10)


def deltaR(x, y):
    return ((x.phi-y.phi)**2 + (x.eta-y.eta)**2)**0.5

def tau21(jet,subR=0.2):
    '''Input: jet from the jet clustering result '''
    jet_substruct_features = {}        
    seq = fj.cluster(jet, R=subR, algo='kt')
    cnsts = jet.constituents()
    cndts1 = seq.exclusive_jets(1)
    tau1 = subjettiness(cndts1, cnsts)
    if (len(cnsts)>1):
        cndts2 = seq.exclusive_jets(2)
        tau2 = subjettiness(cndts2, cnsts)
    else: 
        tau2 = 0
        
    return tau2/tau1


def subjettiness(cndts, cnsts):
    d0 = sum(c.pt for c in cnsts)
    ls = []
    for c in cnsts:
        dRs = [deltaR(c,cd) for cd in cndts]
        ls += [c.pt * min(dRs)]
    return sum(ls)/d0




def clustering_anomaly(data,NPARTS=100,NVOXELS=3,name='',R=1.0,RD=False):
    pid = []
    features = []
    labels = []
    globs = []
    masses = []
    taus = []

    MASSRANGE = np.linspace(10,1000,NVOXELS)
    MASSRANGE = np.append(MASSRANGE, [100000])

    for event in range(data.shape[0]): #Unfortunately couldn't find a faster way to improve this part
        if event%10000 ==0:
            print("processing event {} out of {}".format(event,data.shape[0]))
        pseudojets_input = np.zeros(len([x for x in data[event][::3] if x > 0]), dtype=fj.DTYPE_PTEPM)
        for j in range(700):
            if (data[event][j*3]>0):
                pseudojets_input[j]['pT'] = data[event][j*3]
                pseudojets_input[j]['eta'] = data[event][j*3+1]
                pseudojets_input[j]['phi'] = data[event][j*3+2]

        sequence = fj.cluster(pseudojets_input, R=R, p=-1)
        jets = sequence.inclusive_jets(ptmin=20)
        if len(jets) < 2:continue
        nparts = 0
        label = []
        feature = []
        glob = []
        mass_ordered = []
        vector4 = np.array([0.0,0.0,0.0,0.0])
        for jet in jets:
            mass_ordered.append(jet.mass)
        idx = np.flip(np.argsort(mass_ordered))
        ijet=0
        for jetidx in idx:
            jet = jets[jetidx]
            if jet.mass < 10: continue
            if ijet >= NJETS: continue
            if ijet <=1:vector4+=np.array([jet.px,jet.py,jet.pz,jet.e])
            if len(glob)<4:                
                glob.append(np.log(jet.mass))
                glob.append(tau21(jet))
                
            mass_range = next(x[0] for x in enumerate(MASSRANGE) if x[1] >= jet.mass)
            nparts+=len(jet.constituents())
            ijet+=1
            
            for pf in jet.constituents():
                feature.append(
                    [pf.eta-jet.eta, pf.phi-jet.phi, np.log(pf.pt/jet.pt),
                     np.log(pf.e/jet.e),np.log(pf.pt),np.log(pf.e),
                     np.sqrt((pf.eta-jet.eta)**2 + (pf.phi-jet.phi)**2)]
                )
                label.append(mass_range)

        if ijet<2: continue            

        if RD:
            if data[event][2100] ==1:                
                keep = random.choices([0,1],weights=(90,10)) # Make 99% bkg and 1% sig
                if keep[0] ==0:
                    continue
                
            pid.append(data[event][2100])
        masses.append(np.sqrt(np.abs(vector4[3]**2 - vector4[0]**2 - vector4[1]**2 -vector4[2]**2)))
        globs.append(glob)

        if len(feature)>0: #Zero-padding
            if len(feature) < NPARTS:
                for dummy in range(NPARTS - len(feature)):
                    label.append(0)
                    feature.append([0,0,0,0,0,0,0])
                label = np.array(label,dtype=np.int8)
                feature=np.array(feature)
            if len(feature)>NPARTS:
                label=np.array(label,dtype=np.int8)[:NPARTS]
                feature=np.array(feature)[:NPARTS]
            labels.append(label)
            features.append(feature)
        else:
            if RD:pid[event] = -1
        

    labels = np.array(labels,dtype=np.int8)
    globs = np.array(globs)
    features=np.array(features)
    masses=np.array(masses)
    print('features',features.shape,'globs',globs.shape,'label',labels.shape)
    
    
    if RD:
        pid = np.array(pid,dtype=np.int8)
        
    with h5py.File(os.path.join(save_path,"{}_{}P_{}NJET.h5".format(name,NPARTS,NJETS)), "w") as fh5:         
        dset = fh5.create_dataset("global", data=globs)
        dset = fh5.create_dataset("data", data=features)
        dset = fh5.create_dataset("label", data=labels)
        dset = fh5.create_dataset("masses", data=masses)
        if RD:                
            dset = fh5.create_dataset("pid", data=pid)
            
            
            
    
    
        
    
    

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--nparts", type=int, default=100, help="Number of particles per event")
    parser.add_option("--nvoxels", type=int, default=20, help="Number of mass divisions to use")
    parser.add_option("--njets", type=int, default=2, help="Number of jets to save per event")
    parser.add_option("--boxn", type=int, default=3, help="Number of the black box to parse. If RD is True, then this flag has no effect")
    parser.add_option("--RD", action="store_true", default=False, help="Use the RD dataset")
    parser.add_option("--dir", type="string", default="../samples/", help="Folder containing the input files")
    parser.add_option("--out", type="string", default="../h5/", help="Folder to save output files")

    (flags, args) = parser.parse_args()


    NPARTS=flags.nparts
    NVOXELS=flags.nvoxels
    NJETS=flags.njets


    RD = flags.RD 
    samples_path = flags.dir
    save_path = flags.out

    samples_path = '/scratch/vmikuni/ML/'
        

    if RD:
        sample = 'events_anomalydetection.h5'
    else:
        sample = 'events_LHCO2020_BlackBox{}.h5'.format(flags.boxn)



    # data_train = pd.read_hdf(os.path.join(samples_path,sample),start = 0,stop=400000)
    # data_test =  pd.read_hdf(os.path.join(samples_path,sample),start = 400001,stop=550000)
    # data_eval =  pd.read_hdf(os.path.join(samples_path,sample),start = 650001,stop=990001)

    data_train = pd.read_hdf(os.path.join(samples_path,sample),start = 0,stop=4000)
    data_test =  pd.read_hdf(os.path.join(samples_path,sample),start = 4000,stop=5500)
    data_eval =  pd.read_hdf(os.path.join(samples_path,sample),start = 6500,stop=9900)
    print("Loaded data set")

    if RD:
        clustering_anomaly(data_train.to_numpy(),NPARTS,NVOXELS,name = "train_{}v_RD".format(NVOXELS),R=1.0,RD=True)
        clustering_anomaly(data_test.to_numpy(),NPARTS,NVOXELS,name = "test_{}v_RD".format(NVOXELS),R=1.0,RD=True)
        clustering_anomaly(data_eval.to_numpy(),NPARTS,NVOXELS,name = "eval_{}v_RD".format(NVOXELS),R=1.0,RD=True)
    else:
        clustering_anomaly(data_train.to_numpy(),NPARTS,NVOXELS,name = "train_{}v_B{}".format(NVOXELS,boxn),R=1.0)
        clustering_anomaly(data_test.to_numpy(),NPARTS,NVOXELS,name = "test_{}v_B{}".format(NVOXELS,boxn),R=1.0)
        clustering_anomaly(data_eval.to_numpy(),NPARTS,NVOXELS,name = "eval_{}v_B{}".format(NVOXELS,boxn),R=1.0)

