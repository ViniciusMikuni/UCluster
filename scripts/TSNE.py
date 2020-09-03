import numpy as np
import h5py
import matplotlib.pyplot as plt
from pylab import savefig
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from optparse import OptionParser
from scipy.optimize import linear_sum_assignment


colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#1b9e77','#d6d62b','#a65628','#4ca628']
#colors =['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928','#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']


def GetMassLabel(masses):
    TOP_CUT = [160,200]
    Z_CUT = [85,100]
    W_CUT = [10,85]
    labels = np.zeros(masses.shape[0],dtype=int)
    for label,cut in zip(range(3),[TOP_CUT,Z_CUT,W_CUT]):        
        labels[(masses>=cut[0]) & (masses<cut[1])] = label
    
    #print(labels)
    return labels

def GetSVM(data,labels):    
    NTRAIN = int(0.8*data.shape[0])
    
    train = data[:NTRAIN]
    clf = SVC(gamma='auto')
    clf.fit(data[:NTRAIN], labels[:NTRAIN])
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    print(clf.score(data[NTRAIN:], labels[NTRAIN:]))


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    """

    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

    for pair in ind:
        print(pair[1])
        print(w[pair[0],pair[1]]*1.0/np.sum(y_true==pair[1]))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def Draw_anomaly(data,pid,label):    
    tsne = TSNE(n_components=2,perplexity=30)
    
    centers  = KMeans(NCLUSTERS).fit(data)
    kmeans_label = centers.labels_
    
    #X_2d = tsne.fit_transform(data)
    X_2d = data
    
    if NTRUECLUSTERS==2:
        truth_labels = ['QCD','BSM']
    else:
        truth_labels = ['W','Z','Top']        
    labels = []


    if not os.path.exists(os.path.join(options.plots,options.version)):
        os.makedirs(os.path.join(options.plots,options.version))

    plt.figure(figsize=(5, 5))
    j=0
    for i in range(NCLUSTERS):
        if np.sum(pid==i)>0:
            labels.append('cluster {}'.format(j))
            print("Results from clustering DNN")
            #plt.scatter(X_2d[pid[:,i]==1, 0], X_2d[pid[:,i]==1, 1], c=colors[i],label=labels[i])
            plt.scatter(X_2d[pid==i, 0], X_2d[pid==i, 1], c=colors[j+NCLUSTERS],label=labels[j])
            j+=1
            plt.legend()        
    #plt.show()
    plt.savefig(os.path.join(options.plots,options.version,"TSNE_DNN.pdf"))
    plt.figure(figsize=(5, 5))
    for i in range(NTRUECLUSTERS):
        print("Results from truth label")
        plt.scatter(X_2d[label==i, 0], X_2d[label==i, 1], c=colors[i],label=truth_labels[i])
        plt.legend()
    #plt.show()
    plt.savefig(os.path.join(options.plots,options.version,"TSNE_truth.pdf"))
    print("Saved figure at: {}".format(os.path.join(options.plots,options.version)))
    plt.figure(figsize=(5, 5))
    for i in range(NCLUSTERS):
        print("Results from clustering with k-means")
        plt.scatter(X_2d[kmeans_label==i, 0], X_2d[kmeans_label==i, 1], c=colors[i+2*NCLUSTERS],label=labels[i])
        plt.legend()
    plt.savefig(os.path.join(options.plots,options.version,"TSNE_kNN.pdf"))
    # plt.show()
    



parser = OptionParser(usage="%prog [opt]")
parser.add_option("-p","--plots",dest="plots", type="string", default="../plots", help="Path to store plots. [default: %default]")
parser.add_option("--version", type="string", default="test", help="basename of the resulting files. [default: %default]")
parser.add_option("-d","--dir", type="string", default="../h5/", help="Base path for the folder with input root files. [default: %default]")
parser.add_option("--file", type="string", default="wztop_jedi_100p_dkm.h5", help="file name to use. [default: %default]")
parser.add_option("--nclusters", type=int, default=2, help="basename of the resulting files. [default: %default]")
parser.add_option("--ntrue_clusters", type=int, default=2, help="Number of different samples. [default: %default]")
parser.add_option("--events", type=int, default=1000, help="Number of events to use. [default: %default]")

(options, args) = parser.parse_args()
samples_path = options.dir 
sample = options.file

#sample = 'gwtop_jedi.h5'
#sample = 'wztop_jedi_100p_dkm_full.h5'
#sample = 'anomaly_RD.h5'
#sample = 'anomaly_seg_dkm_1pct_focal_c4.h5'
#sample = 'anomaly_seg_dkm_1pct.h5'
print ("loading data sets...")

NTEST =options.events
NCLUSTERS = options.nclusters
NTRUECLUSTERS=options.ntrue_clusters
f = h5py.File(os.path.join(samples_path,sample),'r')
# acc = cluster_acc(f['pid'][:],f['label'][:])
# mass_label = GetMassLabel(f['masses'][:])
# acc_mass = cluster_acc(f['pid'][:],mass_label)
# centers  = KMeans(NTRUECLUSTERS).fit(f['max_pool'][:])
# kmeans_label = centers.labels_
#acc_kmeans = cluster_acc(f['pid'][:],kmeans_label)
# print(f['label'][:NTEST])
# print(f['pid'][:NTEST])
#print(acc,acc_mass,acc_kmeans)

#GetSVM(f['max_pool'][:10000],f['pid'][:10000])
Draw_anomaly(f['max_pool'][:NTEST],f['pid'][:NTEST],f['label'][:NTEST])
