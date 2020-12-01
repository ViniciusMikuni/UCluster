# Unsupervised clustering for collider physics (UCluster)
This repo contains the main code used in the UCluster studies [public document coming soon], implemented in Tensorflow. 
The data set used for the comparisons can be accessed from zenodo in the following links:

* Unsupervised multiclass: https://doi.org/10.5281/zenodo.3602254

* Anomaly detection: https://doi.org/10.5281/zenodo.2629073

# Requirements

[Tensorflow](https://www.tensorflow.org/)

[h5py](https://www.h5py.org/)

[sklearn](https://scikit-learn.org/stable/)

[scipy](https://www.scipy.org/)

[pyjet](https://github.com/scikit-hep/pyjet)


# Instructions

This repository uses <a href="https://link.springer.com/article/10.1140%2Fepjp%2Fs13360-020-00497-3" target="_blank">ABCNet</a> as the backbone to perform the training. Because of that, the data is expected to be stored in ```.h5``` files containing the following:

* **data**: [N,P,F], 
* **label**:[N,P]
* **pid**: [N]
* **global**: [N,G]

N = Number of events

F = Number of features per point

P = Number of points

G = Number of global features

For Unsupervised multiclass classification, only the **pid** is required, while for anomaly detection, **label** and **global** are required.

Scripts are provided to save the correct data format from the zenodo files. To save the multiclass classification files run

```bash
python prepare_data_multi.py --dir path/where/zenodo/file/are
```
while the unsupervised anomaly detection is done with

```bash
python prepare_data_unsup.py --dir path/where/zenodo/file/are [--RD]
```
Both scripts will save the output files under the golder ```h5```. 

# Training the unsupervided multiclass classification model

The training is performed by running:

```bash
python train_kmeans.py  --log_dir FOLDER_NAME 
```

where ,
* --log_dir: Name of the folder to create with the training results.

To further inspect the additional options, run the code with the option ```-h, --help```

To perform the evaluation step run:

```bash
python evaluate_kmeans.py --log_dir FOLDER_NAME --name FILE_NAME [--full_training]
```

where,
* --name: Name of the .h5 file saved with the evaluation results.
* --full_training: If set, load the full training, otherwise only the pre-training results are loaded

# Training the anomaly detection model

The procedure is identical as the previous steps, for the training, run:

```bash
python train_kmeans_seg.py --log_dir FOLDER_NAME --n_clusters=CLUSTER_SIZE [--RD] 
```

where,
* --n_clusters: Number of clusters to create
* --RD: Expect a data set containing also the true labels. Only used to assess the performance during training.

For the evaluation run:

```bash
python evaluate_kmeans_seg.py --log_dir FOLDER_NAME --name FILE_NAME --n_clusters=CLUSTER_SIZE --RD [--full_train]
```


# License

MIT License