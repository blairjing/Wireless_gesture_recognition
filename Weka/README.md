## Scripts definition

The goal and functioning of the different scripts contained in this folder is summarily described below. Dependencies from third-party packages/applications can be found at the end of this document.

### Dataset-Generator Scripts

`extractor.py` - includes the code for managing feature selection and dataset creation. In it's main method, we can define the folder containing data to be analyzed and which kind of features we want to extract from that data (ex. `fprocessing.RmNoise(input/20170726_rmNoise')` declares we want to extract aggregate features from a sinewave which noise has been removed and which sample data can be found in `input/20170726_rmNoise` folder.

If the global variable `COMPUTE_DTW` is set to `True`, samples will be clustered by DTW and the resulting labels will be included as an extra feature on the final dataset.

Each individual data configuration will originate an accompanying .csv file with the features for each of the samples for that configuration. In its final step, `extractor.py` shall combine all these individual .csv into a unique .csv representing the full dataset (useful for later cross-validation). For generating ad-hoc training/testing sets from individual configurations' .csv , refer to `datasetGen.py`.

---

`f_processing.py` - includes the code for selecting features from input data (sinewaves / summarized data as doppler shifts, etc). Feature selection is implemented as a Template pattern - each new feature corresponds to a new Feature subclass. Default behavior is to read input data as timeseries and compute aggregate features from this data. Both behaviors can be changed by overriding `convertToFreshFormat()` and `generateFeatures()`.

---

`dtw_clustering.py` - includes code responsible for clustering training data. Data is clustered by distance - Euclidean/Dynamic Time Warping. Method `ClusterByDTW()` is called by `extractor.py`. There are currently 3 clustering approaches in use. Different methods for clustering can be seamlessly added. Majority of this code is now deprecated as computing DTW over non-summarized data incurs high overhead. More information can be found in the file comments.

---

`datasetGen.py` - Merges the features obtained for individual configurations for generating a Training and a Testing dataset. The samples to be used for building each dataset must be placed into user-defined "Training" and "Testing" folders.


### Supervised Classification Scripts

`ml.py` - run WEKA classifiers on datasets.

Provides 3 main methods for classification:

* `OnlineClassification()` allows for training a model with a given training dataset. Samples can then be fed to the model (either individually or in a batch) and classified in an online fashion.

* `ClassifyTestSet()` allows for training a model with a training dataset and use it to classify instances on a separate testing dataset.

* `CrossValidateFullDataset()` allows for training and testing a model with a single dataset by employing cross-validation.


### Unsupervised Clustering Scripts

The following scripts are stand-alone, i.e. they are not called on `extractor.py`. These are used to test the ability of different distance metrics on the clustering of our data.

`dwt_dtw.py` - This script reads an upper triangular matrix containing DTW distances computed from the Wavelet transform of data samples. The matrix is converted to its square form and fed to different clustering algorithms. *Method in use: DTW.*

---

`sax_clustering.py` - This script reads any data sample (raw, removed noise, doppler, etc) and computes a distance matrix using Symbolic Aggregate approXimation (SAX). The matrix is fed to different clustering methods. *Method in use: SAX.*


### Auxiliary Scripts

`ml_utilities.py` - contains auxiliary functions for building datasets. In particular, `MergeCSV()` is used for combining features of individual configurations into a larger dataset.

---

`plotter.py` - can be used to generate figures representing our base data (sinewaves, doppler shift, etc.)


## Package Dependencies

#### WEKA

Install the WEKA machine learning workbench from http://www.cs.waikato.ac.nz/ml/weka/downloading.html

#### python-weka-wrapper

Version: 0.3.10

Summary: Python wrapper for Weka using javabridge.

Home-page: http://pythonhosted.org/python-weka-wrapper

> Alternative WEKA wrapper:

> #### weka

> Version: 1.0.1

> Summary: A Python wrapper for the Weka data mining library.

> Home-page: https://github.com/chrisspen/weka

#### pandas

Version: 0.20.3

Summary: Powerful data structures for data analysis, time series,and statistics

Home-page: http://pandas.pydata.org


#### tsfresh

Version: 0.8.1

Summary: tsfresh extracts relevant characteristics from time series

Home-page: https://github.com/blue-yonder/tsfresh


#### python-pptx

Version: 0.6.6

Summary: Generate and manipulate Open XML PowerPoint (.pptx) files

Home-page: http://github.com/scanny/python-pptx


#### fastdtw

Version: 0.3.2

Summary: Python implementation of FastDTW, which is an approximate Dynamic Time Warping (DTW) algorithm that provides optimal or near-optimal alignments with an O(N) time and memory complexity.

Home-page: https://pypi.python.org/pypi/fastdtw


#### ucrdtw

Version: 0.201

Summary: Python extension for UCR Suite highly optimized subsequence search using Dynamic Time Warping (DTW)

Home-page: https://github.com/klon/ucrdtw


#### termcolor

Version: 1.1.0

Summary: ANSII Color formatting for output in terminal.

Home-page: http://pypi.python.org/pypi/termcolor


#### scipy

Version: 0.19.1

Summary: SciPy: Scientific Library for Python

Home-page: https://www.scipy.org

### statsmodel
Version: 0.8.0
Summary: Statistical computations and models for Python
Home-page: http://www.statsmodels.org/
