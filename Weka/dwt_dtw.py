import numpy as np

from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.vq import kmeans2, vq, whiten
from scipy.spatial.distance import euclidean, pdist, squareform
from termcolor import colored

def FeedDistanceMatrix():
    dtw_file = "/home/dmbb/Documents/GestureRecog/dtw/diogo_test_swipe_45cm_45.txt"
    #The original file contains data for filling an upper triangular distance matrix
    dwt_dtw_matrix = np.loadtxt(dtw_file)

    #Build a square matrix
    max_sample = int(np.max(dwt_dtw_matrix,axis=0)[0])
    matrix = np.empty((max_sample+1, max_sample+1), dtype=object)

    #Fill the upper triangle of the square distance matrix
    for l in dwt_dtw_matrix:
        mics = []
        mics.append(l[2])
        mics.append(l[3])
        mics.append(l[4])
        mics.append(l[5])
        matrix[int(l[0])-1,int(l[1])-1] = mics

    #Set matrix diagonal to zeros (comparisons of the same samples)
    for n in range(0,max_sample+1):
        matrix[n,n] = [0,0,0,0]

    #Make the square matrix symmetric
    i_lower = np.tril_indices(max_sample+1, -1)
    matrix[i_lower] = matrix.T[i_lower]

    #Convert numpy array to python list
    matrix = matrix.tolist()


    # Turn each mic_dtw vector into individual features
    # Clustering algorithms can only receive matrixes at most rank = 2
    flat_matrix = []
    for row in matrix:
        features = []
        for mic_values in row:
            for value in mic_values:
                features.append(value)
        flat_matrix.append(features)

    return flat_matrix


#########################################################
#Clusters samples based on DTW distance-matrix
# Returns the label of each sample
#########################################################
def ClusterByDTW():

    # Feed DiscreteWaveletTransform DTW matrix built by Mei
    X = FeedDistanceMatrix()

    ward = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
    ward_labels = ward.labels_
    #print "Agglomerative-ward linkage clusters - "
    #print(ward_labels)


    complete = AgglomerativeClustering(n_clusters=4, linkage='complete').fit(X)
    complete_labels = complete.labels_
    #print "Agglomerative-complete linkage clusters - "
    #print(complete_labels)


    whitened = whiten(X)
    centroids, kmeans_labels = kmeans2(whitened, 4, minit='random')
    #print "K-means clusters - "
    #print(kmeans_labels)

    print "======================================================"

    return {'Agglomerative_ward linkage': ward_labels, 'Agglomerative_complete linkage': complete_labels, 'K-means': kmeans_labels}



def PrintResults(labels):
    ##########################################################################################
    #TODO: These should be merged & simplified. For instance:
    # diogo_test_bloom_15cm_0.txt, 10
    # diogo_test_bloom_15cm_20.txt, 11
    #...

    configuration_list = "/home/dmbb/Documents/GestureRecog/dtw/all_txt_files-20170726.txt"
    samples_per_config = "/home/dmbb/Documents/GestureRecog/dtw/diogo_dtw_label.txt"

    ##########################################################################################
    #Retrieve number of samples from each configuration
    n_samples = []
    with open(samples_per_config) as f:
        i = 5
        for n,line in enumerate(f):
            if(n==i):
                s =''.join(d for d in line if d.isdigit())
                n_samples.append(s)
                i += 7

    #We are just interested in analysing "diogo" configurations in this example
    for k in labels:
        print colored(k,"red")
        with open(configuration_list) as f:
            for n, line in enumerate(f):
                if("diogo" in line):
                    print line.rstrip() + "\t" + str(labels[k][:int(n_samples[n])])
                    labels[k] = labels[k][int(n_samples[n]):]
        print "#################################################\n"



if __name__ == "__main__":
    labels = ClusterByDTW()
    PrintResults(labels)
