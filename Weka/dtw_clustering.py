import numpy as np
from numpy.random import rand

from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.vq import kmeans2, vq, whiten
from scipy.spatial.distance import euclidean, pdist, squareform

from fastdtw import dtw
import _ucrdtw

import pandas

import fprocessing



#########################################
#Computes DTW distance between two series
#########################################
def ComputeDTW(sample_data, sample_data2):
    #Read tsFresh tables for all gestures
    sample_data = sample_data.as_matrix()
    sample_data2 = sample_data2.as_matrix()
        

    mic_distances = []
    for mic in range(1,5):
        #distance, path = dtw(sample_data[:,mic], sample_data2[:,mic], dist=euclidean)
        loc, distance = _ucrdtw.ucrdtw(stats.zscore(sample_data[:,mic]), stats.zscore(sample_data2[:,mic]), 0.05) 

        #Print results
        mic_distances.append(distance)
        print "Mic: " + str(mic) + ", distance= " + str(distance)

    return mic_distances

###############################################
#Computes Euclidean distance between two series
##############################################
def ComputeEuclidean(sample_data, sample_data2):
    #Read tsFresh tables for all gestures
    sample_data = sample_data.as_matrix()
    sample_data2 = sample_data2.as_matrix()
    
    
    mic_distances = []
    for mic in range(1,2):
        #Adjust timeseries size - cut by smaller
        sample_data_size = len(sample_data[:,mic])
        sample2_data_size = len(sample_data2[:,mic])

        cutoff = min(sample_data_size, sample2_data_size)
        distance = euclidean(sample_data[:cutoff,mic], sample_data2[:cutoff,mic]) 

        #Print results
        mic_distances.append(distance)
        print "Mic: " + str(mic) + ", distance= " + str(distance)

    return mic_distances

##############################################################################
#Computes similarity matrix between each sample to one reference configuration
# Each sample is compared to a reference of a gesture performed by a different 
#  person and configuration (less costly than All vs All)

#On this sliced version, DTW is computed for slices of the timeseries data #TODO
##############################################################################
def ComputeSlicedSimilarity_All_vs_Reference(feature,sampleFiles):
    slice_number = 3
    distance_matrixes = []
     
    series = []
    print "Reading timeseries from disk..."
    for sampleFile in sampleFiles:
        series.append(pandas.read_table(feature.getAnalysisFolder() + '/freshData_' + sampleFile + '.txt', delim_whitespace=True, names=feature.getTags()))

    

    dtw_distance_matrix = [] #row = sample, column = distance

    for n, sampleFile in enumerate(sampleFiles):
        timeseries = series[n]
        
        num_samples = timeseries[feature.getTags()[0]].max()
        datapoints_per_sample = timeseries[getattr(timeseries, feature.getTags()[0]) == 0].shape[0]
        slice_size = datapoints_per_sample / slice_number

        dtw_distance_matrix = [] # cell = "slice_number" distance matrixes

        for i in range(0,slice_number): 
            slice_distance_matrix = []

            for n in range(0,num_samples):#maximum+1
                sample_distances = []
                sample_data = timeseries[getattr(timeseries, feature.getTags()[0]) == i][i*slice_size:(i+1)*slice_size]

                for n2, sampleFile2 in enumerate(sampleFiles):
                    timeseries2 = series[n2]

                    for i2 in range(0,1):#maximum2+1
                        sample_data2 = timeseries2[getattr(timeseries2, feature.getTags()[0]) == i2][i*slice_size:(i+1)*slice_size]
                        print "Computing " + type(feature).__name__ + " DTW between " + sampleFile + ": " + str(i) + " and " + sampleFile2 + ": " + str(i2) + " - Slice: " + str(i)
                        mic_distances = ComputeDTW(sample_data,sample_data2)
                        sample_distances.append(mic_distances[0])
                        sample_distances.append(mic_distances[1])
                        sample_distances.append(mic_distances[2])
                        sample_distances.append(mic_distances[3])
                slice_distance_matrix.append(sample_distances) #Append a slice
            dtw_distance_matrix.append(slice_distance_matrix) #Append all slices

    return dtw_distance_matrix


#########################################################
#Clusters samples based on DTW distance-matrix
# Returns the label of each sample to use it as a feature #TODO
#########################################################
def ClusterBySlicedDTW(feature, sampleFiles):
    #Debugging Purposes
    #X = np.random.rand(20,20)

    X = ComputeSlicedSimilarity_All_vs_Reference(feature,sampleFiles)
    #X = ComputeSlicedSimilarity_All_vs_All(feature,sampleFiles)

    slices = []



    ward = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
    ward_labels = ward.labels_
    print "Agglomerative-ward linkage clusters - " + type(feature).__name__ 
    print(ward_labels)


    complete = AgglomerativeClustering(n_clusters=4, linkage='complete').fit(X)
    complete_labels = complete.labels_
    print "Agglomerative-complete linkage clusters - " + type(feature).__name__ 
    print(complete_labels)


    whitened = whiten(X)
    centroids, kmeans_labels = kmeans2(whitened, 4, minit='points')
    print "K-means clusters - " + type(feature).__name__ 
    print(kmeans_labels)

    print "======================================================"

    clusterResults = {'Agg_ward': ward_labels, 'Agg_comp': complete_labels, 'K-means': kmeans_labels}
    return clusterResults



##############################################################################
#Computes similarity matrix between each sample to one reference configuration
# Each sample is compared to a reference of a gesture performed by a different 
#  person and configuration (less costly than All vs All)
##############################################################################
def ComputeSimilarity_All_vs_Reference(feature,sampleFiles):
    dtw_distance_matrix = [] #row = sample, column = distance  
    series = []

    print "Reading timeseries from disk..."
    for sampleFile in sampleFiles:
        series.append(pandas.read_table(feature.getAnalysisFolder() + '/freshData_' + sampleFile + '.txt', delim_whitespace=True, names=feature.getTags()))

    for n, sampleFile in enumerate(sampleFiles):
        timeseries = series[n]
        
        maximum = timeseries[feature.getTags()[0]].max()


        for i in range(0,maximum+1):#maximum+1
            sample_distances = []
            sample_data = timeseries[getattr(timeseries, feature.getTags()[0]) == i]#.groupby(lambda x: x/8).mean() #8x downsample

            for n2, sampleFile2 in enumerate(sampleFiles):
                #if("diogo" not in sampleFile2):
                #    continue
                timeseries2 = series[n2]

                for i2 in range(0,1):#maximum2+1
                    sample_data2 = timeseries2[getattr(timeseries2, feature.getTags()[0]) == i2]#.groupby(lambda x: x/8).mean() # 8x downsample

                    print "Computing " + type(feature).__name__ + " DTW between " + sampleFile + ": " + str(i) + " and " + sampleFile2 + ": " + str(i2)
                    mic_distances = ComputeEuclidean(sample_data,sample_data2)
                    sample_distances.append(mic_distances[0])
                    #sample_distances.append(mic_distances[1])
                    #sample_distances.append(mic_distances[2])
                    #sample_distances.append(mic_distances[3])
            dtw_distance_matrix.append(sample_distances)

    return dtw_distance_matrix


#########################################################################
#Computes similarity matrix between all samples
# Each sample is compared to each and every other sample from different
#  gestures and configurations (Cost is probably prohibitive)
#########################################################################
def ComputeSimilarity_All_vs_All(feature,sampleFiles):
    dtw_distance_matrix = [] #row = sample, column = distance  
    series = []

    print "Reading timeseries from disk..."
    for sampleFile in sampleFiles:
        series.append(pandas.read_table(feature.getAnalysisFolder() + '/freshData_' + sampleFile + '.txt', delim_whitespace=True, names=feature.getTags()))

    for n, sampleFile in enumerate(sampleFiles):
        timeseries = series[n]
        
        maximum = timeseries[feature.getTags()[0]].max()


        for i in range(0,maximum):#maximum+1
            sample_distances = []
            sample_data = timeseries[getattr(timeseries, feature.getTags()[0]) == i]

            for n2, sampleFile2 in enumerate(sampleFiles):
                timeseries2 = series[n2]
                maximum2 = timeseries2[feature.getTags()[0]].max()

                for i2 in range(0,maximum2):#maximum2+1
                    sample_data2 = timeseries2[getattr(timeseries2, feature.getTags()[0]) == i2]

                    print "Computing " + type(feature).__name__ + " DTW between " + sampleFile + ": " + str(i) + " and " + sampleFile2 + ": " + str(i2)
                    mic_distances = ComputeDTW(sample_data,sample_data2)
                    sample_distances.append(mic_distances[0])
                    sample_distances.append(mic_distances[1])
                    sample_distances.append(mic_distances[2])
                    sample_distances.append(mic_distances[3])
            dtw_distance_matrix.append(sample_distances)

    return dtw_distance_matrix

def FeedDistanceMatrix(): #TODO - Just a placeholder
    matrix = np.loadtxt("distance_matrix.txt")
    print matrix
    
    return matrix

#########################################################
#Clusters samples based on DTW distance-matrix
# Returns the label of each sample to use it as a feature
#########################################################
def ClusterByDTW(feature, sampleFiles):
    #Debugging Purposes
    #X = np.random.rand(20,20)

    #X = ComputeSimilarity_All_vs_Reference(feature,sampleFiles)
    #X = ComputeSimilarity_All_vs_All(feature,sampleFiles)

    # Feed DiscreteWaveletTransform DTW matrix built by Mei
    X = FeedDistanceMatrix()

    ward = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
    ward_labels = ward.labels_
    print "Agglomerative-ward linkage clusters - " + type(feature).__name__ 
    print(ward_labels)


    complete = AgglomerativeClustering(n_clusters=4, linkage='complete').fit(X)
    complete_labels = complete.labels_
    print "Agglomerative-complete linkage clusters - " + type(feature).__name__ 
    print(complete_labels)


    whitened = whiten(X)
    centroids, kmeans_labels = kmeans2(whitened, 4, minit='random')
    print "K-means clusters - " + type(feature).__name__ 
    print(kmeans_labels)

    print "======================================================"

    clusterResults = {'Agg_ward': ward_labels, 'Agg_comp': complete_labels, 'K-means': kmeans_labels}
    return clusterResults




