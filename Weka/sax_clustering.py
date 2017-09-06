import numpy as np

import fprocessing
import ml_utilities as ml_util

from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.vq import kmeans2, vq, whiten
from scipy.spatial.distance import euclidean, pdist, squareform

import pandas
import os
from os.path import basename

from saxpy import SAX

def ComputeSax(sample_data, sample_data2):

    sample_data = sample_data.as_matrix()
    sample_data2 = sample_data2.as_matrix()

    #########################################
    # SAX - Symbolic aggregate approximation
    #http://www.cs.ucr.edu/~eamonn/SAX.pdf
    ##########################################
    #PARAMETERS:
    #W: The number of PAA segments representing the time series - aka the len()
    # of the string representing the timeseries - useful for dimensionality reduction
    #Alphabet size: Alphabet size (e.g., for the alphabet = {a,b,c} = 3)

    downsample_ratio = 200
    word_length = len(sample_data[:,1]) / downsample_ratio
    alphabet_size = 7

    s = SAX(word_length, alphabet_size)

    mic_distances = []
    for mic in range(1,5):
        (x1String, x1Indices) = s.to_letter_rep(sample_data[:,mic])
        (x2String, x2Indices) = s.to_letter_rep(sample_data2[:,mic])

        #print x1String

        x1x2ComparisonScore = s.compare_strings(x1String, x2String)

        #Print results
        mic_distances.append(x1x2ComparisonScore)
        print "Mic: " + str(mic) + ", distance= " + str(x1x2ComparisonScore)
    return mic_distances

#TODO: Just compute the upper triangular and generate the square matrix afterwards
def ComputeSimilarity_All_vs_All(feature, sampleFiles):
    SAX_distance_matrix = [] #row = sample, column = distance
    series = []

    print "Reading timeseries from disk..."
    for sampleFile in sampleFiles:
        series.append(pandas.read_table(feature.getAnalysisFolder() + '/freshData_' + sampleFile + '.txt', delim_whitespace=True, names=feature.getTags()))

    for n, sampleFile in enumerate(sampleFiles):
        timeseries = series[n]

        maximum = timeseries[feature.getTags()[0]].max()


        for i in range(0,maximum+1):
            sample_distances = []
            sample_data = timeseries[getattr(timeseries, feature.getTags()[0]) == i]

            for n2, sampleFile2 in enumerate(sampleFiles):
                timeseries2 = series[n2]
                maximum2 = timeseries2[feature.getTags()[0]].max()

                for i2 in range(0,maximum2+1):
                    sample_data2 = timeseries2[getattr(timeseries2, feature.getTags()[0]) == i2]

                    print "Computing " + type(feature).__name__ + " SAX between " + sampleFile + ": " + str(i) + " and " + sampleFile2 + ": " + str(i2)
                    mic_distances = ComputeSax(sample_data,sample_data2)
                    sample_distances.append(mic_distances[0])
                    sample_distances.append(mic_distances[1])
                    sample_distances.append(mic_distances[2])
                    sample_distances.append(mic_distances[3])
            SAX_distance_matrix.append(sample_distances)

    return SAX_distance_matrix



def ClusterBySAX(feature, sampleFiles):

    distance_matrix = ComputeSimilarity_All_vs_All(feature,sampleFiles)

    ward = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(distance_matrix)
    ward_labels = ward.labels_
    print "Agglomerative-ward linkage clusters - " + type(feature).__name__
    print(ward_labels)


    complete = AgglomerativeClustering(n_clusters=4, linkage='complete').fit(distance_matrix)
    complete_labels = complete.labels_
    print "Agglomerative-complete linkage clusters - " + type(feature).__name__
    print(complete_labels)


    whitened = whiten(distance_matrix)
    centroids, kmeans_labels = kmeans2(whitened, 4, minit='random')
    print "K-means clusters - " + type(feature).__name__
    print(kmeans_labels)

    print "======================================================"

    clusterResults = {'Agg_ward': ward_labels, 'Agg_comp': complete_labels, 'K-means': kmeans_labels}
    return clusterResults


if __name__ == "__main__":

    temp_dir='/media/dmbb/TessierAshpool/Samples/'

    features = [fprocessing.RawData(temp_dir+'20170726_raw'),
                fprocessing.RmNoise(temp_dir+'20170726_rmNoise'),
                fprocessing.Doppler(temp_dir+'20170726_moredoppler'),
                ]


    sampleFiles = []
    for f in os.listdir(features[0].getSampleFolder()):
        if 'DS' in str(f):
            print str(sampleFiles)+' nooooo'
            continue

        #Account just for data from one person
        if 'diogo' in str(f):
            sampleFiles.append(f)

    print '--------------\nsampleFiles:'
    print sampleFiles
    print '----------------------'
    for sampleFile in sampleFiles:
        if 'DS' in str(sampleFile):
            print str(sampleFile)+' exist in '+ str(sampleFiles)
            continue
        for feature in features:
            feature.convertToFreshFormat(sampleFile)


    ClusterBySAX(features[0], sampleFiles)
