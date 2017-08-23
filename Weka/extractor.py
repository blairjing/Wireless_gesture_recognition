import fprocessing
import ml_utilities as ml_util
import dtw_clustering

import pandas
import os
from os.path import basename

output_folder = "output"
CSV_COLUMN_ORDER = []
COMPUTE_DTW = False
start = 0

def getSampleGroundTruth(sampleFile):
    label = None
    if("swipeup" in sampleFile):
        label = "SwipeUp"
    elif("swipedown" in sampleFile):
        label = "SwipeDown"
    elif("swipeleft" in sampleFile):
        label = "SwipeLeft"
    elif("swiperight" in sampleFile):
        label = "SwipeRight"
    elif("push" in sampleFile):
        label = "Push"
    elif("pull" in sampleFile):
        label = "Pull"
    elif("bloom" in sampleFile):
        label = "Bloom"
    elif("click" in sampleFile):
        label = "Click"
    elif("point" in sampleFile):
        label = "Point"
    elif("swipe_" in sampleFile):
        label = "Swipe"
    elif("poke" in sampleFile):
        label = "Poke"
    elif("attention" in sampleFile):
        label = "Attention"
    return label

def addClusteringLabels(features, cluster_labels, cluster_features):
    #Add cluster labels to feature set if requested
    if(COMPUTE_DTW):
        global start
        maximum = features.shape[0]

        cluster_methods_results = []
        for clustering_method in cluster_labels:
            f_result = []
            for k in clustering_method.keys():
                print clustering_method[k][start:start+maximum]
                f_result.append(pandas.Series(clustering_method[k][start:start+maximum]))
            cluster_methods_results.append(f_result)

        start += maximum

        print "Max: " + str(maximum)
        print "CLUSTER LABELS TOTAL " + str(len(cluster_labels[0][cluster_labels[0].keys()[0]]))

        for i, r in enumerate(cluster_methods_results):
            for n, result in enumerate(r):
                #print str(cluster_features[i] + "_" + cluster_labels[0].keys()[n%len(cluster_labels[0].keys())])
                features["cluster_" + cluster_features[i] + "_" + cluster_labels[0].keys()[n%len(cluster_labels[0].keys())]] = result.values

    return features


def ExtractFeatures(sampleFile, features_data, cluster_labels, cluster_features):

    extracted_features = []

    for f_d in features_data:
        extracted_features.append(f_d.generateFeatures(sampleFile))


    #Combine all extracted features    
    features = pandas.concat(extracted_features,axis=1)

    #Combine clustering labels to use as feature (Only if COMPUTE_DTW == True)
    features = addClusteringLabels(features, cluster_labels, cluster_features)

    #Assign class label to each sample  
    label = getSampleGroundTruth(sampleFile)

    #Add class label
    features['class'] = label
    

    #Set the csv header order to be the same among all individual .csv datasets
    global CSV_COLUMN_ORDER
    if(len(CSV_COLUMN_ORDER) == 0):
        print "Column order has been set"
        CSV_COLUMN_ORDER = features.columns.values 
    features = features[CSV_COLUMN_ORDER]
    
    #Save extracted features to CSV file
    global output_folder
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    features.to_csv(output_folder + "/extractedFeatures_" + sampleFile + ".csv", sep=',')



if __name__ == "__main__":
    ###############################################################
    #Set data to extract features from
    ###############################################################
    temp_dir='input/'

    features = [#fprocessing.RawData('/Volumes/TessierAshpool/Features_0803/20170803_raw'),
                fprocessing.RawData(temp_dir+'20170726_raw'),
                fprocessing.RmNoise(temp_dir+'20170726_rmNoise'),
                #fprocessing.RmNoiseCut('Samples/20170726_rmNoiseCut_fresh'),
                fprocessing.Doppler(temp_dir+'20170726_moredoppler'),
                fprocessing.MusicAngles(temp_dir+'20170726_musicAngles'),
                fprocessing.MusicTwoPeaksAggregate(temp_dir+'20170726_musicAngles'),
                fprocessing.MusicSlidingSnapshotAggregate(temp_dir+'20170726_musicSlidingSnapshot'),
                ]

    ###############################################################
    #Generate tsFresh-formatted dataseries for all samples
    ###############################################################

    sampleFiles = []
    for f in os.listdir(features[0].getSampleFolder()):
        if 'DS' in str(f):
            print str(sampleFiles)+'nooooo'
            continue
        sampleFiles.append(f)

    print '--------------\nsampleFiles:'
    print sampleFiles
    print '----------------------'
    for sampleFile in sampleFiles:
        if 'DS' in str(sampleFile):
            print str(sampleFile)+' exist in '+str(sampleFiles)
            continue
        for feature in features:
            feature.convertToFreshFormat(sampleFile)


    ##################################################################
    #Cluster samples by DTW distance. Use resulting labels as features
    ##################################################################
    cluster_labels = []
    cluster_features = []
    if(COMPUTE_DTW):
        print "======================================================"
        print "Starting DTW clustering"
        print "======================================================"
        #Add here the data features which should be clustered
        #This procedure is executed before generating .csv files for each configuration
        # All samples must be clustered beforehand

        cluster_labels.append(dtw_clustering.ClusterByDTW(features[0],sampleFiles))
        cluster_features.append(type(features[0]).__name__)

        #Doppler
        #cluster_labels.append(dtw_clustering.ClusterByDTW(features[2],sampleFiles))
        #cluster_features.append(type(features[2]).__name__)

    ###############################################################
    #Compute features for each sample
    ###############################################################
    for i, sampleFile in enumerate(sampleFiles):
        print "Computing features for sample " + str(i) + ": " + sampleFile
        ExtractFeatures(sampleFile, features, cluster_labels, cluster_features)
    

    ###############################################################
    #Merge each individual sample features into the full dataset
    ###############################################################
    ml_util.MergeCSV(output_folder)





