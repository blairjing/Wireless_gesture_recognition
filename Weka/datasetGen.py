#!/usr/bin/env python
#This program reads the path for two folders containing several extracted_features.csv
# The .csvs in the first folder are used to build the training dataset
# The .csvs in the second folder are used to build the testing dataset

#The idea is to allow the creation of segmented datasets for testing the generalization 
# ability of the models

import sys
import csv
import glob


def MergeGestureSamples(training_folder, testing_folder):
    #Generate training dataset
    training_files = glob.glob(training_folder + "/*.csv") 

    header_saved = False
    with open('Testbed/training_dataset.csv','wb') as fout:
        for filename in training_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)

    #Generate testing dataset
    testing_files = glob.glob(testing_folder + "/*.csv") 

    header_saved = False
    with open('Testbed/testing_dataset.csv','wb') as fout:
        for filename in testing_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)



if __name__ == "__main__":
    
    if(len(sys.argv) != 3):
        print 'Enter two folders: python datasetGen.py TrainingDataFolder TestingDataFolder'

    training_folder = sys.argv[1]
    testing_folder = sys.argv[2]


    MergeGestureSamples(training_folder, testing_folder)


