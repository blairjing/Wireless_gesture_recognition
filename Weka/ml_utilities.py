import os
import csv
import glob


##########################################################################
#WEKA has trouble with feature names including quotes/spaces.
# Tidy up if needed (compatibility with some TS_Fresh generated features)
##########################################################################
def FixCSV(csvFile):
    f1 = open(csvFile + ".csv", 'r')
    f2 = open(csvFile + "_out.csv", 'w')
    for line in f1:
        f2.write(line.replace('\"', '').replace('2, 5, 10, 20','2-5-10-20'))
    f1.close()
    f2.close()

    #Remove temp csv file with characters unrecognized by WEKA
    os.remove(csvFile + ".csv")
    


##########################################################################
#Merge the .csv files for every configuration
# Creates a single dataset which can be directly used for cross-validation
##########################################################################
def MergeCSV(output_folder):
    if(os.path.exists(output_folder + '/full_dataset.csv')):
        os.remove(output_folder + '/full_dataset.csv')

    features_files = glob.glob(output_folder + "/*.csv") 

    print "Merging full dataset..."
    header_saved = False
    with open(output_folder + '/full_dataset.csv','wb') as fout:
        for filename in features_files:
            print "merging " + filename
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
    print "Dataset merged!"



