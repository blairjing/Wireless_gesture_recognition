from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters

import pandas

import os
from os.path import basename

##########################################################################
#Use this for handpicked feature extraction
#http://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html
#http://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
##########################################################################
fc_parameters = {
    "standard_deviation": None,
    "variance": None,
    "median": None,
    "mean": None,
    "skewness": None,
    "kurtosis": None,
    "quantile": [{"q": 0.25}, {"q": 0.75}],

    "maximum": None,
    "minimum": None,
    "first_location_of_maximum": None,
    "last_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_minimum": None,

    "sum_values": None,
    "abs_energy": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    
    "fft_coefficient": [{"coeff": coeff} for coeff in range(0, 2)],
}

#Abstract Feature class
class Feature:
    def __init__(self, samples_folder): 
        self.tags = None
        self.dataseries = None
        self.samples_folder = samples_folder
        self.fresh_folder = samples_folder + "_fresh/" 
        if not os.path.exists(self.fresh_folder):
            os.makedirs(self.fresh_folder)

    def _readSampleData(self,sampleFile):
        self.dataseries = pandas.read_table(self.fresh_folder + 'freshData_' + sampleFile + '.txt', delim_whitespace=True, names=self.tags)

    def convertToFreshFormat(self, sampleFile):
        if('freshData_' + sampleFile + '.txt' not in os.listdir(self.fresh_folder)):
            print "Converting " + self.samples_folder + "/" + sampleFile + " input file to tsFresh data format..."
            freshData = open(self.fresh_folder + '/freshData_' + sampleFile + '.txt',"w")
            
            with open(self.samples_folder + "/" + sampleFile) as f:
                sampleIndex = 0
                for line in f:
                    if(line != '\n'):
                        freshData.write(str(sampleIndex) + ' ' + line)
                    else:
                        sampleIndex += 1
                        freshData.write('\n')        
            freshData.close()

    def getTags(self):
        return self.tags

    def getSampleFolder(self):
        return self.samples_folder

    def getAnalysisFolder(self):
        return self.fresh_folder
    
    #Default:Perform aggregate feature generation over the whole timeseries. 
    # Other intended feature selection must be implemented in subclasses by overriding this method
    def generateFeatures(self, sampleFile):
        self._readSampleData(sampleFile)
        return impute(extract_features(self.dataseries, column_id=self.tags[0], default_fc_parameters=fc_parameters))



#Deals with feature generation for sliced timeseries.
# Timeseries are sliced into num_slices equal parts.
# Aggregate features are computed for each slice instead of for the whole timeseries
class SlicedFeature(Feature, object):
    def __init__(self, samples_folder):
        super(SlicedFeature, self).__init__(samples_folder)

    def generateFeatures(self, sampleFile):
        self._readSampleData(sampleFile)

        num_samples = self.dataseries[self.tags[0]].max() 
        datapoints_per_sample = self.dataseries[getattr(self.dataseries, self.tags[0]) == 0].shape[0]
        slice_number = self.num_slices
        slice_size = datapoints_per_sample / slice_number

        slices = []
        for i in range(0,slice_number): 
            newcol = [t + "_" + str(i) for t in self.tags]
            slice_data = pandas.DataFrame(columns=tuple(newcol))
            for n in range(0,num_samples+1):
                raw_slice = self.dataseries[getattr(self.dataseries, self.tags[0]) == n][i*slice_size:(i+1)*slice_size]
                raw_slice.columns = newcol
                slice_data = slice_data.append(raw_slice)
            slices.append(slice_data)

        processed_slices = []
        for n,s in enumerate(slices):
            processed_slices.append(impute(extract_features(s, column_id=self.tags[0] + '_' + str(n), default_fc_parameters=fc_parameters)))

        return pandas.concat(processed_slices,axis=1)



#Class for dealing with raw timeseries data
class RawData(Feature, object):
    def __init__(self, samples_folder):
        super(RawData, self).__init__(samples_folder)
        self.tags = ('raw_id', 'raw_Mic1', 'raw_Mic2', 'raw_Mic3', 'raw_Mic4')


#Class for dealing with raw timeseries data - sliced in different timespans
class RawDataSliced(SlicedFeature, object):
    def __init__(self, samples_folder):
        super(RawDataSliced, self).__init__(samples_folder)
        self.tags = ('raw_sliced_id', 'raw_sliced_Mic1', 'raw_sliced_Mic2', 'raw_sliced_Mic3', 'raw_sliced_Mic4')
        self.num_slices = 10



#Class for dealing with timeseries data without noise
class RmNoise(Feature, object):
    def __init__(self, samples_folder):
        super(RmNoise, self).__init__(samples_folder)
        self.tags = ('rmNoise_id', 'rmNoise_Mic1', 'rmNoise_Mic2', 'rmNoise_Mic3', 'rmNoise_Mic4')

#Class for dealing with timeseries data without noise
class RmNoiseCut(Feature, object):
    def __init__(self, samples_folder):
        super(RmNoiseCut, self).__init__(samples_folder)
        self.tags = ('rmNoiseCut_id', 'rmNoiseCut_Mic1', 'rmNoiseCut_Mic2', 'rmNoiseCut_Mic3', 'rmNoiseCut_Mic4')


#Class for dealing with raw timeseries data - sliced in different timespans
class RmNoiseSliced(SlicedFeature, object):
    def __init__(self, samples_folder):
        super(RmNoiseSliced, self).__init__(samples_folder)
        self.tags = ('rmNoise_sliced_id', 'rmNoise_sliced_Mic1', 'rmNoise_sliced_Mic2', 'rmNoise_sliced_Mic3', 'rmNoise_sliced_Mic4')
        self.num_slices = 10




#Class for dealing with doppler shift data
class Doppler(Feature, object):
    def __init__(self, samples_folder):
        super(Doppler, self).__init__(samples_folder)
        self.tags = ('doppler_id', 'doppler_Mic1', 'doppler_Mic2', 'doppler_Mic3', 'doppler_Mic4')



#Class for dealing with phase shift data 
class Phase(Feature, object):
    def __init__(self, samples_folder):
        super(Phase, self).__init__(samples_folder)
        self.tags = ('phase_id', 'phase_Mic1', 'phase_Mic2', 'phase_Mic3', 'phase_Mic4')
        


#Class for dealing with beamforming data 
class Beamform(Feature, object):
    def __init__(self, samples_folder):
        super(Beamform, self).__init__(samples_folder)
        
        beamform = ['beamform_id']
        for i in range (0,23):
            beamform.append('beamform_'+str(i))
        self.tags = tuple(beamform)
        


#Class for dealing with a snapshot of MUSIC data 
# (180 music coeffs for a single sample)
class MusicSnapshot(Feature, object):
    def __init__(self, samples_folder):
        super(MusicSnapshot, self).__init__(samples_folder)
        
        music = ['music_id']
        for i in range (0,181):
            music.append('music_'+str(i))
        self.tags = tuple(music)
        
    def generateFeatures(self, sampleFile):
        self._readSampleData(sampleFile)
        
        music_features = pandas.DataFrame(columns=self.tags[1:]) #create empty dataframe
        maximum = self.dataseries['music_id'].max() 
        for i in range(0,maximum+1):
            ts = self.dataseries
            coeffs = ts[ts.music_id == i].iloc[:,[1]].as_matrix()
            coeffs = coeffs.transpose()
            music_features.loc[music_features.shape[0]] = coeffs[0]
        
        return music_features



#Class for dealing with snapshots of MUSIC data over a temporal sliding window 
# (180 music coeffs x 180 series for a single sample)
class MusicSlidingSnapshotAggregate(Feature, object):
    def __init__(self, samples_folder):
        super(MusicSlidingSnapshotAggregate, self).__init__(samples_folder)
        
        music_sliding = ['music_sliding_id']
        for i in range (0,181):
            music_sliding.append('music_sliding'+str(i))
        self.tags = tuple(music_sliding)



#Class for dealing with MUSIC data in an aggregated fashion
class MusicTwoPeaksAggregate(Feature, object):
    def __init__(self, samples_folder):
        super(MusicTwoPeaksAggregate, self).__init__(samples_folder)
        
        music_agg = ['music_agg_id']
        for i in range (0,2): 
            music_agg.append('music_agg_'+str(i))
        self.tags = tuple(music_agg)

    #Basically the same as the superclass, replace NaNs by 360, an invalid value for angles in this setting
    def convertToFreshFormat(self, sampleFile):
        if('freshData_' + sampleFile + '.txt' not in os.listdir(self.fresh_folder)):
            print "Converting " + self.samples_folder + "/" + sampleFile + " input file to tsFresh data format..."
            freshData = open(self.fresh_folder + '/freshData_' + sampleFile + '.txt',"w")
            
            with open(self.samples_folder + "/" + sampleFile) as f:
                sampleIndex = 0
                for line in f:
                    if(line != '\n'):
                        freshData.write(str(sampleIndex) + ' ' + line.replace('NaN', '360'))
                    else:
                        sampleIndex += 1
                        freshData.write('\n')        
            freshData.close()




#Class for dealing with MUSIC angle of arrival at different times
class MusicAngles(Feature, object):
    def __init__(self, samples_folder):
        super(MusicAngles, self).__init__(samples_folder)
        
        music_angles = ['music_angles_id']
        for i in range (0,179):
            music_angles.append('music_angles_'+str(i))
        self.tags = tuple(music_angles)
    
    #Basically the same as the superclass, replace NaNs by 360, an invalid value for angles in this setting
    def convertToFreshFormat(self, sampleFile):
        if('freshData_' + sampleFile + '.txt' not in os.listdir(self.fresh_folder)):
            print "Converting " + self.samples_folder + "/" + sampleFile + " input file to tsFresh data format..."
            freshData = open(self.fresh_folder + '/freshData_' + sampleFile + '.txt',"w")
            
            with open(self.samples_folder + "/" + sampleFile) as f:
                sampleIndex = 0
                for line in f:
                    if(line != '\n'):
                        freshData.write(str(sampleIndex) + ' ' + line.replace('NaN', '360'))
                    else:
                        sampleIndex += 1
                        freshData.write('\n')        
            freshData.close()

    def generateFeatures(self, sampleFile):
        self._readSampleData(sampleFile)
        
        music_angles_features = pandas.DataFrame(columns=self.tags[1:]) #create empty dataframe
        music_angles_features2 = pandas.DataFrame(columns=[x + "_two" for x in self.tags[1:]]) #create empty dataframe
        
        maximum = self.dataseries['music_angles_id'].max() 
        for i in range(0,maximum+1):
            ts = self.dataseries
            #1st peak data
            coeffs = ts[ts.music_angles_id == i].iloc[:,[1]].as_matrix()
            coeffs = coeffs.transpose()
            music_angles_features.loc[music_angles_features.shape[0]] = coeffs[0]

            #2nd peak data
            coeffs = ts[ts.music_angles_id == i].iloc[:,[2]].as_matrix()
            coeffs = coeffs.transpose()
            music_angles_features2.loc[music_angles_features2.shape[0]] = coeffs[0]
        
        return pandas.concat([music_angles_features, music_angles_features2],axis=1)

    