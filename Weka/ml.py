import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.experiments import SimpleCrossValidationExperiment
from weka.experiments import Tester, ResultMatrix
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.filters import Filter

from itertools import chain, combinations
from termcolor import colored

def OnlineClassification():
	#Classifies instances in an online way.
	#TODO: This is just a simple example of how online learning can be automated w/ WEKA
	# May be useful to later stages of the project.

	data_dir = "Testbed/"

	training = converters.load_any_file(data_dir + "training_dataset.csv")
	training.class_is_last()

	testing = converters.load_any_file(data_dir + "testing_dataset.csv")
	testing.class_is_last()

	a = open(data_dir +"testing_dataset.csv", "r") 
	print len(a.readlines())


	cls_classes = ["weka.classifiers.trees.J48",
					"weka.classifiers.trees.RandomForest",
					"weka.classifiers.lazy.IBk"]

	classifiers = []
	for cls in cls_classes:
		classifiers.append(Classifier(classname=cls))


	#Set class attribute

	print colored("======================================================",'green')
	print colored("Experiment for dataset",'green')
	print colored("======================================================",'green')


	for i, cls in enumerate(classifiers):
		cls.build_classifier(training)

		print("# - actual - predicted - right - distribution")
		for index, inst in enumerate(testing):
			pred = cls.classify_instance(inst)
			dist = cls.distribution_for_instance(inst)
			print(
				"%d - %s - %s - %s  - %s" %
				(index+1,
					inst.get_string_value(inst.class_index),
					inst.class_attribute.value(int(pred)),
					"yes" if pred == inst.get_value(inst.class_index) else "no",
					str(dist.tolist())))


def FilterAttribute(attribute, data):
	#Filter dataset
	remove = Filter(classname="weka.filters.unsupervised.attribute.RemoveByName", options=["-E", attribute])
	remove.inputformat(data)
	return remove.filter(data)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def ClassifyTestSet():
	#Tests a classifier performance with a dedicated test set
	# Models are evaluated for different combinations of features
	# Several classifiers may be used 

	# Load Datasets
	data_dir = "Testbed/"

	#h=open(data_dir+"training_dataset.csv","rb")

	#print h
	a = open(data_dir +"training_dataset.csv", "r") 
	print len(a.readlines())
	a = open(data_dir +"testing_dataset.csv", "r") 
	print len(a.readlines())

	training = converters.load_any_file(data_dir+"training_dataset.csv")
	training.class_is_last()

	testing = converters.load_any_file(data_dir +"testing_dataset.csv")
	testing.class_is_last() #set class attribute to be the last one listed


	#Choose classifiers to use
	cls_classes = ["weka.classifiers.trees.RandomForest",
					"weka.classifiers.trees.J48",
					"weka.classifiers.lazy.IBk"
					]

	classifiers = []
	for cls in cls_classes:
		classifiers.append(Classifier(classname=cls))

	
	#Regex for attribute selection
	#(Useful for testing different combinations of attributes)
	identifier_att = ".*id.*"
	timeseries_att = "Mic.*"
	doppler_att = "doppler.*"
	phase_att = "phase.*"
	music_att = "music.*"
	beamform_att = "beamform.*"
	att_set = [timeseries_att, doppler_att, phase_att, music_att, beamform_att]

	##################################################
	#Remove instances identifier attribute
	training = FilterAttribute(identifier_att,training)
	testing = FilterAttribute(identifier_att,testing)
	################################################


	for att_comb in powerset(att_set):
		training_filtered = training
		testing_filtered = testing

		for att in att_comb:
			if(len(att) != len(att_set)):
				training_filtered = FilterAttribute(att,training_filtered)
				testing_filtered  = FilterAttribute(att,testing_filtered)

		print colored("======================================================",'green')
		print colored("Full attribute set: " + str(att_set),'green')
		print colored("Removed attributes: " + str(att_comb),'green')
		print colored("======================================================",'green')


		for i, cls in enumerate(classifiers):
			cls.build_classifier(training_filtered)

			evl = Evaluation(training)
			evl.test_model(cls, testing_filtered)


			print colored("=> Testing for " + cls_classes[i], 'red')
			print(evl.summary())
			print(evl.matrix())



def CrossValidateFullDataset():
	#Tests a classifier performance with 10x cross-validation

	data_dir = "test/"
	print "Loading Dataset..."
	data = converters.load_any_file(data_dir + "full_dataset.csv")
	print "Dataset Loaded!"
	
	#Set class attribute
	data.class_is_last()


	cls_classes = [#"weka.classifiers.trees.J48",
					"weka.classifiers.trees.RandomForest",
					#"weka.classifiers.lazy.IBk"
				]

	classifiers = []
	for cls in cls_classes:
		classifiers.append(Classifier(classname=cls))


	#Regex for attribute selection
	#(Useful for testing different combinations of attributes)
	identifier_att = ".*id.*"
	
	#timeseries_att = "raw.*"
	rmNoise_att = "rmNoise.*"
	#doppler_att = "doppler.*"
	#phase_att = "phase.*"
	#music_att = "music.*"
	#beamform_att = "beamform.*"
	#music_sliding_att = "music_sliding.*"
	#music_agg_att = "music_agg.*"
	#music_angles_att = "music_angles.*"

	att_set = [rmNoise_att]

	##################################################
	#Remove instances identifier attribute
	data = FilterAttribute(identifier_att,data)
	################################################


	for att_comb in powerset(att_set):
		data_filtered = data

		for att in att_comb:
			if(len(att) != len(att_set)):
				data_filtered = FilterAttribute(att,data_filtered)
		if str(list(set(att_set) - set(att_comb)))=='[]':
			continue
		print att_set
		print att_comb
		print colored("======================================================",'green')
		print colored("Full attribute set: " + str(att_set),'green')
		print colored("Removed attributes: " + str(att_comb),'red')
		if(len(att_comb) > 0):
			print colored("Using attributes: " + str(list(set(att_set) - set(att_comb))), 'green')
		print colored("======================================================",'green')

		print data_dir

		for i, cls in enumerate(classifiers):

			evl = Evaluation(data_filtered)
			evl.crossvalidate_model(cls, data_filtered, 10, Random(1))

			print colored("=> 10x cross-validation for " + cls_classes[i], 'red')
			print(evl.summary())
			print(evl.matrix())


if __name__ == "__main__":
	#Start WEKA execution
	jvm.start(max_heap_size="4096m")
	#OnlineClassification()
	CrossValidateFullDataset()
	#ClassifyTestSet()
	#Stop WEKA execution
	print "over"
	jvm.stop()
