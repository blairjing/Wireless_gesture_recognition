#!/usr/bin/env python
#
# plotTimeseries() plots a figure with the raw sinewave alongside its points' amplitude CDF & the Doppler Shift. 
#   It can present n_samples of a given gesture at a time.
# plotMusic() plots a figure with the MUSIC coefficients of n_samples.

import os
import pandas
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

n_samples = 3
colors=['red','green','black','blue','teal','purple','orange','grey']
alphas = [1,0.8,0.6,0.4,0.2]


music = ['music_id', 'coeff']
musicFolder = "/Volumes/TessierAshpool/Samples/20170726_music_fresh/"
rawFolder = "/Volumes/TessierAshpool/Samples/20170726_raw_fresh/"
dopplerFolder = "/Volumes/TessierAshpool/Samples/20170726_moredoppler_fresh/"
rmNoiseFolder = "/Volumes/TessierAshpool/Samples/20170726_rmNoise_fresh/"
rmNoiseCutFolder = "/Volumes/TessierAshpool/Samples/20170726_rmNoiseCut_fresh/"


def plotMusic(f_name):
	fig = plt.figure()
	for i in range(0,n_samples):
		d = pandas.read_table(musicFolder + f_name, delim_whitespace=True, names=tuple(music))
		f_name = f_name[15:].replace('.txt', '') #shorten sample name for visualization
		d = d[d.music_id == i]['coeff']
		plt.plot(np.linspace(0,len(d),len(d)),d, color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)

		plt.xlabel('MUSIC coefficient', fontsize = 'large')
		plt.ylabel('Coeff value', fontsize = 'large')
		plt.legend(loc='upper right', fontsize='medium')
		plt.title('MUSIC coefficients - ' + f_name)
	
	#plt.show() #for debugging purposes
	print "Plotting " + f_name + " MUSIC"
	fig.savefig('/Volumes/TessierAshpool/Plots/Music/' + f_name + '.png')   # save the figure to file
	plt.close(fig)



def plotTimeseries(f_name):
	timeseries = pandas.read_table(rmNoiseFolder + f_name, delim_whitespace=True, names=('id','Mic1', 'Mic2', 'Mic3', 'Mic4'))
	doppler = pandas.read_table(dopplerFolder + f_name, delim_whitespace=True, names=('id','Mic1', 'Mic2', 'Mic3', 'Mic4'))
	f_name = f_name[15:].replace('.txt', '')
	fig, axarr = plt.subplots(4, 3,figsize=(15,8))
	fig.suptitle("Sinewave amplitude CDF - " + f_name, fontsize=14)

	################## LOAD DATA FOR 1st ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries[timeseries.id == i]['Mic1']
		n, base = np.histogram(data,bins=50)

		cumulative = np.cumsum(n)
		cumulative = cumulative/float(max(cumulative))
		axarr[0,0].plot(base[:-1], cumulative, color=colors[i%len(colors)],label="Sample " + str(i))

		axarr[0,0].legend(loc='best')
		axarr[0,0].set_ylim([0,1])
		axarr[0,0].set_xlabel('Sinewave amplitude', fontsize='medium')
		axarr[0,0].set_ylabel('CDF', fontsize='medium')
		axarr[0,0].legend(loc='lower right', fontsize='medium')
		axarr[0,0].set_title('Sinewave Amplitude CDF',fontsize='large')
		axarr[0,0].set_xlim([-0.026,0.026])


		axarr[0,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[0,1].set_title('Sinewave Amplitude Plot',fontsize='large')
		axarr[0,1].set_ylim([-0.026,0.026])

		data = doppler[doppler.id == i]['Mic1']
		n, base = np.histogram(data,bins=50)
		axarr[0,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[0,2].set_title('Doppler Shift Plot',fontsize='large')

	################## LOAD DATA FOR 2nd ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries[timeseries.id == i]['Mic2']
		n, base = np.histogram(data,bins=50)

		cumulative = np.cumsum(n)
		cumulative = cumulative/float(max(cumulative))
		axarr[1,0].plot(base[:-1], cumulative, color=colors[i%len(colors)],label="Sample " + str(i))

		axarr[1,0].legend(loc='best')
		axarr[1,0].set_ylim([0,1])
		axarr[1,0].set_xlabel('Sinewave amplitude', fontsize='medium')
		axarr[1,0].set_ylabel('CDF', fontsize='medium')
		axarr[1,0].legend(loc='lower right', fontsize='medium')
		axarr[1,0].set_xlim([-0.022,0.022])

		axarr[1,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[1,1].set_ylim([-0.022,0.022])

		data = doppler[doppler.id == i]['Mic2']
		n, base = np.histogram(data,bins=50)
		axarr[1,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
	
	################## LOAD DATA FOR 3rd ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries[timeseries.id == i]['Mic3']
		n, base = np.histogram(data,bins=50)

		cumulative = np.cumsum(n)
		cumulative = cumulative/float(max(cumulative))
		axarr[2,0].plot(base[:-1], cumulative, color=colors[i%len(colors)],label="Sample " + str(i))

		axarr[2,0].legend(loc='best')
		axarr[2,0].set_ylim([0,1])
		axarr[2,0].set_xlabel('Sinewave amplitude', fontsize='medium')
		axarr[2,0].set_ylabel('CDF', fontsize='medium')
		axarr[2,0].legend(loc='lower right', fontsize='medium')
		axarr[2,0].set_xlim([-0.023,0.023])

		axarr[2,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[2,1].set_ylim([-0.023,0.023])

		data = doppler[doppler.id == i]['Mic3']
		n, base = np.histogram(data,bins=50)
		axarr[2,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
	
	################## LOAD DATA FOR 4th ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries[timeseries.id == i]['Mic4']
		n, base = np.histogram(data,bins=50)

		cumulative = np.cumsum(n)
		cumulative = cumulative/float(max(cumulative))
		axarr[3,0].plot(base[:-1], cumulative, color=colors[i%len(colors)],label="Sample " + str(i))

		axarr[3,0].legend(loc='best')
		axarr[3,0].set_ylim([0,1])
		axarr[3,0].set_xlabel('Sinewave amplitude', fontsize='medium')
		axarr[3,0].set_ylabel('CDF', fontsize='medium')
		axarr[3,0].legend(loc='lower right', fontsize='medium')
		axarr[3,0].set_xlim([-0.022,0.022])

		axarr[3,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[3,1].set_xlabel('Sinewave i point', fontsize='medium')
		axarr[3,1].set_ylim([-0.022,0.022])

		data = doppler[doppler.id == i]['Mic4']
		n, base = np.histogram(data,bins=50)
		axarr[3,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)

	#Write lateral Microphone number indicator
	plt.figtext(0.92, 0.78, 'Mic 1', fontsize=20)
	plt.figtext(0.92, 0.58, 'Mic 2', fontsize=20)
	plt.figtext(0.92, 0.38, 'Mic 3', fontsize=20)
	plt.figtext(0.92, 0.18, 'Mic 4', fontsize=20)

	#plt.show() #for debugging purposes
	print "Plotting " + f_name + " timeseries"
	fig.savefig('/Volumes/TessierAshpool/Plots/Timeseries/' + f_name + '.png')   # save the figure to file
	plt.close(fig)


def plotNoiseRemoval(f_name):
	timeseries = pandas.read_table(rmNoiseFolder + f_name, delim_whitespace=True, names=('id','Mic1', 'Mic2', 'Mic3', 'Mic4'))
	timeseries_cut = pandas.read_table(rmNoiseCutFolder + f_name, delim_whitespace=True, names=('id','Mic1', 'Mic2', 'Mic3', 'Mic4'))
	doppler = pandas.read_table(dopplerFolder + f_name, delim_whitespace=True, names=('id','Mic1', 'Mic2', 'Mic3', 'Mic4'))
	f_name = f_name[15:].replace('.txt', '')
	fig, axarr = plt.subplots(4, 3,figsize=(15,8))
	fig.suptitle("Sinewave amplitude CDF - " + f_name, fontsize=14)

	################## LOAD DATA FOR 1st ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries_cut[timeseries_cut.id == i]['Mic1']
		axarr[0,0].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[0,0].legend(loc='best')
		axarr[0,0].set_ylabel('Sinewave Amplitude', fontsize='medium')
		axarr[0,0].legend(loc='lower right', fontsize='medium')
		axarr[0,0].set_title('Raw Sinewave Amplitude',fontsize='large')
		#axarr[0,0].set_ylim([-1.763,1.763])

		data = timeseries[timeseries.id == i]['Mic1']
		axarr[0,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[0,1].set_title('Removed Noise Sinewave Amplitude',fontsize='large')
		axarr[0,1].set_ylim([-0.026,0.026])

		data = timeseries_cut[timeseries_cut.id == i]['Mic1'].groupby(lambda x: x/8).mean()
		#n, base = np.histogram(data,bins=50)
		axarr[0,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[0,2].set_title('Extended Doppler Shift Plot',fontsize='large')

	################## LOAD DATA FOR 2nd ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries_cut[timeseries_cut.id == i]['Mic2']
		axarr[1,0].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[1,0].legend(loc='best')
		axarr[1,0].set_xlabel('Sinewave amplitude', fontsize='medium')
		axarr[1,0].set_ylabel('Sinewave Amplitude', fontsize='medium')
		axarr[1,0].legend(loc='lower right', fontsize='medium')
		#axarr[1,0].set_ylim([-1.763,1.763])

		data = timeseries[timeseries.id == i]['Mic2']
		axarr[1,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[1,1].set_ylim([-0.022,0.022])

		data = timeseries_cut[timeseries_cut.id == i]['Mic2'].groupby(lambda x: x/16).mean()
		#n, base = np.histogram(data,bins=50)
		axarr[1,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
	
	################## LOAD DATA FOR 3rd ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries_cut[timeseries_cut.id == i]['Mic3']
		axarr[2,0].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)


		axarr[2,0].legend(loc='best')
		axarr[2,0].set_xlabel('Sinewave amplitude', fontsize='medium')
		axarr[2,0].set_ylabel('Sinewave Amplitude', fontsize='medium')
		axarr[2,0].legend(loc='lower right', fontsize='medium')
		#axarr[2,0].set_ylim([-1.763,1.763])

		data = timeseries[timeseries.id == i]['Mic3']
		axarr[2,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[2,1].set_ylim([-0.023,0.023])

		data = timeseries_cut[timeseries_cut.id == i]['Mic3'].groupby(lambda x: x/8).mean()
		#n, base = np.histogram(data,bins=50)
		axarr[2,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
	
	################## LOAD DATA FOR 4th ROW ############
	#####################################################

	for i in range(0,n_samples):#range(0,timeseries[0]['id'].max()):
		data = timeseries_cut[timeseries_cut.id == i]['Mic4']
		axarr[3,0].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)


		axarr[3,0].legend(loc='best')
		axarr[3,0].set_xlabel('Sinewave amplitude', fontsize='medium')
		axarr[3,0].set_ylabel('Sinewave Amplitude', fontsize='medium')
		axarr[3,0].legend(loc='lower right', fontsize='medium')
		#axarr[3,0].set_ylim([-1.763,1.763])

		data = timeseries[timeseries.id == i]['Mic4']
		axarr[3,1].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)
		axarr[3,1].set_xlabel('Sinewave i point', fontsize='medium')
		axarr[3,1].set_ylim([-0.022,0.022])

		data = timeseries_cut[timeseries_cut.id == i]['Mic4'].groupby(lambda x: x/8).mean()
		#n, base = np.histogram(data,bins=50)
		axarr[3,2].plot(np.linspace(0,len(data),len(data)), data,color=colors[i%len(colors)],alpha=alphas[i%len(alphas)],label="Sample " + str(i), linewidth=2)

	#Write lateral Microphone number indicator
	plt.figtext(0.92, 0.78, 'Mic 1', fontsize=20)
	plt.figtext(0.92, 0.58, 'Mic 2', fontsize=20)
	plt.figtext(0.92, 0.38, 'Mic 3', fontsize=20)
	plt.figtext(0.92, 0.18, 'Mic 4', fontsize=20)

	#plt.show() #for debugging purposes
	print "Plotting " + f_name + " timeseries"
	fig.savefig('/Volumes/TessierAshpool/Plots/Timeseries/' + f_name + '.png')   # save the figure to file
	plt.close(fig)

if __name__ == "__main__":

	for f in os.listdir(rawFolder):
		if("freshData" in f):
			plotNoiseRemoval(f)
			#plotTimeseries(f)
			#plotMusic(f)
