import os
import numpy as np
import pandas as pd
import logging
import scipy.stats
from astropy.stats import median_absolute_deviation
from numpy.fft import rfft
import librosa

position = 'Hand'
train = 'test'
feature_set = 5

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

intermediate_dataset_path = path + 'dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'
primary_dataset_path = path + 'dataset\\' + train + ' csv\\' + position + '\\'
output_path = path + 'dataset\\feature set\\set ' + repr(feature_set) + ' (attempt 4, final feature set)\\'

try:  
    os.mkdir(output_path)
except OSError:  
    print(output_path + ' folder already existed')
else:
	print(output_path + ' folder created')

#setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# create error file handler and set level to error
handler = logging.FileHandler(output_path+'feature set.log',"w", encoding=None, delay="true")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def feat_extract(sensor_name, path = intermediate_dataset_path, train = train):
	train_feat = []

	train_data = pd.read_csv(path+train+'_'+sensor_name+'.csv', header =None)
	logging.debug(train_data.shape)
	logging.debug(sensor_name+ ' read done')
	
	train_feat.append(np.mean(train_data, axis=1))
	logging.info(sensor_name+ ' mean')
	train_feat.append(np.std(train_data, axis=1))
	logging.info(sensor_name+ ' std')
	train_feat.append(np.var(train_data, axis = 1)) 
	logging.info(sensor_name+ ' var')
	train_feat.append(np.max(train_data, axis = 1))
	logging.info(sensor_name+ ' max')
	train_feat.append(np.min(train_data, axis = 1))
	logging.info(sensor_name+ ' min')
	train_feat.append(median_absolute_deviation(train_data, axis =1))
	logging.info(sensor_name+ ' mad')
	train_feat = np.transpose(train_feat)
	train_feat = np.c_[train_feat, fft_feat_extract(np.array(train_data), sensor_name)]
	return train_feat

def fft_feat_extract(data, sensor_name):
	try:
		fft = np.array(pd.read_csv(intermediate_dataset_path+train+'_'+sensor_name+'_fft.csv', header = None))
	except:
		fft = []
		for i in range (len(data)):
			fft.append(np.abs(rfft(data[i])))
		fft = np.array(fft)
		np.savetxt(intermediate_dataset_path+train+'_'+sensor_name+'_fft.csv', fft, delimiter=",", fmt ='%.3f')
	logging.debug('fft done')
	
	fft_feat = np.sum( (fft * np.arange(len(fft[0])) ), axis = 1)
	fft_feat = np.divide(fft_feat, np.sum(fft, axis = 1))
	logging.info(sensor_name+ ' mean freq')
	fft_feat = np.c_[fft_feat, scipy.stats.skew(fft, axis = 1)]
	logging.info(sensor_name+ ' skewness')
	fft_feat = np.c_[fft_feat, scipy.stats.kurtosis(fft, axis = 1)]
	logging.info(sensor_name+ ' kurtosis')
	fft_feat = np.c_[fft_feat, np.sum(np.multiply(fft, fft), axis = 1)]
	logging.info(sensor_name+ ' energy')
	log = np.nan_to_num(np.log(fft))
	fft_feat = np.c_[fft_feat, -1 * np.sum(np.multiply(log, fft), axis = 1)]
	logging.info(sensor_name+ ' entropy')
	return np.array(fft_feat)

def lacc_feat_extract(av='', ah='', intermediate_file = intermediate_dataset_path, basic_file = primary_dataset_path):
	feature = []
	#feature from vertical acceleration
	av_data = np.array(pd.read_csv(intermediate_file+train+'_av'+av+'.csv', header = None))
	logging.debug(av_data.shape)
	lacc = np.array(pd.read_csv(basic_file+train+'_LAcc_mag.csv', header = None))
	logging.debug(lacc.shape)
	ah_data = np.array(pd.read_csv(intermediate_file+train+'_ah'+ah+'.csv', header = None))
	logging.debug(ah_data.shape)

	#range of av
	feature.append(np.ptp(av_data, axis = 1))
	logging.info('av'+av+' range')

	#max value of diff of av
	max_change_rate = np.diff(av_data, axis = 1)
	max_change_rate = np.max(max_change_rate, axis = 1)
	feature.append(max_change_rate)
	logging.info('av'+av+' max value of diff')

	#Energy of lacc
	fft = np.fft.rfft(lacc, axis = 1)
	fft = np.multiply(fft, fft)
	fft = np.sum(fft, axis = 1)
	feature.append(abs(fft))
	logging.info('lacc_mag Energy')

	'''
	#Zero-crossing rate av
	Zero_crossing_rate = [librosa.feature.zero_crossing_rate(av_data[n]) for n in range(len(av_data))]
	Zero_crossing_rate = np.array(Zero_crossing_rate)[:,0,0]
	feature.append(Zero_crossing_rate)
	logging.info('av'+av+' Zero-crossing rate')'''

	#Mean lacc
	feature.append(np.mean(lacc, axis=1))
	logging.info('lacc_mag Mean')

	#var lacc
	feature.append(np.var(lacc, axis=1))
	logging.info('lacc_mag var')

	#Mean ah
	feature.append(np.mean(ah_data, axis=1))
	logging.info('ah'+ah+' Mean')

	#range of lacc
	feature.append(np.ptp(lacc, axis = 1))
	logging.info('lacc_mag range')

	feature.append(np.trapz(lacc, dx = 0.01, axis = 1))
	logging.info('lacc_mag velocity')

	feature = np.transpose(feature)
	return feature


train_set = feat_extract('Acc_mag', path = primary_dataset_path)
logging.debug(train_set.shape)
train_set = np.c_[train_set, feat_extract('Mag_mag', path = primary_dataset_path)]
logging.debug(train_set.shape)

sensor = np.array(['acc', 'mag'])
axis = np.array(['_x', '_y', '_z'])

for sen in sensor:
	for ax in axis:
		train_set = np.c_[train_set, feat_extract(sen+ax+'(normalized)')]
		logging.debug(train_set.shape)

train_set = np.c_[train_set, lacc_feat_extract()]
logging.debug(train_set.shape)

pressure = np.array(pd.read_csv(primary_dataset_path+train+'_Pressure.csv', header = None))
train_set = np.c_[train_set, np.mean(pressure, axis = 1)]
logging.info('pressure mean')
logging.debug(train_set.shape)

label_file = pd.read_csv(output_path + 'feature set.log', header = None)
df = pd.DataFrame(train_set, columns = label_file)
df.to_csv(output_path+'features_'+train+'_'+position+'.csv', index=False)
lab = pd.read_csv(primary_dataset_path+train+'_label.csv', header = None)
np.savetxt(output_path+'label_'+train+'_'+position+'.csv', lab, delimiter=",", fmt ='%.1f')