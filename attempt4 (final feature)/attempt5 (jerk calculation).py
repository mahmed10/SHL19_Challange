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
previous_feature_set = 7
feature_set = 8

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

intermediate_dataset_path = path + 'dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'
primary_dataset_path = path + 'dataset\\' + train + ' csv\\' + position + '\\'
output_path = path + 'dataset\\feature set\\set ' + repr(feature_set) + ' (attempt 4, final feature set with jerk)\\'
previous_feature_set_path = path + '\\dataset\\feature set\\'
previous_feature_set_path = previous_feature_set_path + os.listdir(previous_feature_set_path)[previous_feature_set-1]+'\\'

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

	train_data = np.array(pd.read_csv(path+train+'_'+sensor_name+'.csv', header =None))
	logging.debug(train_data.shape)
	logging.debug(sensor_name+ ' read done')
	train_data = [np.gradient(train_data[i]) for i in range(len(train_data))]
	train_data = np.array(train_data)
	np.savetxt(intermediate_dataset_path+train+'_'+sensor_name+'_jerk.csv', train_data, delimiter=",", fmt ='%.3f')
	logging.debug(train_data.shape)
	
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
		fft = np.array(pd.read_csv(intermediate_dataset_path+train+'_'+sensor_name+'_jerk_fft.csv', header = None))
	except:
		fft = []
		for i in range (len(data)):
			fft.append(np.abs(rfft(data[i])))
		fft = np.array(fft)
		np.savetxt(intermediate_dataset_path+train+'_'+sensor_name+'_jerk_fft.csv', fft, delimiter=",", fmt ='%.3f')
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

previous_feature = pd.read_csv(previous_feature_set_path+'features_'+train+'_'+position+'.csv')
logging.debug(previous_feature.shape)

feature_set = feat_extract('av')
logging.debug(feature_set.shape)
feature_set = np.c_[feature_set, feat_extract('ah(normalized with alpha)')]
logging.debug(feature_set.shape)
feature_set = np.c_[feature_set, feat_extract('Acc_mag', path = primary_dataset_path)]
logging.debug(feature_set.shape)
feature_set = np.c_[feature_set, feat_extract('Lacc_mag', path = primary_dataset_path)]
logging.debug(feature_set.shape)

label_file = pd.read_csv(output_path + 'feature set.log', header = None)
feature_set = pd.DataFrame(feature_set, columns = label_file)
df = previous_feature.join(feature_set)
df.to_csv(output_path+'features_'+train+'_'+position+'.csv', index=False)
lab = pd.read_csv(primary_dataset_path+train+'_label.csv', header = None)
np.savetxt(output_path+'label_'+train+'_'+position+'.csv', lab, delimiter=",", fmt ='%.1f')