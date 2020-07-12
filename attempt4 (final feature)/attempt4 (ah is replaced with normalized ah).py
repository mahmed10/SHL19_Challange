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
previous_feature_set = 5
feature_set = 7

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

intermediate_dataset_path = path + 'dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'
primary_dataset_path = path + 'dataset\\' + train + ' csv\\' + position + '\\'
output_path = path + 'dataset\\feature set\\set ' + repr(feature_set) + ' (attempt 4, final feature set where ah normalized with alpha)\\'
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

previous_feature = pd.read_csv(previous_feature_set_path+'features_'+train+'_'+position+'.csv')
ah_data = np.array(pd.read_csv(intermediate_dataset_path+train+'_ah(normalized with alpha).csv', header = None))
logging.debug(ah_data.shape)

ah_data = np.mean(ah_data, axis = 1)
logging.debug(ah_data.shape)
logging.debug(previous_feature.iloc[:,93])
previous_feature.iloc[:,93] = ah_data
previous_feature.rename(columns={'(\'ah Mean\',)': '(\'ah(normalized with alpha) Mean\',)'}, inplace=True)
logging.debug(previous_feature.iloc[:,93])
previous_feature.to_csv(output_path+'features_'+train+'_'+position+'.csv', index=False)
logging.info('same as set 5 but av is replaced with normalized ah')

lab = pd.read_csv(primary_dataset_path+train+'_label.csv', header = None)
np.savetxt(output_path+'label_'+train+'_'+position+'.csv', lab, delimiter=",", fmt ='%.1f')