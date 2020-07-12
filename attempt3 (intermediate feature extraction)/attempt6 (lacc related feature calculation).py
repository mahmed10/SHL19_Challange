import os
import numpy as np
import pandas as pd
import logging
import librosa

train = 'train'
position = 'Torso'
feature_set = 2

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

output_path = path + 'dataset\\feature set\\set ' + repr(feature_set) + ' (attempt 2, lacc related)\\'
intermediate_file = path + 'dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'
basic_file = path + 'dataset\\' + train + ' csv\\' + position + '\\'

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

feature = []
feature_name = []
#feature from vertical acceleration
av = np.array(pd.read_csv(intermediate_file+train+'_av.csv', header = None))
logging.debug(av.shape)
lacc = np.array(pd.read_csv(basic_file+train+'_LAcc_mag.csv', header = None))
logging.debug(lacc.shape)
ah = np.array(pd.read_csv(intermediate_file+train+'_ah(normalized with alpha).csv', header = None))
logging.debug(ah.shape)

#range of av
feature.append(np.ptp(av, axis = 1))
feature_name.append('range of av')
logging.info('range of av done')

#max value of diff of av
max_change_rate = np.diff(av, axis = 1)
max_change_rate = np.max(max_change_rate, axis = 1)
feature.append(max_change_rate)
feature_name.append('max value of diff of av')
logging.info('max value of diff of av done')

#Energy of lacc
fft = np.fft.rfft(lacc, axis = 1)
fft = np.multiply(fft, fft)
fft = np.sum(fft, axis = 1)
feature.append(abs(fft))
feature_name.append('Energy of lacc')
logging.info('Energy of lacc done')

#Zero-crossing rate av
Zero_crossing_rate = [librosa.feature.zero_crossing_rate(av[n]) for n in range(len(av))]
Zero_crossing_rate = np.array(Zero_crossing_rate)[:,0,0]
feature.append(Zero_crossing_rate)
feature_name.append('Zero-crossing rate av')
logging.info('Zero-crossing rate av done')

#Mean lacc
feature.append(np.mean(lacc, axis=1))
feature_name.append('Mean lacc')
logging.info('Mean lacc done')

#var lacc
feature.append(np.var(lacc, axis=1))
feature_name.append('var lacc')
logging.info('Mean var done')

#Mean ah
feature.append(np.mean(ah, axis=1))
feature_name.append('Mean ah')
logging.info('Mean ah done')

#range of lacc
feature.append(np.ptp(lacc, axis = 1))
feature_name.append('range of lacc')
logging.info('range of lacc done')

feature = np.array(feature)
logging.debug(feature.shape)

np.savetxt(output_path+'features_'+train+'_'+position+'.csv', np.transpose(feature), delimiter=",", fmt ='%.3f')
logging.debug('features_'+train+' done')
lab = pd.read_csv(basic_file+train+'_label.csv', header = None)
np.savetxt(output_path+'label_'+train+'_'+position+'.csv', lab, delimiter=",", fmt ='%.1f')