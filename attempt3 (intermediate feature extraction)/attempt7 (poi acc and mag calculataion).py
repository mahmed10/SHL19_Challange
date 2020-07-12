import os
import numpy as np
import pandas as pd
import logging
import joblib

train = 'validate'
position = 'Torso'
sensor_name = 'Ori'
axis = np.array(['_x', '_y', '_z'])

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

intermediate_file = path + 'dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'
basic_file = path + 'dataset\\' + train + ' csv\\' + position + '\\'

#defing logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

#reading acc file
acc = []
for i in range(3):
	acc.append(np.array(pd.read_csv(basic_file+train+'_acc'+axis[i]+'.csv', header = None)))
acc = np.array(acc)
acc = np.transpose(acc,(1, 2, 0))
logging.info('acc read done')

#reading mag file
mag = []
for i in range(3):
	mag.append(np.array(pd.read_csv(basic_file+train+'_mag'+axis[i]+'.csv', header = None)))
mag = np.array(mag)
mag = np.transpose(mag,(1, 2, 0))
logging.info('mag read done')

#reading rnb
rnb = joblib.load(intermediate_file + train + '_rnb.pkl')
logging.info('rnb read done')

logging.debug(rnb.shape)
logging.debug(acc.shape)
logging.debug(mag.shape)

norm_acc = []
norm_mag = []
logging.info('Total row: ' + repr(len(acc)))
for j in range(len(acc)):
    result = []
    result_1 = []
    for i in range(500):
        result.append(np.matmul(rnb[j,i,:,:],acc[j,i,:]))
        result_1.append(np.matmul(rnb[j,i,:,:],mag[j,i,:]))
    norm_acc.append(np.transpose(result))
    norm_mag.append(np.transpose(result_1))
    if(j%500 == 0):
		logging.info(repr(j) + ' row done')
norm_acc =  np.transpose(norm_acc, (1,0,2))
norm_mag =  np.transpose(norm_mag, (1,0,2))
for i in range(3):
	np.savetxt(intermediate_file+train+'_acc'+axis[i]+'(normalized).csv', norm_acc[i] , delimiter=",", fmt ='%.3f')
	np.savetxt(intermediate_file+train+'_mag'+axis[i]+'(normalized).csv', norm_mag[i] , delimiter=",", fmt ='%.3f')
	logging.info(axis[i]+' write done')