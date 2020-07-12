import os
import numpy as np
import pandas as pd
import logging
import joblib

train = 'validate'
position = 'Hips'
sensor_name = '_mag'
axis = np.array(['_x', '_y', '_z','_w'])

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

intermediate_file = path + 'dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'
basic_file = path + 'dataset\\' + train + ' csv\\' + position + '\\'

#defing logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

try:  
    os.mkdir(intermediate_file)
except OSError:  
    logging.debug(intermediate_file + ' folder already existed')
else:
	logging.debug(intermediate_file + ' folder created')
#reading acc file
acc = []
for i in range(3):
	acc.append(np.array(pd.read_csv(basic_file+train+sensor_name+axis[i]+'.csv', header = None)))
acc = np.array(acc)
acc = np.transpose(acc,(1, 2, 0))
logging.info('acc read done')

#reading orientation file
ori = []
for i in range(4):
	ori.append(np.array(pd.read_csv(basic_file+train+'_Ori'+axis[i]+'.csv', header = None)))
logging.info('ori read done')
ori = np.array(ori)
logging.debug(ori.shape)
logging.debug(ori[0,:,:])

oripower2 = np.square(ori)
logging.debug(oripower2.shape)
logging.debug(len(oripower2[0,0,:]))


norm_acc = []
logging.info('Total row: ' + repr(len(oripower2[0])))
for j in range(len(oripower2[0])):
	result = []
	for i in range (500):
		#in this loop calculating the basic 3x3 rnb matrix for each time stamp
		rnb_basic = []
		rnb_basic_row = []
		rnb_basic_row.append(1 - 2*(oripower2[1,j,i]+oripower2[2,j,i]))
		rnb_basic_row.append(2*(ori[0,j,i]*ori[1,j,i] - ori[3,j,i]*ori[2,j,i]))
		rnb_basic_row.append(2*(ori[0,j,i]*ori[2,j,i] + ori[3,j,i]*ori[1,j,i]))
		rnb_basic.append(np.array(rnb_basic_row))

		rnb_basic_row = []
		rnb_basic_row.append(2*(ori[0,j,i]*ori[1,j,i] + ori[3,j,i]*ori[2,j,i]))
		rnb_basic_row.append(1 - 2*(oripower2[0,j,i]+oripower2[2,j,i]))
		rnb_basic_row.append(2*(ori[1,j,i]*ori[2,j,i] - ori[3,j,i]*ori[0,j,i]))
		rnb_basic.append(np.array(rnb_basic_row))

		rnb_basic_row = []
		rnb_basic_row.append(2*(ori[0,j,i]*ori[2,j,i] - ori[3,j,i]*ori[1,j,i]))
		rnb_basic_row.append(2*(ori[1,j,i]*ori[2,j,i] + ori[3,j,i]*ori[0,j,i]))
		rnb_basic_row.append(1 - 2*(oripower2[0,j,i]+oripower2[1,j,i]))
		rnb_basic.append(np.array(rnb_basic_row))

		rnb_basic = np.array(rnb_basic)
		result.append(np.matmul(rnb_basic,acc[j,i,:]))

	norm_acc.append(np.transpose(result))
	if(j%500 == 0):
		logging.info(repr(j) + ' row done')

norm_acc =  np.transpose(norm_acc, (1,0,2))
for i in range(3):
	np.savetxt(intermediate_file+train+sensor_name+axis[i]+'(normalized).csv', norm_acc[i] , delimiter=",", fmt ='%.3f')
	logging.info(axis[i]+' write done')