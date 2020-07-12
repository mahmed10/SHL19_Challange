import os
import numpy as np
import pandas as pd
import logging
import joblib

train = 'train'
position = 'Torso'
sensor_name = 'Ori'
axis = np.array(['_x', '_y', '_z', '_w'])

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path)

input_file = path + '\\dataset\\' + train + ' csv\\' + position + '\\'
output_file = path + '\\dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'

#defing logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


try:  
    os.mkdir(output_file)
except OSError:  
    logging.debug(output_file + ' folder already existed')
else:
	logging.debug(output_file + ' folder created')

#reading orientation file
ori = []
for i in range(4):
	ori.append(np.array(pd.read_csv(input_file+train+'_'+sensor_name+axis[i]+'.csv', header = None)))
logging.info('read done')
ori = np.array(ori)
logging.debug(ori.shape)
logging.debug(ori[0,:,:])

oripower2 = np.square(ori)
logging.debug(oripower2.shape)
logging.debug(len(oripower2[0,0,:]))

#calculatin rnb
rnb = []
logging.info('Total row: ' + repr(len(oripower2[0])))
for j in range(len(oripower2[0])):
	rnb_row = []
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

		rnb_row.append(np.array(rnb_basic))
	rnb.append(np.array(rnb_row))
	if(j%500 == 0):
		logging.info(repr(j) + ' row done')
#rnb = np.array(rnb)
#logging.info(rnb.shape)
joblib.dump(np.array(rnb), output_file + train + '_rnb.pkl')
