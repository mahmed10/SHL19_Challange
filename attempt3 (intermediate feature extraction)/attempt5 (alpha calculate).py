import numpy as np
import pandas as pd
import os
import logging

train = 'validate'
position = 'Hips'
axis = np.array(['_x', '_y', '_z'])

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path)

input_file = path + '\\dataset\\' + train + ' csv\\' + position + '\\'
output_path = path + '\\dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'

#defing logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

max_gyr_diff = []
for i in range(3):
	gyr = np.array(pd.read_csv(input_file+train+'_Gyr'+axis[i]+'.csv', header = None))
	logging.debug(gyr.shape)
	gyr = np.diff(gyr, axis = 1)
	logging.debug(gyr.shape)
	#logging.debug(gyr)
	max_gyr_diff.append(np.max(gyr, axis = 1))
	logging.info('axis '+ axis[i]+ ' done')
max_gyr_diff = np.transpose(np.array(max_gyr_diff))
alpha = np.mean(max_gyr_diff, axis = 1)
logging.debug(alpha.shape)
alpha = 1 + np.divide(1, alpha)
logging.debug(alpha.shape)
logging.debug(alpha)
logging.info('alpha done')

ah = np.array(pd.read_csv(output_path+train+'_ah.csv', header = None))
logging.debug(ah.shape)
ah = np.multiply(alpha, np.transpose(ah))
np.savetxt(output_path+train+'_ah(normalized with alpha).csv', np.transpose(np.array(ah)) , delimiter=",", fmt ='%.3f')
logging.info('ah done')

av = np.array(pd.read_csv(output_path+train+'_av.csv', header = None))
av = np.multiply(alpha,np.transpose(av))
np.savetxt(output_path+train+'_av(normalized with alpha).csv', np.transpose(np.array(av)) , delimiter=",", fmt ='%.3f')
logging.info('av done')