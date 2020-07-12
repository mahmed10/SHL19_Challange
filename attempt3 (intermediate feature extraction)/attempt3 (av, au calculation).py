import os
import numpy as np
import pandas as pd
import logging

train = 'test'
position = 'Hand'
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

try:  
    os.mkdir(output_path)
except OSError:  
    logging.debug(output_path + ' folder already existed')
else:
	logging.debug(output_path + ' folder created')

#reading lacc file
lacc = []
for i in range(3):
	lacc.append(np.array(pd.read_csv(input_file+train+'_LAcc'+axis[i]+'.csv', header = None)))
logging.info('lacc read done')
lacc = np.array(lacc)
logging.debug(lacc.shape)
logging.debug(lacc[0,:,:])

laccpower2 = np.square(lacc)
logging.debug(laccpower2.shape)
logging.debug(len(laccpower2[0,0,:]))
lacc_mag = np.sqrt(laccpower2[0,:,:] + laccpower2[1,:,:] + laccpower2[2,:,:])


#reading gravity file
grav = []
for i in range(3):
	grav.append(np.array(pd.read_csv(input_file+train+'_Gra'+axis[i]+'.csv', header = None)))
logging.info('grav read done')
grav = np.array(grav)
logging.debug(grav.shape)
logging.debug(grav[0,:,:])

gravpower2 = np.square(grav)
logging.debug(gravpower2.shape)
logging.debug(len(gravpower2[0,0,:]))
grav_mag = np.sqrt(gravpower2[0,:,:] + gravpower2[1,:,:] + gravpower2[2,:,:])

cos_theta = np.multiply(grav[0,:,:],lacc[0,:,:]) + np.multiply(grav[1,:,:],lacc[1,:,:]) + np.multiply(grav[2,:,:],lacc[2,:,:])
cos_theta = np.divide(cos_theta, np.multiply(lacc_mag,grav_mag))
cos_theta = np.nan_to_num(cos_theta)

sin_theta = np.sin(np.arccos(cos_theta))

av = np.multiply(lacc_mag, cos_theta)
au = np.multiply(lacc_mag, sin_theta)
np.savetxt(output_path+train+'_av.csv', np.array(av) , delimiter=",", fmt ='%.3f')
np.savetxt(output_path+train+'_ah.csv', np.array(au) , delimiter=",", fmt ='%.3f')