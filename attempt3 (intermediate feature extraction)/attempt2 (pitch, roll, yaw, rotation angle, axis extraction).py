import os
import numpy as np
import pandas as pd
import logging

train = 'test'
position = 'Hand'
sensor_name = 'Ori'
axis = np.array(['_x', '_y', '_z', '_w'])

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

#reading orientation file
ori = []
for i in range(4):
	ori.append(np.array(pd.read_csv(input_file+train+'_'+sensor_name+axis[i]+'.csv', header = None)))
	logging.info('axis ' + axis[i] + ' done')
logging.info('read done')
ori = np.array(ori)
logging.debug(ori.shape)
logging.debug(ori[0,:,:])

oripower2 = np.square(ori)
logging.debug(oripower2.shape)
logging.debug(len(oripower2[0,0,:]))

#variable
pitch = []
roll = []
yaw = []
rotation_angle = []
rotation_x = []
rotation_y = []
rotation_z = []

logging.info('Total row: ' + repr(len(oripower2[0])))
for j in range(len(oripower2[0])):
	pitch_row = []
	roll_row = []
	yaw_row = []
	rotation_angle_row = []
	rotation_x_row = []
	rotation_y_row = []
	rotation_z_row = []
	for i in range (500):
		pitch_row.append(np.arctan((2*(ori[3,0,i]*ori[0,0,i]+ori[1,0,i]*ori[2,0,i])) / (1-2*(oripower2[0,0,i]+oripower2[1,0,i]))))
		roll_row.append(np.arcsin( 2*( (ori[3,0,i])*ori[1,0,i] - oripower2[0,0,i]) ))
		yaw_row.append(np.arctan((2*(ori[0,0,i]*ori[2,0,i]+ori[0,0,i]*ori[1,0,i])) / (1-2*(oripower2[1,0,i]+oripower2[2,0,i]))))
		rotation_angle_row.append(2*np.arccos( ori[3,0,i] ))
		rotation_x_row.append(2*np.arcsin( ori[0,0,i] ))
		rotation_y_row.append(2*np.arcsin( ori[1,0,i] ))
		rotation_z_row.append(2*np.arcsin( ori[2,0,i] ))
	pitch.append(np.array(pitch_row))
	roll.append(np.array(roll_row))
	yaw.append(np.array(yaw_row))
	rotation_angle.append(np.array(rotation_angle_row))
	rotation_x.append(np.array(rotation_x_row))
	rotation_y.append(np.array(rotation_y_row))
	rotation_z.append(np.array(rotation_z_row))
	if(j%100 == 0):
		logging.info(repr(j) + ' row done')
np.savetxt(output_path+train+'_pitch.csv', np.array(pitch) , delimiter=",", fmt ='%.3f')
np.savetxt(output_path+train+'_roll.csv', np.array(roll) , delimiter=",", fmt ='%.3f')
np.savetxt(output_path+train+'_yaw.csv', np.array(yaw) , delimiter=",", fmt ='%.3f')
np.savetxt(output_path+train+'_rotation_angle.csv', np.array(rotation_angle) , delimiter=",", fmt ='%.3f')
np.savetxt(output_path+train+'_rotation_x.csv', np.array(rotation_x) , delimiter=",", fmt ='%.3f')
np.savetxt(output_path+train+'_rotation_y.csv', np.array(rotation_y) , delimiter=",", fmt ='%.3f')
np.savetxt(output_path+train+'_rotation_z.csv', np.array(rotation_z) , delimiter=",", fmt ='%.3f')