import os
import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
import random

train = 'validate'
position = np.array(['Torso', 'Hips', 'Bag', 'Hand'])
activity = 2

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path)

#defing logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

plt.figure()
plt.title('activity = ' + repr(activity))

label_path = path + '\\dataset\\' + train + ' csv\\' + position[0] + '\\'
label = np.array(pd.read_csv(label_path+train+'_label'+'.csv', header = None))
random_label_index = random.choice(np.array(np.where(label==activity))[0])
logging.debug(random_label_index)
for i in range(len(position)):
	data_file_path = path + '\\dataset\\' + 'intermediate ' + train + ' feature csv\\' + position[i] + '\\'
	au = np.array(pd.read_csv(data_file_path+train+'_ah(normalized with alpha)'+'.csv', header = None))
	av = np.array(pd.read_csv(data_file_path+train+'_av'+'.csv', header = None))
	data_file_path = path + '\\dataset\\' + train + ' csv\\' + position[i] + '\\'
	gyro = np.array(pd.read_csv(data_file_path+train+'_gyr_x'+'.csv', header = None))
	plot_number = int(2*100+20+int(i)+1)
	logging.debug(plot_number)
	plt.subplot(plot_number, title=position[i])
	plt.plot(au[random_label_index,:30], 'b-', label = 'ah')
	plt.plot(av[random_label_index,:30], 'g-', label = 'av')
	plt.plot(gyro[random_label_index,:30], 'y-', label = 'gyro_x')
	plt.legend(loc='upper right')
	#plt.figure(figsize=(8.0, 5.0))
plt.savefig('graph.eps')
plt.show()