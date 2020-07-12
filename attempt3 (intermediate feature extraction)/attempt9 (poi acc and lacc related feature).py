import os
import numpy as np
import pandas as pd
from astropy.stats import median_absolute_deviation
import logging

position = 'Torso'
train = 'validate'
feature_set = 3
sensor_name = np.array(['acc'])
axis = np.array(['_x', '_y', '_z'])

path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

output_path = path + '\\dataset\\feature set\\set ' + repr(feature_set) + ' (attempt 2, t domain, lacc and poi acc)\\'
input_path = path + 'dataset\\' + 'intermediate ' + train + ' feature csv\\' + position + '\\'
set2_path = path + '\\dataset\\feature set\\'
set2_path = set2_path + os.listdir(set2_path)[2-1]+'\\'
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


def feat_extract(sensor_name, path = input_path):
	train_feat = []

	train_data = pd.read_csv(path+sensor_name+'.csv', header =None)
	logging.debug(train_data.shape)
	logging.debug(sensor_name+ ' read done')
	
	train_feat.append(np.mean(train_data, axis=1))
	logging.info(sensor_name+ ' mean done')
	train_feat.append(np.std(train_data, axis=1))
	logging.info(sensor_name+ ' std done')
	train_feat.append(np.var(train_data, axis = 1)) 
	logging.info(sensor_name+ ' var done')
	train_feat.append(np.max(train_data, axis = 1))
	logging.info(sensor_name+ ' max done')
	train_feat.append(np.min(train_data, axis = 1))
	logging.info(sensor_name+ ' min done')
	train_feat.append(median_absolute_deviation(train_data, axis =1))
	logging.info(sensor_name+ ' mad done')
	train_feat = np.array(train_feat)
	return train_feat.transpose()

sensor = []
for n in range(len(sensor_name)):
	for i in range(3):
		sensor.append(train+'_'+sensor_name[n]+axis[i]+'(normalized)')
	try:
		sensor.append(train+'_'+sensor_name[n]+axis[3]+'(normalized)')
	except:
		continue
logging.debug(sensor)

for i in range(len(sensor)):
	try:
		train_features = np.c_[train_features, feat_extract(sensor[i])]
	except:
		train_features = feat_extract(sensor[i])
logging.debug(train_features.shape)

train_features = np.c_[train_features,
	np.array(pd.read_csv(set2_path+'features_'+train+'_'+position+'.csv', header = None))]
logging.info('+ feature_set 2')

np.savetxt(output_path+'features_'+train+'_'+position+'.csv', train_features, delimiter=",", fmt ='%.3f')
logging.debug('features_'+train+' done')
lab = pd.read_csv(basic_file+train+'_label.csv', header = None)
np.savetxt(output_path+'label_'+train+'_'+position+'.csv', lab, delimiter=",", fmt ='%.1f')