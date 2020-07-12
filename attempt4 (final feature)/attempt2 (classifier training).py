import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import joblib
from time import time
import winsound

#some info you need to edit
training_status = 'all positions'
position = 'Torso'
train = 'train'
feature_set = 8
classifier_name = 'svm linear' #first is class name and second is argument
#classifier rnf 20, svm linear, svm rbf, lr, lda, dt

#defining path and creating path
path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

input_path = path + '\\dataset\\feature set\\'
input_path = input_path + os.listdir(input_path)[feature_set-1]+'\\'
output_path = './models/set '+repr(feature_set)+'/'

try:  
    os.mkdir(output_path)
except OSError:  
    print(output_path + ' folder already existed')
else:
	print(output_path + ' folder created')

#defing logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# create error file handler and set level to error
handler = logging.FileHandler('.\\log\\'+classifier_name+' training phase.log',"w", encoding=None, delay="true")
handler.setLevel(logging.CRITICAL)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


#training file read
t1 = time()
X_train = np.array(pd.read_csv(input_path+'features_'+train+'_'+position+'.csv'))
y_train = np.array(pd.read_csv(input_path+'label_'+train+'_'+position+'.csv', header = None)[0])
for position in (np.array(['Hips', 'Bag'])):
	X_train = np.r_[X_train, np.array(pd.read_csv(input_path+'features_'+train+'_'+position+'.csv'))]
	y_train = np.r_[y_train, np.array(pd.read_csv(input_path+'label_'+train+'_'+position+'.csv', header = None)[0])]
train = 'validate'
for position in (np.array(['Torso', 'Hips', 'Bag', 'Hand'])):
	X_val = np.array(pd.read_csv(input_path+'features_'+train+'_'+position+'.csv'))
	y_val = np.array(pd.read_csv(input_path+'label_'+train+'_'+position+'.csv', header = None)[0])
	X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.35, random_state=42)
	X_train = np.r_[X_train, X_val]
	y_train = np.r_[y_train, y_val]
logging.debug(X_train.shape)
logging.debug(y_train.shape)
logging.info('train read done')
logging.critical('time for dataset reading: ' + str(time()-t1).zfill(2) + 's')

#normalization
t1 = time()
sc = StandardScaler()
X_train = np.nan_to_num(X_train)
X_train = sc.fit_transform(X_train)
joblib.dump(sc, output_path+'norm feature_set ' + repr(feature_set) + ' ' + training_status + '.pkl')
logging.info('normalization done')
logging.critical('time for normalization: ' + str(time()-t1).zfill(2) + 's')

#classifier training
t1 = time()
classifier_name = classifier_name.lower()
logging.debug(classifier_name)
if (classifier_name.split()[0] == 'rnf'):
	classifier = RandomForestClassifier(n_estimators = int(classifier_name.split()[1]),
		criterion = 'entropy', random_state = 0, verbose=2)
elif (classifier_name.split()[0] == 'svm'):
	classifier = SVC(kernel = classifier_name.split()[1], random_state = 0, probability = True, verbose=True)
elif (classifier_name.split()[0] == 'lr'):
	classifier = LogisticRegression(random_state = 0, verbose=True)
elif (classifier_name.split()[0] == 'lda'):
	classifier = LinearDiscriminantAnalysis()
elif (classifier_name.split()[0] == 'dt'):
	classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
elif (classifier_name.split()[0] == 'gradboost'):
	classifier = GradientBoostingClassifier(n_estimators = int(classifier_name.split()[1]), verbose=True)
logging.info(classifier)
classifier.fit(X_train, y_train)
logging.info('training done')
logging.critical('time for training: ' + str(time()-t1).zfill(2) + 's')

joblib.dump(classifier, output_path+classifier_name+' feature_set ' + repr(feature_set) + ' ' + training_status + '.pkl')
winsound.Beep(1000, 2000)