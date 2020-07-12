import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import logging
import joblib

#some info you need to edit
training_status = 'all positions'
position = 'Hand'
test = 'validate'
feature_set = 8
classifier_name = 'dt' #first is class name and second is argument
#classifier rnf 20, svm linear, svm rbf, lr, lda, dt 

#defining path and creating path
path = os.path.abspath('..')
path = os.path.dirname(path) + '\\' + os.path.basename(path) + '\\'

input_path = path + '\\dataset\\feature set\\'
input_path = input_path + os.listdir(input_path)[feature_set-1]+'\\'
output_path = './result/set '+repr(feature_set)+'/'

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

#testing file read
X_test = np.array(pd.read_csv(input_path+'features_'+test+'_'+position+'.csv'))
y_test = np.array(pd.read_csv(input_path+'label_'+test+'_'+position+'.csv', header = None)[0])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.35, random_state=42)
logging.debug(X_test.shape)
logging.debug(y_test.shape)
logging.info('test read done')

#normalization
sc = joblib.load('./models/set '+repr(feature_set)+'/norm feature_set ' + repr(feature_set) + ' ' + training_status + '.pkl')
X_test = np.nan_to_num(X_test)
X_test = sc.transform(X_test)
logging.info('normalization done')

#making prediction
classifier = joblib.load('./models/set '+repr(feature_set)+'/'+classifier_name+' feature_set ' + repr(feature_set) + ' ' + training_status + '.pkl')
logging.info('classifier load done')
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
logging.info('prediction done')

#confusion matrix
labels = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
cm = confusion_matrix(y_test, y_pred)
print(cm)

#accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

accuracy = getAccuracy(y_test, y_pred)
print('Accuracy: ' + repr(accuracy) + '%')

#saving result
f1 = f1_score(y_test, y_pred, average=None)
df = pd.DataFrame(f1, columns = ['f1_score'])
df.to_csv(output_path+'f1_score '+classifier_name+' feature_set ' + repr(feature_set) + '.csv', index=False)
result = np.c_[y_test, y_pred]
df = pd.DataFrame(result, columns = ['true', 'pred'])
df.to_csv(output_path+'result '+classifier_name+' feature_set ' + repr(feature_set) + '.csv', index=False)
df = pd.DataFrame(cm, columns = labels, index = labels)
df.to_csv(output_path+'confusion_matrix '+classifier_name+' feature_set ' + repr(feature_set) + '.csv')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df = pd.DataFrame(cm, columns = labels, index = labels).round(2)
df.to_csv(output_path+'normalized confusion_matrix '+classifier_name+' feature_set ' + repr(feature_set) + '.csv')
df = pd.DataFrame(y_prob, columns = labels)
df.to_csv(output_path+'probability '+classifier_name+' feature_set ' + repr(feature_set) + '.csv', index=False)
np.savetxt(output_path+'accuracy '+classifier_name+' feature_set ' + repr(feature_set) + '.txt', np.array([accuracy]), fmt = '%.4f')