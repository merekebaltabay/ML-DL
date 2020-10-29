from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
from scipy import stats
import datetime
import dill
import matplotlib.pyplot as plt
import imblearn

file_path = 'mHealth_subjects.csv'

names=['accX', 'accY', 'accZ', 'ecg1', 'ecg2', 'alaX', 'alaY', 'alaZ', 'glaX', 'glaY', 'glaZ', 'mlaX', 'mlaY', 'mlaZ',  'arlaX', 'arlaY', 'arlaZ', 'grlaX', 'grlaY', 'grlaZ', 'mrlaX', 'mrlaY', 'mrlaZ', 'label']
#names = ['ecg1', 'ecg2', 'label']
df = pd.read_csv(file_path,
#                 usecols = [3, 4, 23],
                 names = names,
#                 usecols=[0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 23],
#                 names=names,
                delimiter=';')


#df = df[df.label != 0]
#df = df.reset_index(drop=True)
print(df.shape)

print(df.label.value_counts())


df[df['label']==0] = df[df['label']==0].sample(15360).astype(int)
df.dropna(axis=0, how='any', inplace=True)
print(df.shape)
print(df.label.value_counts())
#df[df['label']==12.0] = df[df['label']==12.0].sample(15360, replace = True)


for n in names:
    if(n!= 'label'):
        df[n] = df[n]/df[n].max()


y = df['label']


df = df.drop('label', axis=1)
df = np.array(df)

pd.options.mode.chained_assignment = None
x_train, x_test, y_train, y_test = train_test_split(df, y,  test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

rf = RandomForestClassifier(n_estimators = 1000, max_depth=23, random_state=42, min_samples_split=23)
rf.fit(x_train, y_train)
with open('rf.h5', 'wb') as f:
    dill.dump(rf, f)


start = datetime.datetime.now()
y_pred = rf.predict(x_test)
f = open("ECG_RF_results.txt", "a+")

print("Accuracy:", accuracy_score(y_test, y_pred))
f.write("Accuracy:"+str(accuracy_score(y_test, y_pred)))

elapsed = datetime.datetime.now() - start

print(confusion_matrix(y_test,y_pred))
f.write("confusion_matrix:"+str(confusion_matrix(y_test, y_pred)))

print(classification_report(y_test,y_pred))
f.write("classification_report:"+str(classification_report(y_test, y_pred)))

print("Elapsed time: ",elapsed.seconds,":",elapsed.microseconds)
f.write("Elapsed time: "+str(elapsed.seconds)+":"+str(elapsed.microseconds))
f.close()


labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12']
plot_confusion_matrix(rf, x_test, y_test,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize="true")
plt.show()
plt.savefig('rf'+str(start)+'.png')
