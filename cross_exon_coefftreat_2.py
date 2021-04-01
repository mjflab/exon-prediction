from __future__ import division
import numpy as np
import pandas as pd
import math
import random

from sklearn.utils import resample
from sklearn.metrics import accuracy_score,roc_curve,precision_recall_curve,auc,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

print (' unbalance k test hela 2-label coeff_treat')
num1=0
dataraw0 = pd.read_csv('K562_exonFeatureMatrix_sampleOneExonPerID_quantileCoeff.txt',delimiter = '\t')
explabel = dataraw0.coeff_treat


# no balance
ytrain=np.zeros((len(explabel)))
for i in range (len(explabel)):
    if not np.isnan(explabel[i]):
        ytrain[i]=1
        num1+=1
num0 = int(len(ytrain))-num1
print (len(ytrain),num0,num1)
dataraw0 = np.array(dataraw0) # 
Xtrain = dataraw0[:,5:45] #gm 5:46 other 5:45
#Xtrain = np.concatenate((dataraw0[:,5:6],dataraw0[:,7:46]), axis=1)
'''
# balance
y0=[]
y1=[]
ytrain0 = np.zeros((len(explabel)))
for i in range (len(explabel)):
    if not np.isnan(explabel[i]):
        y1.append(i)
        ytrain0[i]=1
    else:
        y0.append(i)
y00=resample(y0,replace=False,n_samples=len(y1))
trainidx = np.concatenate((y00,y1),axis=0)


dataraw0 = np.array(dataraw0) #
X = dataraw0[:,5:45] #gm 5:46 other 5:45
#X = np.concatenate((dataraw0[:,5:6],dataraw0[:,7:46]), axis=1)

Xtrain = np.array(X[trainidx])
ytrain = np.array(ytrain0[trainidx])
'''
dataraw1 = pd.read_csv('HeLaS3_exonFeatureMatrix_sampleOneExonPerID_quantileCoeff.txt',delimiter = '\t')
explabeltest = dataraw1.coeff_treat
ytest = np.zeros((len(explabeltest)))

for i in range (len(ytest)):
    if not np.isnan(explabeltest[i]):
        ytest[i]=1

dataraw1 = np.array(dataraw1) #
Xtest = dataraw1[:,5:45]
#Xtest = np.concatenate((dataraw1[:,5:6],dataraw1[:,7:46]), axis=1)
print ('train test num:',len(Xtrain),len(Xtest))

clf2 = GradientBoostingClassifier(min_samples_leaf=7, max_depth=7, min_samples_split=50, n_estimators=500, learning_rate=0.2, subsample=0.9,random_state=0)

clf2.fit(Xtrain, ytrain)
'''
y_predict2 = clf2.predict(Xtest)
acc2 = accuracy_score(ytest, y_predict2)
probs2 = clf2.predict_proba(Xtest)
fpr, tpr, threshs = roc_curve(ytest,probs2[:,1])
aoauc2 = auc(fpr,tpr)
precision, recall, thresholds = precision_recall_curve(ytest,probs2[:,1])
auprc2 = auc(recall,precision)
tn, fp, fn, tp = confusion_matrix(ytest, y_predict2).ravel()
sn = tp/(tp+fn)
sp = tn/(fp+tn)
pre = tp/(tp+fp)
recall = tp/(tp+fn)
print ('tn, fp, fn, tp',tn, fp, fn, tp)
print ('GB (acc,auc,prc,sn,sp,pre,recall):',round(acc2,4),round(aoauc2,4),round(auprc2,4),round(sn,4),round(sp,4),round(pre,4),round(recall,4))
'''
# random
y_predict20 = np.random.choice(ytest, len(ytest),replace=False)
acc2 = accuracy_score(ytest, y_predict20)
probs2 = clf2.predict_proba(Xtest)
probs20 = np.random.choice(probs2[:,1], len(probs2),replace=False)

fpr, tpr, threshs = roc_curve(ytest,probs20)
aoauc2 = auc(fpr,tpr)
precision, recall, thresholds = precision_recall_curve(ytest,probs20)
auprc2 = auc(recall,precision)
print ('random (acc,auc,prc):',round(acc2,4),round(aoauc2,4),round(auprc2,4))

