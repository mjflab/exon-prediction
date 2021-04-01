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

print (' unbalance h exon 2 coeff_treat cross chronmatin')

dataraw0 = pd.read_csv('HeLaS3_exonFeatureMatrix_sampleOneExonPerID_quantileCoeff.txt',delimiter = '\t')
explabel = dataraw0.coeff_treat
chrom =dataraw0.chromosome
num1=0
y = np.zeros((len(explabel)))
for i in range (len(y)):
    if not np.isnan(explabel[i]):
        y[i]=1
        num1+=1
num0 = int(len(y))-num1
print (len(y),num0,num1)

dataraw0 = np.array(dataraw0) # 
X = dataraw0[:,5:45] #gm 5:46 other 5:45


testidx=[]
trainchrom = ['chr1','chr2','chr4','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr14','chr15','chr17','chr18','chr19','chr21','chrX']
testchrom = ['chr3','chr5','chr13','chr16','chr20','chr22']

# no balance
trainidx = []
for i in range(len(chrom)):
    if chrom[i]  in testchrom:
        testidx.append(i)
    elif chrom[i] in trainchrom:
        trainidx.append(i)
'''

# balance
y0=[]
y1=[]
for i in range(len(chrom)):
    if chrom[i]  in testchrom:
        testidx.append(i)
    elif chrom[i] in trainchrom:
        if y[i]==0:
            y0.append(i)
        else:
            y1.append(i)
y00=resample(y0,replace=False,n_samples=len(y1))
trainidx = np.concatenate((y00,y1),axis=0)
print ('train test num:',len(trainidx),len(testidx))
'''
Xtrain = np.array(X[trainidx])
Xtest = np.array(X[testidx])
ytrain = np.array(y[trainidx])
ytest = np.array(y[testidx])

clf2 = GradientBoostingClassifier(min_samples_leaf=7, max_depth=7, min_samples_split=50, subsample=0.9,n_estimators=500, learning_rate=0.2, random_state=0 )

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
cm = confusion_matrix(ytest, y_predict2)
print (cm)

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
