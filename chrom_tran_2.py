from __future__ import division
import numpy as np
import pandas as pd
import math
import random


from sklearn.metrics import accuracy_score,roc_curve,precision_recall_curve,auc,confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

print ('tran k chrom 2-class')

dataraw0 = pd.read_csv('K562_quantileExp_featureMatrix_log.txt',delimiter = '\t')#HeLaS3_quantileExp_featureMatrix_log K562exon_log2fold_overlap0
chrom =dataraw0.seqname


y = dataraw0.label#log2FoldChange#log2fold_treat_control#label
print (len(y))

dataraw0 = np.array(dataraw0) # pos: 0:222582 neg 222582:
X = dataraw0[:,11:51]# gm  trans 11:52 other 11:51

trainidx=[]
testidx=[]
trainchrom = ['chr1','chr2','chr4','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr14','chr15','chr17','chr18','chr19','chr21','chrX']
testchrom = ['chr3','chr5','chr13','chr16','chr20','chr22']
for i in range(len(chrom)):
    if chrom[i]  in testchrom:
        testidx.append(i)
    elif chrom[i] in trainchrom:
        trainidx.append(i)

print ('train test num:',len(trainidx),len(testidx))

Xtrain = np.array(X[trainidx])
Xtest = np.array(X[testidx])
ytrain = np.array(y[trainidx])
ytest = np.array(y[testidx])

clf2 = GradientBoostingClassifier(min_samples_leaf=3, max_depth=5, min_samples_split=30, n_estimators=100, learning_rate=0.15, subsample=1,random_state=0)

clf2.fit(Xtrain, ytrain)
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
print ('GB (acc,auc,prc,sn,sp,pre,recall):',acc2,aoauc2,auprc2,sn,sp,pre,recall)
