from __future__ import division
import numpy as np
import pandas as pd
import math
import random


from sklearn.metrics import accuracy_score,roc_curve,precision_recall_curve,auc,confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

print ('tran gm test k 2-class')

dataraw0 = pd.read_csv('GM12878_quantileExp_featureMatrix_log.txt',delimiter = '\t')#HeLaS3_quantileExp_featureMatrix_log K562exon_log2fold_overlap0
ytrain = dataraw0.label#log2FoldChange#log2fold_treat_control#label
print (len(ytrain))

dataraw0 = np.array(dataraw0) # pos: 0:222582 neg 222582:
#Xtrain = dataraw0[:,12:52]# gm  trans 11:52 other 11:51
Xtrain = np.concatenate((dataraw0[:,11:12],dataraw0[:,13:52]), axis=1)

dataraw1 = pd.read_csv('K562_quantileExp_featureMatrix_log.txt',delimiter = '\t')#HeLaS3_quantileExp_featureMatrix_log K562exon_log2fold_overlap0
ytest = dataraw1.label#log2FoldChange#log2fold_treat_control#label
print (len(ytest))

dataraw1 = np.array(dataraw1) # pos: 0:222582 neg 222582:
Xtest = dataraw1[:,11:51]
#Xtest = np.concatenate((dataraw1[:,11:12],dataraw1[:,13:52]), axis=1)


clf2 = GradientBoostingClassifier(min_samples_leaf=7, max_depth=5, min_samples_split=50, n_estimators=100, learning_rate=0.15, subsample=0.9,random_state=0)

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
print ('GB (acc,auc,prc,sn,sp,pre,recall):',acc2,aoauc2,auprc2,sn,sp,pre,recall)
