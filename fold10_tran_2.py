from __future__ import division
import numpy as np
import pandas as pd
import math
import random


from sklearn.metrics import accuracy_score,roc_curve,precision_recall_curve,auc,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

print ('tran gm balance no overlap 10-fold 2-class')

dataraw0 = pd.read_csv('GM12878_quantileExp_featureMatrix_log.txt',delimiter = '\t')#HeLaS3_quantileExp_featureMatrix_log K562exon_log2fold_overlap0
y = dataraw0.label#log2FoldChange#log2fold_treat_control#label
print (len(y))

dataraw0 = np.array(dataraw0) # pos: 0:222582 neg 222582:
X = dataraw0[:,11:52]# gm  trans 11:52 other 11:51


fold_n = 10
kf = StratifiedKFold(n_splits=fold_n,shuffle=True) # Define the split - into 2 folds
kf.get_n_splits(X)



for train, test in kf.split(X,y):
    
    clf2 = GradientBoostingClassifier(min_samples_leaf=7, max_depth=5, min_samples_split=50, n_estimators=100, learning_rate=0.15, subsample=0.9,random_state=0 )

    clf2.fit(X[train], y[train])
    y_predict2 = clf2.predict(X[test])
    acc2 = accuracy_score(y[test], y_predict2)
    probs2 = clf2.predict_proba(X[test])
    fpr, tpr, threshs = roc_curve(y[test],probs2[:,1])
    aoauc2 = auc(fpr,tpr)
    precision, recall, thresholds = precision_recall_curve(y[test],probs2[:,1])
    auprc2 = auc(recall,precision)
    tn, fp, fn, tp = confusion_matrix(y[test], y_predict2).ravel()
    sn = tp/(tp+fn)
    sp = tn/(fp+tn)
    pre = tp/(tp+fp)
    recall = tp/(tp+fn)
    print ('GB (acc,auc,prc,sn,sp,pre,recall):',acc2,aoauc2,auprc2,sn,sp,pre,recall)
