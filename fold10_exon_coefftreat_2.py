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

print (' unbalance k 10-fold exon 2 coeff_treat')
num1=0
dataraw0 = pd.read_csv('K562_exonFeatureMatrix_sampleOneExonPerID_quantileCoeff.txt',delimiter = '\t')
explabel = dataraw0.coeff_treat
y = np.zeros((len(explabel)))

for i in range (len(y)):
    if not np.isnan(explabel[i]):
        y[i]=1
        num1+=1
num0 = int(len(y))-num1
print (len(y),num0,num1)

dataraw0 = np.array(dataraw0) # 
X = dataraw0[:,5:45] #gm 5:46 other 5:45

fold_n = 10
kf = StratifiedKFold(n_splits=fold_n,shuffle=True) # Define the split - into 2 folds
kf.get_n_splits(X)


for trainix, test in kf.split(X,y):
    '''
        # balance train
    y0=[]
    y1=[]
    for i in range(len(train)):
        if y[train[i]]==0:
            y0.append(train[i])
        else:
            y1.append(train[i])

    y00=resample(y0,replace=False,n_samples=len(y1))
    trainix = np.concatenate((y00,y1),axis=0)
    print (len(trainix),len(y0),len(y00),len(y1))
    '''
    clf2 = GradientBoostingClassifier(min_samples_leaf=7, max_depth=7, min_samples_split=50, subsample=0.9,n_estimators=500, learning_rate=0.2, random_state=0 )


    clf2.fit(X[trainix], y[trainix])
    
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
    print ('GB (acc,auc,prc,sn,sp,pre,recall):',round(acc2,4),round(aoauc2,4),round(auprc2,4),round(sn,4),round(sp,4),round(pre,4),round(recall,4))
    
    # random results
    y_predict20 = np.random.choice(y[test], len(y[test]),replace=False)
    acc2 = accuracy_score(y[test], y_predict20)
    probs2 = clf2.predict_proba(X[test])
    probs20 = np.random.choice(probs2[:,1], len(probs2),replace=False)
    fpr, tpr, threshs = roc_curve(y[test],probs20)
    aoauc2 = auc(fpr,tpr)
    precision, recall, thresholds = precision_recall_curve(y[test],probs20)
    auprc2 = auc(recall,precision)
    print ('random (acc,auc,prc):',round(acc2,4),round(aoauc2,4),round(auprc2,4))

