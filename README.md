# Chromatin loop anchors predict exon usage
The codes here are used to explore if the chromatin loop information, i.e. ChIA-PET data, together with other epigenomics and transcriptomics data could contribute to the transcription and exon usage prediction.  
The datasets can be found at: https://researchdata.ntu.edu.sg/dataverse/chrom_pred_exon.

## PUBLICATION

## EXPLANATION

 
```
fold10_tran_2.py -- 10-fold cross validation for transcription prediction.  
chrom_tran_2.py -- chromsome split validation for transcription prediction.  
cross_tran_2.py -- cross cell line validation for transcription prediction.  
fold10_exon_coefftreat_2.py -- 10-fold cross validation for exon usage prediction.  
chrom_exon_coefftreat_2.py -- chromsome split validation for exon usage prediction.  
cross_exon_coefftreat_2.py -- cross cell line validation for exon usage prediction.  
```


## WORKING MECHANISM
The overview of the whole pipeline illustrated in Figure 1.

<img src="https://github.com/mjflab/exon-prediction/blob/main/method.jpg" width="600" height="400">

figure 1.An overview of the pipeline.

## USAGE:
Based on python2.  
tested on Linux

Python modules:  
```
numpy  
pandas  
sklearn
```



## CONTACT
If you have any inqueries, please contact mfullwood@ntu.edu.sg.
