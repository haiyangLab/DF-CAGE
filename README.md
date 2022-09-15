# DF-CAGE
DF-CAGE is a novel machine learning-based method for cancer-druggable gene discovery. The input for the framework is copy number, mRNA, DNA methylation, and other omics data. The output is the classification score vector. DF-CAGE is mainly divided into two components: 1. The multi-granularity scan module is used to obtain latent variables from each type of omics data. 2. The cascaded forest module determines the classification score vector for each sample.

Stored in the input directory is the input data of DF-CAGE from the three gene sets of Oncokb, Target, and Drugbank. The results of the 465 druggable genes predicted by the DF-CAGE model are stored in the results directory, and we divide the 465 genes into known, reliable, and potential gene sets, among which the known gene sets contain the druggable genes of OncoKB and Target.
```shell
# The data in the Input folder is the extracted feature data, now cross-validated on the OncoKB dataset with the following command： 
python oncoKB.py   
# The predicted results of ~20,000 protein-coding region genes are stored in ./results/  
```
The same performance runs on the target dataset as follows：
```shell
# Shown are the results under different ratios of positive and negative samples, taking positive and negative samples 1:1 as an example.You can modify the parameters and you can experiment with the other ratios separately
python TARGET.py 
```
During the experiment on drugBank database, the data related to the experiment should be obtained by regular expression as follows:
```shell
# Start with a sample of cancer drug related data from drugBank database.
python drug_targets.py
# Subsequently, the OncoKB dataset was used for training and the DrugBank collection was used for prediction, resulting in performance runs as follows.
python drugbank.py
```
The predicted results of ~20,000 protein-coding region genes:
```shell
Python exp.py
```
DF-CAGE is based on the Python program language. The deep learning network's implementation was based on Numpy 1.19.2, Scipy 1.4.1, Scikit-learn 0.24.1, Matplotlib 3.3.4, Pandas 1.2.4, joblib 1.1.0. After the testing, DF-CAGE can operate normally.
