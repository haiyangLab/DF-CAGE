# DF-CAGE
DF-CAGE is a novel machine learning-based method for cancer-druggable gene discovery. The input for the framework is copy number, mRNA, DNA methylation, and other omics data. The output is the classification score vector. DF-CAGE is mainly divided into two components: 1. The multi-granularity scan module is used to obtain latent variables from each type of omics data. 2. The cascaded forest module determines the classification score vector for each sample.
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
