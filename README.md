# DF-CAGE
DF-CAGE is a novel machine learning-based method for cancer-druggable gene discovery. The input for the framework is copy number, mRNA, DNA methylation, and other omics data. The output is the classification score vector. DF-CAGE is mainly divided into two components: 1. The multi-granularity scan module is used to obtain latent variables from each type of omics data. 2. The cascaded forest module determines the classification score vector for each sample.
```shell
# the input list of BRCA omics data set is input.txt. We can use the following command to finish the subtyping process: 
python oncoKB.py -m    
# The predicted results of ~20,000 protein-coding region genes are stored in ./results/  
```
