# Explanatory package for "Fairea: A Model Behaviour Mutation Approach to Benchmarking Bias Mitigation Methods",
This on-line appendix is supplementary to the paper entitled "Fairea: A Model Behaviour Mutation Approach to Benchmarking Bias Mitigation Methods", which has been submitted for review at FSE'21. It contains the raw results, Python code for the proposed approach, and scripts to replicate our experiments. 

This `README` file describes the structure of the provided files (source code and results). as wel information on the content of this package.

For more detailed instructions on how to install datasets and required libraries, refer to the `INSTALL.md` file.
---

## Content

### "Scripts" directory
This folder contains files used to run bias mitigation methods and obtain their performace (accuracy, fairness). Additionally, "get_values.py" is used to perform the prediction overwritting process on an original classification model.
"utility.py" provides additional functionality to obtain datasets from AIF360, and unmodified classification models (LR, DT, SVM).
The remaining files are implementation of bias mitigation methods, with the help of AIF360.

### "Results" directory

This directory contains all the experimental results to answer the research questions in our paper.  
The result files are organized according to the RQ they answer. 
1. RQ1-RQ2: This directory contains results for the performance of the bias mitigation methods (pre-processing and post-processing) for the three classification models (LR, DT, SVM). The performance is evaluated for each dataset and protected attribute and repeated for 50 different datasplits.
Each file contains 50 lines, each representing the results of a single repetition of the respective experiment. Each line lists "SPD Accuracy AOD" space separated. F
Filenames follow this pattern: "method_classifier_dataset_attribut". For CO and ROC, filenames are extend with respective fairness metrics used by the respective method.

2. RQ3 (adversarial, prejudice): These results are divided on two folders, one for each in-processing method (Adversarial Debiasing and Prejudice Remover).
	Each folder contains two types of files. 
	One contains the experimental result for different parameter values (eta or weights) similar to the results of RQ1-RQ2. These filenames start with "eta" or "weights".
    The other type of files contain the results of the overwritting stage on the original model (without bias mitigation).
	The files start with the a parameter value, followed by 50 lines with experimental results for the respective parameter value.
	The other file type contains the fairness and accuracy of the baseline with two mutation strategies (0,1).
3. RQ4:	The results for RQ4 contain information on the fairness and accuracy achieved by three different mutation strategies (0, Random, 1).
Accuracy, SPD and AOD values for the three strategies are given in consecutive lines (in respective order (0, Random, 1)).
	Each line contains 11 values, the first with 0% mutation, followed by 10%, 20%, ..., 100%.

### "Example.ipynb"
This jupyter notebook provides a visual example on how Fairea can be used to benchmark the performance of bias mitigation methods. Furthermore, the bias mitigation method is accessed for bias mitigation methods that achieve a "good trade-off".

This purpose of Example.ipynb is to show how Fairea could be used by practicioners.
### "Fairea" directory
This directory contains the source code used in "Example.ipynb".
It contains the following files:

1. utility.py: Additional functions to write content to files and obtain datasets from AIF360.
2. Fairea.py: Functions used to create baselines, normalize, categorize mitigation region and compute the area of a bias mitigation method in accordance to the baseline.

