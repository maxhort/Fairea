# Installation and Usage Instructions

This document contains information on how to install packages and datasets.

---

## Installation

### Python
For our experiments, we used Python 3.7. Python 3.6 is possible as well.
Furthermore, we require the following Python packages:
```bash
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
```



## AIF360
We are using the AI Fairness 360 toolkit for bias mitigation methods and to compute fairness metrics.
More information on AIF360 can be found here: https://github.com/Trusted-AI/AIF360

### Download AIF360
AIF360 can be installed as follows:
```bash
pip install aif360
```

### Download datasets
We use three datasets for our experiments, which can be retrieved online:
1. Adult: https://archive.ics.uci.edu/ml/datasets/adult
2. COMPAS: https://github.com/propublica/compas-analysis
3. German: https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

For more information on the download of datasets, we refer to: https://github.com/Trusted-AI/AIF360/tree/master/aif360/data.