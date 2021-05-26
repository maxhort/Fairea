import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utility import get_data,write_to_file,get_classifier
import os
import argparse
import datetime
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover

print (datetime.datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="adult",
                    help="Dataset name")
parser.add_argument("-c", "--clf", type=str, default="lr",
                    help="Classifier name")
parser.add_argument("-p", "--protected", type=str, default="sex",
                    help="Protected attribute")

parser.add_argument("-s", "--start", type=int, default=0,
                    help="Start")
parser.add_argument("-e", "--end", type=int, default=5,
                    help="End")
parser.add_argument("-b", "--batch", type=int, default=0,
                    help="Batch")
args = parser.parse_args()

scaler = StandardScaler()
dataset_used = args.dataset # "adult", "german", "compas"
attr = args.protected
clf_name = args.clf
start = args.start
end = args.end
batch = args.batch

val_name = "pr_{}_{}_{}_{}.txt".format(clf_name,dataset_used,attr,batch)
val_name= os.path.join("results",val_name)

dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr)


etas = [1,5,10,25]
etas = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,75,100]
etas = [x for x in range(batch*14,(batch+1)*14)]

for e in etas:
    write_to_file(val_name,str(e))
    print ("eta",e)
    hist = []
    for i in range(start,end):
        np.random.seed(i)
        # Split into train, validation, and test
        #dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
        dataset_orig_test.features = scaler.transform(dataset_orig_test.features)
        
        prejudice = PrejudiceRemover(sensitive_attr="sex",eta=e)
        prejudice = prejudice.fit(dataset_orig_train)
        pred = prejudice.predict(dataset_orig_test)
        
        class_metric = ClassificationMetric(dataset_orig_test, pred,
                        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        
        stat = abs(class_metric.statistical_parity_difference())
        #aod = abs(class_metric.average_odds_difference())
        aod = abs(class_metric.average_abs_odds_difference())
        #print (size,abs(stat),accuracy)
        content = "{} {} {}".format(stat,class_metric.accuracy(),aod)
        write_to_file(val_name,content)
        hist.append([stat,class_metric.accuracy(),aod])
    res = np.array(hist)
    print (np.mean(res,axis = 0))

print (datetime.datetime.now())