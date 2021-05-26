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

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from aif360.algorithms.preprocessing.reweighing import Reweighing

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="adult",
                    help="Dataset name")
parser.add_argument("-c", "--clf", type=str, default="lr",
                    help="Classifier name")
parser.add_argument("-p", "--protected", type=str, default="sex",
                    help="Protected attribute")

parser.add_argument("-s", "--start", type=int, default=0,
                    help="Start")
parser.add_argument("-e", "--end", type=int, default=50,
                    help="End")
args = parser.parse_args()

scaler = StandardScaler()
dataset_used = args.dataset # "adult", "german", "compas"
attr = args.protected
clf_name = args.clf
start = args.start
end = args.end

val_name = "rew_{}_{}_{}_{}_{}.txt".format(clf_name,dataset_used,attr,start,end)
val_name= os.path.join("results",val_name)

dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr,preprocessed = True)



hist = []
for r in range(start,end):
    print (r)
    np.random.seed(r)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = scaler.transform(dataset_orig_test.features)
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)
    dataset_transf_test = RW.transform(dataset_orig_test)
    

    clf = get_classifier(clf_name)
    clf = clf.fit(dataset_transf_train.features, dataset_transf_train.labels,sample_weight=dataset_transf_train.instance_weights.ravel())
    pred = clf.predict(dataset_orig_test.features).reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_orig_test_pred.labels = pred

    
    class_metric = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                     unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    
    stat = abs(class_metric.statistical_parity_difference())
    #aod = abs(class_metric.average_odds_difference())
    aod = abs(class_metric.average_abs_odds_difference())
    #print (size,abs(stat),accuracy)
    print (stat,class_metric.accuracy(),aod)
    content = "{} {} {}".format(stat,class_metric.accuracy(),aod)
    write_to_file(val_name,content)
    hist.append([stat,class_metric.accuracy(),aod])
a = np.array(hist)
print (np.mean(a,axis=0))