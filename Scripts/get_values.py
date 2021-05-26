import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric, utils, ClassificationMetric
from utility import get_data,write_to_file,get_classifier
import pandas as pd
import numpy as np
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="german",
                    help="Dataset name")
parser.add_argument("-c", "--clf", type=str, default="svm",
                    help="Classifier name")
parser.add_argument("-p", "--protected", type=str, default="sex",
                    help="Protected attribute")
args = parser.parse_args()

scaler = StandardScaler()
dataset_used = args.dataset # "adult", "german", "compas"
attr = args.protected
clf_name = args.clf

f_name = "{}_{}_{}.txt".format(clf_name,dataset_used,attr)
dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr)
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

#rand = np.random.randint(2, size=(14653, 1))
ids = [x for x in range(len(dataset_orig_test.labels))]
l = len(dataset_orig_test.labels)
splits = 50
reps = 50
results = defaultdict(lambda: defaultdict(list))
odds = [["0",[0,1]],["0.25",[0.25,0.75]],["0.5",[0.5,0.5]],["0.75",[0.75,0.25]],["1",[1,0]]]
odds = [["0",[0,1]],["0.5",[0.5,0.5]],["1",[1,0]]]
options = [0,1]
options = [1,2] # german


for s in range(splits):
    print ("current split",s)
    np.random.seed(s)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    scaler = StandardScaler()
    dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = scaler.transform(dataset_orig_test.features)
    lmod = get_classifier(clf_name)
    lmod = lmod.fit(dataset_orig_train.features, dataset_orig_train.labels)
    pred = lmod.predict(dataset_orig_test.features).reshape(-1,1)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_orig_test_pred.labels = pred
    for x in range(0,11):
        #print (x)
        size = x/10
        i = int(l*size)
        for name,o in odds:
            hist = []
            for _ in range(reps):
                #rand = np.random.randint(2, size=(i, 1)) + 1
                rand = np.random.choice(options, i, p=o)
                to_change = np.random.choice(ids, size=i, replace=False)
                changed = np.copy(pred)
                #changed = np.concatenate([rand,pred[i:]])
                for t,r in zip(to_change, rand):
                    if name == "swap":
                        changed[t] += 1
                        changed[t] %= 2
                    else:
                        changed[t] = r

                dataset_orig_test_pred.labels = changed
                class_metric = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                 unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                stat = abs(class_metric.statistical_parity_difference())
                #aod = abs(class_metric.average_odds_difference())
                aod = abs(class_metric.average_abs_odds_difference())
                theil = class_metric.theil_index()
                #print (size,abs(stat),accuracy)
                hist.append([stat,class_metric.accuracy(),aod,theil])
            results[x][name] += hist

def to_text(f_,vals):
    v = np.mean(vals,axis=1)
    f_.write(" ".join(str(x) for x in v))
    f_.write("\n")

val_name = os.path.join("values",f_name)
f = open(val_name, "w")

f.write("Accuracy")
f.write("\n")
a0 = np.array([[row[1] for row in results[x]["0"]] for x in range(11)])
to_text(f,a0)
a50 = np.array([[row[1] for row in results[x]["0.5"]] for x in range(11)])
to_text(f,a50)
a1 = np.array([[row[1] for row in results[x]["1"]] for x in range(11)])
to_text(f,a1)
to_plot = [0,"SPD"],[2,"AOD"],[3,"Theil"]


for i, name in to_plot:
    f.write(name)
    f.write("\n")
    p0 = np.array([[row[i] for row in results[x]["0"]] for x in range(11)])
    p50 = np.array([[row[i] for row in results[x]["0.5"]] for x in range(11)])
    p1 = np.array([[row[i] for row in results[x]["1"]] for x in range(11)])
    to_text(f,p0)
    to_text(f,p50)
    to_text(f,p1)
f.close()