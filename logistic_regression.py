#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import argparse
from torcheval.metrics import BinaryAUROC

import ct_experiment_utils as ceu
from split_names import get_cv_split_from_csv
from folder_locations import get_results_folder, get_data_folder

def load_train_test_val(csv_path, train_names, val_names, test_names):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    
    train_names = [tup[0] for tup in train_names]
    val_names = [tup[0] for tup in val_names]
    test_names = [tup[0] for tup in test_names]
    
    with open(csv_path, "r") as file:
        for i, line in enumerate(file):
            if i==0:
                continue
            if line.find(",") == -1:
                break
            parts = line.split(",")
            x = float(parts[2])
            y = int(parts[1])
            sample_name = parts[0][:-14]
            if sample_name in test_names:
                test_x.append(x)
                test_y.append(y)
            elif sample_name in val_names:
                val_x.append(x)
                val_y.append(y)
            elif sample_name in train_names:
                train_x.append(x)
                train_y.append(y)
            else:
                raise Exception(f"Sample name ({sample_name}) not found in any set.")
                
    train_x = np.array(train_x)[:,None]
    val_x = np.array(val_x)[:,None]
    test_x = np.array(test_x)[:,None]
    
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    test_y = np.array(test_y)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, train_names, val_names, test_names

def get_cv_split_names(path, num):
    return (
        get_cv_split_from_csv(path / f"train_{num}.csv"),
        get_cv_split_from_csv(path / f"val_{num}.csv"),
        get_cv_split_from_csv(path / f"test_{num}.csv")
        )
    
    
def write_results(csv_folder, clf, x, y, names, organ, split_type, cv_split_nr):
        probs = clf.predict_proba(x)
        probs_pos = probs[:, 1]
        confs = np.max(probs, axis=1)
        preds = clf.predict(x)
        
        confusion_matrix = np.zeros((2, 2), dtype=int)
        raw_filename = f"raw_{organ}_{split_type}_{cv_split_nr}.csv"
        
        with open(csv_folder / raw_filename, "w") as raw_file:
            raw_file.write("label,prediction,confidence,name\n")
            
            for i in range(len(preds)):
                confusion_matrix[y[i], preds[i]] += 1
                raw_file.write(f"{y[i]},{preds[i]},{confs[i]},{names[i]}\n")
            
        TN = confusion_matrix[0,0]
        TP = confusion_matrix[1,1]
        FN = confusion_matrix[1,0]
        FP = confusion_matrix[0,1]
        
        accuracy = (TN+TP)/(TN+TP+FN+FP)
        precision = (TP)/(TP+FP)
        sensitivity = (TP)/(TP+FN)
        specificity = (TN)/(TN+FP)
        
        auc_metric = BinaryAUROC()
        auc_metric.update(torch.tensor(probs_pos, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
        AOC = auc_metric.compute()
        
        metrics_filename = f"metrics_{organ}_{split_type}_{cv_split_nr}.csv"
        with open(csv_folder / metrics_filename, "w") as metrics_file:
            metrics_file.write("organ,cv_split_nr,set,TN,TP,FN,FP,accuracy,precision,recall/sensitivity,specificity,AOC\n")
            metrics_file.write(f"{organ},{cv_split_nr},{split_type},{TN},{TP},{FN},{FP},{accuracy},{precision},{sensitivity},{specificity},{AOC}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and apply logistic regression classification.")
    parser.add_argument("--organ", default="hearts", help="Type of organ: 'hearts' or 'livers'.")
    args = parser.parse_args()
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    csv_folder = experiment_folder / "CSVs"
    csv_folder.mkdir()
    
    # First run calc_MGV.py to generate these csv files
    if args.organ == "hearts":
        csv_path = "hearts_MGV.csv"
        cv_folder = (get_data_folder()
                     / "CT SCANS OF HEARTS"
                     / "cross_validation_splits")
    else:
        csv_path = "livers_MGV.csv"
        cv_folder = (get_data_folder()
                     / "CT SCANS OF LIVERS"
                     / "cross_validation_splits")
    
    for num in range(5):
        print(f"Split num {num}")
        train_x, train_y, val_x, val_y, test_x, test_y, train_names, val_names, test_names = \
            load_train_test_val(csv_path,
                                *get_cv_split_names(cv_folder, num))
        
        clf = LogisticRegression(random_state=0, fit_intercept=True, C=1)
        clf.fit(train_x, train_y)
        
        write_results(csv_folder, clf, train_x, train_y, train_names, args.organ, "train", num)
        write_results(csv_folder, clf, val_x, val_y, val_names, args.organ, "val", num)
        write_results(csv_folder, clf, test_x, test_y, test_names, args.organ, "test", num)