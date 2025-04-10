#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC
import numpy as np
import re
import argparse

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from train_nn import LCTClassificationModule, AugmentedCTDataset
from split_names import get_cv_split_from_csv

def get_best_trial(path):
    ckpt_paths = list(path.glob("trial_*/checkpoints/comb_val/best_comb_val_epoch=*_comb_val=*.ckpt"))
    scores = [float(re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", p.name)[1]) for p in ckpt_paths]
    
    index = np.argmin(scores)
    score = scores[index]
    ckpt_path = ckpt_paths[index]
    trial_nr = int(ckpt_path.parents[2].name[6:])
    
    argument_string = open(ckpt_path.parents[3] / "console_argv.txt", "r").readline()
    organ_letter = next(re.finditer("[l,h]", argument_string[argument_string.find("--organ")+len("--organ"):])).group()
    organ = {"l" : "livers", "h" : "hearts"}[organ_letter]
    cv_split_nr = int(next(re.finditer("\d", argument_string[argument_string.find("--cv_split")+len("--cv_split"):])).group())
    
    return organ, cv_split_nr, trial_nr, score, ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply chicken organ classification network.")
    parser.add_argument("--name", help="Name of the folder of the experiment.")
    args = parser.parse_args()
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    csv_folder = experiment_folder / "CSVs"
    csv_folder.mkdir()
    
    organ, cv_split_nr, trial_nr, score, ckpt_path = get_best_trial(get_results_folder() / args.name)
    print(organ, cv_split_nr, trial_nr, score, ckpt_path)
    
    if organ == "hearts":
        paths = (
            get_data_folder() / "CT SCANS OF HEARTS",
            ["HEAL_H_resized_128", "DIS_H_resized_128"]
        )
    elif organ == "livers":
        paths = (
            get_data_folder() / "CT SCANS OF LIVERS",
            ["HEAL_L_resized_128", "DIS_L_resized_128"],
        )
    
    # not the real mean and std, but this way the foreground (~150) is set close to 1
    # and the background (~1000) is set close to -1
    mean = -425
    std = 575
    
    l_module = LCTClassificationModule.load_from_checkpoint(ckpt_path)
    l_module.freeze()
    l_module.eval()

    for split_type in ["train", "val", "test"]:
        names_path = paths[0] / "cross_validation_splits" / f"{split_type}_{cv_split_nr}.csv"
        selected_names = get_cv_split_from_csv(names_path)
    
        dataset = AugmentedCTDataset(paths[0], paths[1], selected_names, mean, std, False)
        
        class1_softmaxes = []
        confusion_matrix = np.zeros((2, 2), dtype=int)
        labels = []
        preds = []
        
        raw_filename = f"raw_{organ}_{split_type}_{cv_split_nr}.csv"
        with open(csv_folder / raw_filename, "w") as raw_file:
            raw_file.write("label,prediction,confidence,name\n")
            
            for i, tup in zip(range(len(dataset)), selected_names):
                name, _ = tup
                
                img, label = dataset[i]
                out = l_module(img[None,:,:,:].cuda()).cpu()[0]
                softmax = F.softmax(out)
                pred = torch.argmax(softmax)
                conf = torch.max(softmax)
                
                round_pred = np.round(pred.numpy())
                raw_file.write(f"{label},{round_pred},{conf},{name}\n")
                preds.append(round_pred)
                confusion_matrix[label, round_pred.astype(int)] += 1
                class1_softmaxes.append(softmax[1].numpy().item())
                labels.append(label)
            
        TN = confusion_matrix[0,0]
        TP = confusion_matrix[1,1]
        FN = confusion_matrix[1,0]
        FP = confusion_matrix[0,1]
        
        accuracy = (TN+TP)/(TN+TP+FN+FP)
        precision = (TP)/(TP+FP)
        sensitivity = (TP)/(TP+FN)
        specificity = (TN)/(TN+FP)
        
        auc_metric = BinaryAUROC()
        auc_metric.update(torch.tensor(class1_softmaxes, dtype=torch.float64), torch.tensor(labels, dtype=torch.float64))
        AOC = auc_metric.compute()
        
        metrics_filename = f"metrics_{organ}_{split_type}_{cv_split_nr}.csv"
        with open(csv_folder / metrics_filename, "w") as metrics_file:
            metrics_file.write("organ,cv_split_nr,set,TN,TP,FN,FP,accuracy,precision,recall/sensitivity,specificity,AOC\n")
            metrics_file.write(f"{organ},{cv_split_nr},{split_type},{TN},{TP},{FN},{FP},{accuracy},{precision},{sensitivity},{specificity},{AOC}\n")
