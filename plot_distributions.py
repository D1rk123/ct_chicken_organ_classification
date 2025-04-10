#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ct_experiment_utils as ceu
from folder_locations import get_results_folder

if __name__ == "__main__":
    liver_csv_path = "livers_MGV.csv"
    heart_csv_path = "hearts_MGV.csv"
    
    heart_df = pd.read_csv(heart_csv_path, header=0, names=["Day", "Label", "MGV"], converters={0: lambda a : a[0]})
    liver_df = pd.read_csv(liver_csv_path, header=0, names=["Day", "Label", "MGV"], converters={0: lambda a : a[0]})
    
    heart_df["Label"] = heart_df["Label"].map({0: "HEAL_H", 1: "DIS_H"})
    liver_df["Label"] = liver_df["Label"].map({0: "HEAL_L", 1: "DIS_L"})
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    plt.figure(figsize=(9, 3.5))
    plt.subplot(121)
    ax = sns.histplot(data=heart_df, x="MGV", hue="Label", bins=15)
    sns.move_legend(ax, "upper left")
    plt.xlabel("Mean radiodensity (HU)")
    plt.title("Heart mean radiodensity distribution")
    plt.subplot(122)
    ax = sns.histplot(data=liver_df, x="MGV", hue="Label", bins=15)
    sns.move_legend(ax, "upper left")
    plt.title("Liver mean radiodensity distribution")
    plt.xlabel("Mean radiodensity (HU)")
    plt.tight_layout()
    plt.savefig(experiment_folder/"organs_distribution.png", dpi=300)
    plt.show()
        