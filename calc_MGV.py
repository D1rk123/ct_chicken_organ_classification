#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torchio as tio

from folder_locations import get_data_folder

if __name__ == "__main__":
    base_folder = get_data_folder()
    
    hearts_folder = base_folder / "CT SCANS OF HEARTS"
    livers_folder = base_folder / "CT SCANS OF LIVERS"
    
    organ_folder_map = {
        "hearts" : (hearts_folder / "HEAL_H", hearts_folder / "DIS_H"),
        "livers" : (livers_folder / "HEAL_L", livers_folder / "DIS_L"),
        }
    
    for organ in ["hearts", "livers"]:
        print(organ)
        print("=======================================================")
        with open(f"{organ}_MGV.csv", "w") as file:
            file.write("filename,label,MGV\n")
            for folder in organ_folder_map[organ]:
                for scan_path in folder.glob("*.nii.gz"):
                    # if the name of the parent folder contains DIS the label is diseased (1)
                    # otherwise it is healthy (0)
                    label = int(scan_path.parent.name.find("DIS") == 0)
                    
                    img = tio.ScalarImage(scan_path)
                    mask = img.tensor > 0
                    # calculate the mean over the masked area
                    MGV = torch.sum(img.tensor*mask)/torch.sum(mask)
                    
                    print(f"{scan_path.name},{label},{MGV}")
                    file.write(f"{scan_path.name},{label},{MGV}\n")