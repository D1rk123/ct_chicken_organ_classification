#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

from folder_locations import get_data_folder

def shuffle_groups(names):
    # Shuffles names within each group, but maintains the group order
    # The group is decided based on the first two letters of each string
    shuffled_names = []
    curr_group = []
    for i, name in enumerate(names):
        if len(curr_group) > 0 and name[0:2] != curr_group[0][0:2]:
            random.shuffle(curr_group)
            shuffled_names += curr_group
            curr_group = []
        curr_group.append(name)
    
    random.shuffle(curr_group)
    shuffled_names += curr_group
    
    return shuffled_names

def write_to_csv(names, file_path):
    with open(file_path, "w") as file:
        for name in names:
            file.write(f"{name[1:-14]}, {name[0]}\n")

if __name__ == "__main__":
    
    hearts_paths = (
        get_data_folder() / "CT SCANS OF HEARTS",
        ["HEAL_H", "DIS_H"]
    )
    livers_paths = (
        get_data_folder() / "CT SCANS OF LIVERS",
        ["HEAL_L", "DIS_L"]
    )
    num_splits = 5
    test_step = 10
    test_start_step = test_step // num_splits
    val_step = 3
    
    for scans_path, category_names in (livers_paths, ):#(hearts_paths, livers_paths):
        print(f"\n{scans_path}")
        category_paths = [scans_path / category_name for category_name in category_names]

        all_names = []
        for i, category_path in enumerate(category_paths):
            all_names += [str(i)+f.name for f in category_path.glob("*.nii.gz")]
                
        all_names = shuffle_groups(sorted(all_names))
        
        train_names_list = []
        val_names_list = []
        test_names_list = [sorted(all_names[i*test_start_step::test_step]) for i in range(num_splits)]
        print(len(test_names_list))
        
        for test_names in test_names_list:
            other_names = [name for name in all_names if name not in test_names]
            val_names = sorted(shuffle_groups(other_names)[::val_step])
            train_names = sorted([name for name in other_names if name not in val_names])
            
            val_names_list.append(val_names)
            train_names_list.append(train_names)
        
        output_folder = scans_path / "cross_validation_splits"
        output_folder.mkdir()
        for i, train_names, val_names, test_names in zip(range(num_splits), train_names_list, val_names_list, test_names_list):
            print(f"train=({len(train_names)}): {[(name[1:-14], name[0]) for name in train_names]}")
            print(f"val=({len(val_names)}): {[(name[1:-14], name[0]) for name in val_names]}")
            print(f"test=({len(test_names)}): {[(name[1:-14], name[0]) for name in test_names]}")
            write_to_csv(train_names, output_folder / f"train_{i}.csv")
            write_to_csv(val_names, output_folder / f"val_{i}.csv")
            write_to_csv(test_names, output_folder / f"test_{i}.csv")
            print()
            print()