#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:34:04 2024

@author: des
"""

def get_cv_split_from_csv(path):
    names = []
    with open(path, "r") as file:
        for line in file.readlines():
            if line.find(",") == -1:
                continue
            
            parts = line.split(",")
            names.append((parts[0].strip(), int(parts[1].strip())))
            
    return names