#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import SimpleITK as sitk
import torchio as tio
from tqdm import tqdm

from folder_locations import get_data_folder


def resize_and_save(data_dir, output_dir, new_shape):
    print(f"Resizing {data_dir} to shape {new_shape}")
    output_dir.mkdir(exist_ok=True)
    transf = tio.transforms.Resize(new_shape)
    for file in tqdm(list(data_dir.glob("*.nii.gz"))):
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(file))
        sitk_img = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
        img = tio.ScalarImage.from_sitk(sitk_img)

        # Resize data
        img = transf(img)

        # Save resized data
        new_file_path = output_dir / file.name
        img.save(new_file_path)
            

if __name__ == "__main__":
    
    liver_folder = get_data_folder() / "CT SCANS OF LIVERS (HEAL_L = 82, DIS_L = 102, n = 184)"
    heart_folder = get_data_folder() / "CT SCANS OF HEARTS (HEAL_H = 137, H_DIS_H = 122, n = 259)"
    for state_name in ["DIS", "HEAL"]:

        resize_and_save(
            liver_folder / f"{state_name}_L",
            liver_folder / f"{state_name}_L_resized_128",
            new_shape = np.array((128, 128, 128)),
        )

        resize_and_save(
            heart_folder / f"{state_name}_H",
            heart_folder / f"{state_name}_H_resized_128",
            new_shape = np.array((128, 128, 128)),
        )
    
