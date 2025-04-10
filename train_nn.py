#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch import optim, utils
import torch.multiprocessing
import pytorch_lightning as pl
import torchio as tio
import random
import timm_3d

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from split_names import get_cv_split_from_csv

class SimplifiedCNN3D(nn.Module):
    def __init__(self, dropout_frac):
        super(SimplifiedCNN3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        self.bn3 = nn.BatchNorm3d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout_frac)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation function here

        return x

class LCTClassificationModule(pl.LightningModule):
    def __init__(self, weight_decay, dropout_frac, drop_path_frac, init_lr, augmentation_hyperparameters, network):
        super().__init__()
        self.weight_decay = weight_decay
        self.init_lr = init_lr
        self.augmentation_hyperparameters = augmentation_hyperparameters
        
        if network == "deep":
            self.model = timm_3d.create_model(
                'resnet18.a1_in1k',
                pretrained=False,
                num_classes=2,
                in_chans=1,
                drop_rate=dropout_frac,
                drop_path_rate = drop_path_frac
            )
        elif network == "shallow":
            self.model = SimplifiedCNN3D(dropout_frac)

        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        out = self.model(x)
        
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1)
        error = torch.mean((pred != y).float())
        
        ce = self.loss_ce(out, y)
        self.log('ce_train', ce, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('error_train', error, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return ce
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        
        ce = self.loss_ce(out, y)
        
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1)
        error = torch.mean((pred != y).float())
        
        combined = error + 0.001 * ce
        
        self.log('ce_val', ce, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('error_val', error, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('comb_val', combined, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return combined

    def configure_optimizers(self):
        lr = self.init_lr
        wd = self.weight_decay
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "comb_val"}



class AugmentedCTDataset(torch.utils.data.Dataset):
    
    def _load_scan_metadata(self, scans_path, category_names, scans):
        self.file_names = []
        self.labels = []
        
        for name, label in scans:
            file_name = str(scans_path / category_names[label] / self.fn_template.format(name))
            self.file_names.append(file_name)
            self.labels.append(label)
            
    def _make_augmentations(self):
        blur = tio.transforms.RandomBlur(self.hp["blur_max_std"], p=self.hp["blur_p"])
        
        affine = tio.RandomAffine(
            scales = (1/self.hp["scale_max"], self.hp["scale_max"]),
            degrees = (self.hp["rotate_max_x"], self.hp["rotate_max_y"], self.hp["rotate_max_z"]),
            translation = self.hp["translate_max"],
            default_pad_value = -1000,
            p = self.hp["affine_p"])
        
        self.augmentations = tio.transforms.Compose([blur, affine])


    def __init__(self, scans_path, category_names, scans, mean, std, apply_augmentations, hyperparameters=None, fn_template = "{}_masked.nii.gz"):
        self.apply_augmentations = apply_augmentations
        self.hp=hyperparameters
        self.mean = mean
        self.std = std
        self.fn_template = fn_template
        
        if (apply_augmentations == True) and (hyperparameters is None):
            raise Exception("hyperparameters should be provided when augmentations are applied")
        
        self._load_scan_metadata(scans_path, category_names, scans)
        
        if self.apply_augmentations:
            self._make_augmentations()
        
    def __getitem__(self, index):
        tio_img = tio.ScalarImage(self.file_names[index])
        label = self.labels[index]
        
        if self.apply_augmentations:
            tio_img = self.augmentations(tio_img)
        
        img = tio_img.tensor
        
        if self.apply_augmentations and random.random() < self.hp["flip_x_p"]:
            img = torch.flip(img, [1])
        if self.apply_augmentations and random.random() < self.hp["flip_y_p"]:
            img = torch.flip(img, [2])
        if self.apply_augmentations and random.random() < self.hp["flip_z_p"]:
            img = torch.flip(img, [3])
        
        if self.apply_augmentations and random.random() < self.hp["noise_p"]:
            noise_std = random.uniform(0, self.hp["noise_max_std"])
            img += torch.randn_like(img)*noise_std
            
        img = (img-self.mean)/self.std
            
        return img, label

    def __len__(self):
        return len(self.labels)


def objective(trial, experiment_folder, data_path, category_names, train_names, val_names, network):
    trial_folder = experiment_folder / f"trial_{trial.number}"
    trial_folder.mkdir()
    
    # not the real mean and std, but this way the foreground (~150) is set close to 1
    # and the background (~1000) is set close to -1
    mean = -425
    std = 575
    
    augmentation_hyperparameters = {
        "blur_max_std" : trial.suggest_float("blur_max_std", 0, 1),
        "blur_p" : trial.suggest_float("blur_p", 0, 1),
        
        "translate_max" : trial.suggest_float("max_translate", 0, 10),
        "scale_max" : trial.suggest_float("max_scale", 1, 1.2),
        "rotate_max_x" : trial.suggest_float("max_rotate_x", 0, 180),
        "rotate_max_y" : trial.suggest_float("max_rotate_y", 0, 180),
        "rotate_max_z" : trial.suggest_float("max_rotate_z", 0, 180),
        "affine_p" : trial.suggest_float("affine_p", 0, 1),
        
        "flip_x_p" : trial.suggest_float("flip_x_p", 0, 0.5),
        "flip_y_p" : trial.suggest_float("flip_y_p", 0, 0.5),
        "flip_z_p" : trial.suggest_float("flip_z_p", 0, 0.5),
        
        "noise_max_std" : trial.suggest_float("noise_max_std", 1, 150),
        "noise_p" : trial.suggest_float("noise_p", 0, 1)
    }
    
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1.0, log=True)
    init_lr = trial.suggest_float("init_lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 2, 5)
    
    dropout_frac = trial.suggest_float("dropout_frac", 0, 0.75)
    if network == "deep":
        drop_path_frac = trial.suggest_float("drop_path_frac", 0, 0.75)
    else:
        drop_path_frac = None
    

    train_dataset = AugmentedCTDataset(data_path, category_names, train_names, mean, std, True, augmentation_hyperparameters)
    val_dataset = AugmentedCTDataset(data_path, category_names, val_names, mean, std, False)
    
    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=6, shuffle=True, drop_last=True, persistent_workers=True)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    l_module = LCTClassificationModule(weight_decay, dropout_frac, drop_path_frac, init_lr, augmentation_hyperparameters, network)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=trial_folder / "checkpoints" / "comb_val",
        filename="best_comb_val_{epoch}_{comb_val:.10f}",
        monitor="comb_val",
        save_top_k=1,
        mode="min")
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor = "comb_val",
        patience = 200
        )

    callbacks = [checkpoint_callback, early_stopping_callback]
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=trial_folder / "logs")
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        # For multi-GPU training the ddp_spawn strategy should be used for good
        # interoperability between PyTorch and Optuna
        #strategy="ddp_spawn",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks = callbacks)
    trainer.fit(l_module, train_loader, val_loader)
    
    checkpoint_filename = next((trial_folder / "checkpoints" / "comb_val").glob("best_comb_val_*")).name
    best_model_score = float(checkpoint_filename[checkpoint_filename.find("_comb_val=")+10:-5])
    
    return best_model_score
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chicken organ classification network.")
    parser.add_argument("--organ", default="hearts", help="Type of organ: 'hearts' or 'livers'.")
    parser.add_argument("--storage", help="URL to storage location (e.g. MySQL server) for Optuna.")
    parser.add_argument("--cv_split", default=0, help="Cross-validation split")
    parser.add_argument("--network", default="deep", help="Network architecture ('deep' or 'shallow')")
    args = parser.parse_args()
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    if args.organ == "hearts":
        data_path = get_data_folder() / "CT SCANS OF HEARTS"
        splits_path = data_path / "cross_validation_splits"
        train_names = get_cv_split_from_csv(splits_path / f"train_{args.cv_split}.csv")
        val_names = get_cv_split_from_csv(splits_path / f"val_{args.cv_split}.csv")
        objective_with_args = functools.partial(
            objective,
            experiment_folder = experiment_folder,
            data_path = data_path,
            category_names = ["HEAL_H_resized_128", "DIS_H_resized_128"],
            train_names = train_names,
            val_names = val_names,
            network = args.network
        )
        study_name = f"chicken_organs_hearts_{args.network}_cv{args.cv_split}"
    elif args.organ == "livers":
        data_path = get_data_folder() / "CT SCANS OF LIVERS"
        splits_path = data_path / "cross_validation_splits"
        train_names = get_cv_split_from_csv(splits_path / f"train_{args.cv_split}.csv")
        val_names = get_cv_split_from_csv(splits_path / f"val_{args.cv_split}.csv")
        objective_with_args = functools.partial(
            objective,
            experiment_folder = experiment_folder,
            data_path = data_path,
            category_names = ["HEAL_L_resized_128", "DIS_L_resized_128"],
            train_names = train_names,
            val_names = val_names,
            network = args.network
        )
        study_name = f"chicken_organs_livers_{args.network}_cv{args.cv_split}"
    else:
        raise ValueError("--organ argument should be 'hearts' or 'livers'.")
    

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    
    study.optimize(
        objective_with_args,
        n_trials=50,
        timeout=None,
        n_jobs=1
    )
    
