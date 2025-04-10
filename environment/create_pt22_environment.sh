conda create -n pt22 tomosipo pytorch=2.2 pytorch-cuda=12.1 torchvision=0.17 pytorch-lightning=2.2.1 astra-toolbox=2.1.3 captum optuna optuna-dashboard simpleitk mysqlclient ocl-icd-system matplotlib scikit-image albumentations jsonargparse tensorboard opencv tqdm tifffile pandas seaborn -c pytorch -c nvidia -c astra-toolbox -c aahendriksen -c simpleitk -c conda-forge
conda activate pt22
pip install torcheval
pip install timm-3d
