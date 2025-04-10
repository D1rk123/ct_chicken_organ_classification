# ct_chicken_organ_classification

This repository contains code to train classifiers to perform classification between healthy and diseased chicken hearts and livers from CT scans.

## Preparations for running the code
To clone the repository with submodules use the following command:
```
git clone --recurse-submodules git@github.com:D1rk123/ct_chicken_organ_classification.git
```

To run the scripts you need to create an extra script *folder_locations.py* that contains two functions: get\_data\_folder() and get\_results\_folder(). The path returned by get\_data\_folder() has to contain the data. The results will be saved in the path returned by get\_results\_folder(). For example:
```python
from pathlib import Path

def get_data_folder():
    return Path.home() / "scandata"
    
def get_results_folder():
    return Path.home() / "experiments" / "ct_chicken_organ_classification"
```

To create a conda environment that can run the experiments, first insall conda and then run the commands in *environment/create\_pt22\_environment.sh*. The exact environment that was used in the paper is described in environment/pt22\_environment.yml.

