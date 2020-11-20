# MED-LOADER
This repository contains code for downloading, sorting, extracting (to [.mha i.e. a MetaImage](https://itk.org/Wiki/ITK/MetaIO/Documentation#Quick_Start)) and viewing CT slices from open datasets like:-
1. [MICCAI 2015 Head and Neck (HaN) Segmentation Challenge](http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge)


## Usage
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) with python3.7
2. Install [git](https://git-scm.com/downloads)
3. Open a terminal and follow the commands
    - Clone this repository
        - `git clone git@github.com:prerakmody/medloader.git`
    - Install
        - `cd medloader`
        - `conda env create --file environment.yml --name medloader`  (Ensure it is in UTF-8 encoding)
            - _this will take time as it need to download and install all packages_
    - Use
        - `conda activate medloader`
        - `conda develop .`
            - Adds medloader as a python pacakge on your local machine
4. To download/sort/extract the MICCAI 2015 dataset
    - Keep [medloader.dataloader.config.VOXEL_RESO](./medloader/dataloader/config.py) as an empty tuple `=()` if you dont want to resmaple
        - Time consuming step (_but it is recommended that all 3D volumes have the same pixel spacing_)
    - `python` [demo/tf_han_miccai2015_loader.py](./demo/tf_han_miccai2015_loader.py)
        - This shall create a `./_data/HaN_MICCAI2015` directory with `raw/` and `processed/` data files for each patient
        - If [medloader.dataloader.config.VOXEL_RESO](./medloader/dataloader/config.py) is kept empty, then also set the `resampled` flag to `False` in [demo.tf_han_miccai2015_trainer.params.dataloader](./demo/tf_han_miccai2015_trainer.py)
    - Note
        - You may want to remove patient-id=`0522c0125` due to its small dimensions in the z-axis
            - If medloader.dataloader.config.HaN_MICCAI2015['GRID_3D']['SIZE'] = [96,96,96] 
                - Remove it from _data/HaN_MICCAI2015/processed/train/data_3D/{*.csv}
5. To train a model
    - Run `python` [demo/tf_han_miccai2015_trainer.py](./demo/tf_han_miccai2015_trainer.py)
        - You can change the params within that file if need be
    - Note: 
        - To enable [demo.tf_han_miccai2015_trainer.params.model.profiler](./demo.tf_han_miccai2015_trainer.py)
            - To run training script without sudo: [Instrutions by Nvidia](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag)

## CleanUp
1. Remove conda env
    - `conda env remove -n medloader`

# TODO
 - [ ] Conbvert tensorflow code to PyTorch