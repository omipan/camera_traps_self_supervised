# Focus on the Positives: Self-Supervised Learning for Biodiversity Monitoring
This repository contains the code for reproducing the results of our ICCV 2021 [paper](https://arxiv.org/abs/2108.06435).

![Overview of Context approach](figs/siamese_net.png)


# WORK IN PROGRESS 
We are in the process of updating this repository. 

The organization of the repository is the following:

* `requirements.txt` Contains the libraries needed to run the code
* `demo.sh` is a sample script that launches parameterized training.
* `main.py` is the main process that orchestrates model training and evaluation.
* `models.py` Model utility functions.
* `datasets.py` Data utility functions i.e. Dataset classes
* `losses.py` has a set of supported loss functions.
* `evaluate.py` contains the code for evaluating model.
* `utils.py` contains useful utility functions, such as feature extraction, linear evaluation etc.


## Data
A large repository of camera trap datasets can be found at [lila.science](http://lila.science/), including Caltech Camera Traps (CCT20), Island Conservation Camera Traps (ICCT) and Snapshot Serengeti which were used for our experiments  


### Getting started
*  Add images and their metadata files under directory cam_data/ i.e. cam_data/cct20/train_images/* , cam_data/cct20/cct20_context_file.csv
* Install libraries based on requirements.txt

## Reference  
If you find our work useful in your research please consider citing our paper:  

```latex
@inproceedings{PantazisICCV2021,
  title={Focus on the Positives: Self-Supervised Learning for Biodiversity Monitoring},
  author={Pantazis, Omiros and 
          Brostow, Gabriel and 
          Jones, Kate and 
          Mac Aodha, Oisin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```
