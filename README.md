# Deep Meta-Learning Energy-Aware Path Planner for Unmanned Ground Vehicles in Unknown Terrain
This is the implementation of the paper: *"Deep Meta-Learning Energy-Aware Path Planner for Unmanned Ground Vehicles in Unknown Terrain", Visca et al., 2021, DOI: [techrxiv.14812905.v1](https://www.techrxiv.org/articles/preprint/Deep_Meta-Learning_Energy-Aware_Path_Planner_for_Unmanned_Ground_Vehicles_in_Unknown_Terrains/14812905/1)*.

<img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition.gif" width="270"> <img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition2.gif" width="270"> <img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition3.gif" width="270">

## Dataset Collection
A dataset of geometry-energy pairs from different terrain types has already been collected and it is avalilable at `./Dataset/Exp00/data.csv`.

If you want to collect new data, modify and run `collect_dataset.py`.
### Note!
For visualisation, Line 37 in `my_chrono_simulator.py` must be changed with the correct local path to the Chrono Data directory.

## Training Model
The different neural network models have already been trained on the `./Dataset/Exp00/data.csv` dataset and the model weights are available at `./Training/Exp00/log*`.

If you want to create new models, modify `models.py`.

If you want to train new models, modify and run `train_meta.py` or `train_separate_model.py`.

## Experiments
### Note!
For visualisation, Line 37 in `my_chrono_simulator.py` must be changed with the correct local path to the Chrono Data directory.
### Effect of Terrain Transition
In this experiment, the performance of the meta-adaptive path planner are tested, when the vehicle transitions on a new terrain.

Run `planning_simulation_trans_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial vehicle position, etc..

### Effect of Heuristic Function
In this experiment, the effect of different heuristic functions for the meta-adaptive path planner are tested.

Run `planning_simulation_h_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial vehicle position, etc..


## Miscellaneous
### Terrain Types and SCM Parameters
Deformable terrains are modelled using the Project Chrono [[1]](#1) implementation of the Soil Contact Model (SCM) [[2]](#2). The complete list of implemented terrain types and respective terramechanical parameters is given in the image below and at `terrain_list.py`.

<p align="center">
<img src="https://github.com/picchius94/META-UGV/blob/main/Images/terrain_types.png" width="700">
</p>

### Geometry Generator
The geometry of the environments is generated using a Perline Noise algorithm described in [[3]](#3).
For more info check `terrain_generator.py`.

### Path Planning
The file `A_star.py` contains a class and functions that handle many utilities related to path planning. In principle, there is no need to access this file.

## Dependencies
The following dependencies are required:
- numpy
- matplotlib
- pandas
- pychrono
- tensorflow
- opensimplex




## References
<a id="1">[1]</a> 
A. Tasora, R. Serban, H. Mazhar, A. Pazouki, D. Melanz, J. Fleischmann, M. Taylor, H. Sugiyama, and D. Negrut, “Chrono: An open source multi-physics dynamics engine” in International Conference on High Performance Computing in Science and Engineering. Springer, 2015, pp. 19–49.

<a id="2">[2]</a>
F. Buse, R. Lichtenheldt, and R. Krenn, “Scm-a novel approach for soil deformation in a modular soil contact model for multibody simulation”, IMSD2016 e-Proceedings, 2016.

<a id="3">[3]</a>
Visca, M., Kuutti, S., Powell, R., Gao, Y., & Fallah, S. (2021). Deep Learning Traversability Estimator for Mobile Robots in Unstructured Environments. arXiv preprint [arXiv:2105.10937](https://arxiv.org/abs/2105.10937).
