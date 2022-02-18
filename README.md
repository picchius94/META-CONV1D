# Meta-Conv1D Energy-Aware Path Planner for Mobile Robots in Unstructured Terrains
This is the implementation of the paper: *"Meta-Conv1D Energy-Aware Path Planner for Mobile Robots in Unstructured Terrains", Visca et al., 2022.

<img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition.gif" width="270"> <img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition2.gif" width="270"> <img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition3.gif" width="270">

## Dataset Collection
- Run `collect_dataset_handler.py` to collect multiple geometry-energy pairs datasets.
- Run `merge_dataset.py` to merge the datasets
- Run `generate_sum_indices.py` (needed for the models training)

### Note!
For visualisation, Line 37 in `my_chrono_simulator.py` must be changed with the correct local path to the Chrono Data directory.

## Training Model
The different neural network models have already been trained and the model weights are available at `./Training/Exp00/log*`.

If you want to create new models, modify `models.py`.

If you want to train new models, modify and run `train_*.py`.

Run `evaluate_*.py` for evaluating the models on the validation datasets.

## Path Planning
Run `path_planning_experiment_quantitative.py` to test the performance of the three methods, integrated into the path planner, in randomly generated unstructured environments.

All the entries of the dictionary `params` can be changed to modify map size, initial vehicle position, etc..

### Note!
For visualisation, Line 37 in `my_chrono_simulator.py` must be changed with the correct local path to the Chrono Data directory.

### Terrain Types and SCM Parameters
Deformable terrains are modelled using the Project Chrono [[1]](#1) implementation of the Soil Contact Model (SCM) [[2]](#2). The complete list of implemented terrain types and respective terramechanical parameters is given in the image below and at `terrain_list.py`.

<p align="center">
<img src="https://github.com/picchius94/META-UGV/blob/main/Images/terrain_types.png" width="700">
</p>

### Geometry Generator
The geometry of the environments is generated using a Perline Noise algorithm described in [[3]](#3).
For more info check `terrain_generator.py`.


## Dependencies
The following dependencies are required:
- numpy
- matplotlib
- pandas
- pychrono
- tensorflow
- tensorflow-addons
- opensimplex




## References
<a id="1">[1]</a> 
A. Tasora, R. Serban, H. Mazhar, A. Pazouki, D. Melanz, J. Fleischmann, M. Taylor, H. Sugiyama, and D. Negrut, “Chrono: An open source multi-physics dynamics engine” in International Conference on High Performance Computing in Science and Engineering. Springer, 2015, pp. 19–49.

<a id="2">[2]</a>
F. Buse, R. Lichtenheldt, and R. Krenn, “Scm-a novel approach for soil deformation in a modular soil contact model for multibody simulation”, IMSD2016 e-Proceedings, 2016.

<a id="3">[3]</a>
Visca, M., Kuutti, S., Powell, R., Gao, Y., & Fallah, S. (2021). Deep Learning Traversability Estimator for Mobile Robots in Unstructured Environments. arXiv preprint [arXiv:2105.10937](https://arxiv.org/abs/2105.10937).
