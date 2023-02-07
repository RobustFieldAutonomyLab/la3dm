# Learning-Aided 3D Mapping
[![Build Status](https://travis-ci.org/RobustFieldAutonomyLab/la3dm.svg?branch=master)](https://travis-ci.org/RobustFieldAutonomyLab/la3dm)

A suite of algorithms for learning-aided mapping. Includes implementations of Gaussian process regression and Bayesian generalized kernel inference for occupancy prediction using test-data octrees. 

## Overview

This implementation as it stands now is primarily intended to enable replication of these methods over a few datasets. In addition to the implementation of relevant learning algorithms and data structures, we provide two sets of range data (sim_structured and sim_unstructured) collected in Gazebo for demonstration. Parameters of the sensors and environments are set in the relevant `yaml` files contained in the `config/datasets` directory, while configuration of parameters for the mapping methods can be found in `config/methods`.

## Getting Started

### Dependencies

The current package runs with ROS Noetic, but for testing in ROS Kinetic and ROS Indigo, you can set the CMAKE flag in the CMAKELists file to c++11.

Octomap is a dependancy, which can be installed using the command below. Change distribution as necessary.

```bash
$ sudo apt-get install ros-noetic-octomap*
```

### Building with catkin

The repository is set up to work with catkin, so to get started you can clone the repository into your catkin workspace `src` folder and compile with `catkin_make`:

```bash
my_catkin_workspace/src$ git clone https://github.com/RobustFieldAutonomyLab/la3dm.git
my_catkin_workspace/src$ cd ..
my_catkin_workspace$ catkin_make
my_catkin_workspace$ source ~/my_catkin_workspace/devel/setup.bash
```

## Running the Demo

To run the demo on the `sim_structured` environment, simply run:

```bash
$ roslaunch la3dm la3dm_static.launch
```

which by default will run using the BGKOctoMap-LV method. If you want to try a different method or dataset, simply pass the
name of the method or dataset as a parameter. For example, if you want to run GPOctoMap on the `sim_unstructured` map,
you would run:

```bash
$ roslaunch la3dm la3dm_static.launch method:=gpoctomap dataset:=sim_unstructured
```

## Relevant Publications

If you found this code useful, please cite the following:

Improving Obstacle Boundary Representations in Predictive Occupancy Mapping([PDF](https://www.sciencedirect.com/science/article/abs/pii/S0921889022000380))

```
@article{pearson2022improving,
  title={Improving Obstacle Boundary Representations in Predictive Occupancy Mapping},
  author={Pearson, Erik and Doherty, Kevin and Englot, Brendan},
  journal={Robotics and Autonomous Systems},
  volume={153},
  pages={104077},
  year={2022},
  publisher={Elsevier}
}
```

Learning-Aided 3-D Occupancy Mapping with Bayesian Generalized Kernel Inference ([PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8713569))
```
@article{Doherty2019,
  doi = {10.1109/tro.2019.2912487},
  url = {https://doi.org/10.1109/tro.2019.2912487},
  year = {2019},
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
  pages = {1--14},
  author = {Kevin Doherty and Tixiao Shan and Jinkun Wang and Brendan Englot},
  title = {Learning-Aided 3-D Occupancy Mapping With Bayesian Generalized Kernel Inference},
  journal = {{IEEE} Transactions on Robotics}
}
```

Fast, accurate gaussian process occupancy maps via test-data octrees and nested Bayesian fusion ([PDF](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7487232))
```
@INPROCEEDINGS{JWang-ICRA-16,
author={J. Wang and B. Englot},
booktitle={2016 IEEE International Conference on Robotics and Automation (ICRA)},
title={Fast, accurate gaussian process occupancy maps via test-data octrees and nested Bayesian fusion},
year={2016},
pages={1003-1010},
month={May},
}
```

Bayesian Generalized Kernel Inference for Occupancy Map Prediction ([PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989356))
```
@INPROCEEDINGS{KDoherty-ICRA-17,
author={K. Doherty and J. Wang, and B. Englot},
booktitle={2017 IEEE International Conference on Robotics and Automation (ICRA)},
title={Bayesian Generalized Kernel Inference for Occupancy Map Prediction},
year={2017},
month={May},
}
```

## Contributors

Jinkun Wang, Kevin Doherty and Erik Pearson, [Robust Field Autonomy Lab (RFAL)](https://robustfieldautonomylab.github.io/), Stevens Institute of Technology.
