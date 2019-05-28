# Learning-Aided 3D Mapping
[![Build Status](https://travis-ci.org/RobustFieldAutonomyLab/la3dm.svg?branch=master)](https://travis-ci.org/RobustFieldAutonomyLab/la3dm)

A suite of algorithms for learning-aided mapping. Includes implementations of Gaussian process regression and Bayesian generalized kernel inference for occupancy prediction using test-data octrees. This framework also contains the components necessary to run OctoMap as a baseline.

## Overview

This implementation as it stands now is primarily intended to enable replication of these methods over a few datasets. In addition to the implementation of relevant learning algorithms and data structures, we provide two sets of range data (sim_structured and sim_unstructured) collected in Gazebo for demonstration. Parameters of the sensors and environments are set in the relevant `yaml` files contained in the `config/datasets` directory, while configuration of parameters for the mapping methods can be found in `config/methods`.

## Getting Started

### Dependencies

We tested LA3DM with ROS Kinetic, but it also works with ROS Indigo, just ensure you have the correct dependencies by running:

```bash
$ sudo apt-get install ros-kinetic-octomap*
```
if you're using ROS Kinetic, or:

```bash
$ sudo apt-get install ros-indigo-octomap*
```
if you're using Indigo.

### Building with catkin

The repository is set up to work with catkin, so to get started you can clone the repository into your catkin workspace `src` folder and compile with `catkin_make`:

```bash
my_catkin_workspace/src$ git clone https://github.com/RobustFieldAutonomyLab/la3dm
my_catkin_workspace/src$ cd ..
my_catkin_workspace$ catkin_make
my_catkin_workspace$ source ~/my_catkin_workspace/devel/setup.bash
```

## Running the Demo

To run the demo on the `sim_structured` environment, simply run:

```bash
$ roslaunch la3dm la3dm_static.launch
```

which by default will run using the BGKOctoMap method. If you want to try a different method or dataset, simply pass the
name of the method or dataset as a parameter. For example, if you want to run standard OctoMap on the `sim_unstructured` map,
you would run:

```bash
$ roslaunch la3dm la3dm_static.launch method:=octomap dataset:=sim_unstructured
```

## Relevant Publications

If you found this code useful, please cite the following:

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

Bayesian Generalized Kernel Inference for Occupancy Map Prediction ([PDF](http://personal.stevens.edu/~benglot/Doherty_Wang_Englot_ICRA_2017.pdf))
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

Jinkun Wang and Kevin Doherty, [Robust Field Autonomy Lab (RFAL)](http://personal.stevens.edu/~benglot/index.html), Stevens Institute of Technology.
