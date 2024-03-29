# Code for Symmetry Constraints Enhance Long-term Stability and Accuracy in Unsupervised Learning of Geophysical Fluid Flows

This pytorch code shows how to train a deep neural network model for solving PDEs. We combined 1D group equivariant convolutional layers with mixed scalar-vector input fields for the neural network. An unsupervised learning approach, physical constraint loss was employed during the training. Here, this example shows to train geophysical fluid flows, 1D Shallow Water Equations. 

## Installation
Create a new Conda-environment. We provide an envrironyment.yaml file for dependencies.

```bash
 conda env create -f environment.yaml
```

## Train a model for 1D Shallow Water Equations with Gaussian bells ICs

```bash
python train.py
```
