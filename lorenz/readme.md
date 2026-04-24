## Overview

This toy example implements a neural network approximation of the classical **Lorenz-63 system**, following the example from:

- Brunton, S. L., & Kutz, J. N.  
  *Data-Driven Science and Engineering* (Cambridge University Press, 2022)

The goal is to learn a **discrete-time dynamical system** directly from data:

x_{k+1} = NN(x_k)

instead of integrating the governing ODEs.

---

## Features

- Generates training data using `ode45`
- Trains a feedforward neural network to approximate system evolution
- Performs **autoregressive (rollout) prediction**
- Compares predicted trajectories and Jacobians of neural network and analytical solver

## Usage

Run in MATLAB:

```matlab
run NNLorenz.m