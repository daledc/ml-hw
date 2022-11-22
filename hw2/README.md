## Table of Contents 
- [Program Requirements](#program-requirements)
- [Usage Example](#usage-example)
- [Usage Instructions](#usage-instructions)
- [Homework 2: Polynomial Solver using SGD and TinyGrad](#homework-2-polynomial-solver-using-sgd-and-tinrygrad)
  - [Objective](#learning-objective)
  - [Problem](#problem)
  - [Solution](#solution)

## Program Requirements
The minimum requirements to run the program are: `python>=3.2`, and [`tinygrad`](https://github.com/geohot/tinygrad) (and its corresponding dependencies).

Assuming you already have `python`, `pip`, and `git` installed, you may run the command below:
```
pip install -r requirements.txt
```

The program has been tested to work on Windows 10 (Build 19045) and PopOS 22.04 (Ubuntu-based distribution) using the specific versions below:
- `python==3.8.15`
- `tinygrad==0.4.0`

## Usage Example
You can run the program in the terminal by specifying an input image path option and output image path option. An example is given below:
```
python solver.py --train data_train.csv --test data_test.csv
```
Note: *Omitting an option will use the default values provided.*

```
python main.py --help
```

## Usage Instructions
1. 


## Homework 2: Polynomial Solver using SGD and TinyGrad

### Objective 
SGD is a useful algorithm with many applications. In this assignment, we will use SGD in the TinyGrad framework as polynomial solver - to find the degree and coefficients.

### Usage:

The application will be used as follows:

python3 solver.py

The solver will use data_train.csv to estimate the degree and coefficients of a polynomial. To test the generalization of the learned function, it should have small test error on data_test.csv.

The function should be modeled using tinygrad : https://github.com/geohot/tinygrad

Use SGD to learned the polynomial coefficients.

### Solution