## Table of Contents 
- [Program Requirements](#program-requirements)
- [Usage Example](#usage-example)
- [Usage Instructions](#usage-instructions)
- [Homework 2: Polynomial Solver using SGD and TinyGrad](#homework-2-polynomial-solver-using-sgd-and-tinygrad)
  - [Objective](#objective)
  - [Problem](#usage)

## Program Requirements
The minimum requirements to run the program are: `python>=3.8`, `matplotlib`, `pandas`, `scikit-learn`, `tqdm` and [`tinygrad`](https://github.com/geohot/tinygrad) (and its corresponding dependencies).

Assuming you already have `python`, `pip`, and `git` installed, you may run the command below:
```
pip install -r requirements.txt
```
or 
```
pip install -r min_requirements.txt
```

Note that [requirements.txt](requirements.txt) and [min_requirements.txt](min_requirements.txt) both include tinygrad. If you already have tinygrad installed from the link above, you should remove the line corresponding to tinygrad.


## Usage Example
You can run the program in the terminal as shown below. This will assume that both `data_train.csv` and `data_test.csv` exist in the current path.
```
python solver.py
```

You can also run the program in the terminal by specifying the path containing the dataset files as shown below.
```
python solver.py --data sample_data
```

## Homework 2: Polynomial Solver using SGD and TinyGrad

### Objective 
SGD is a useful algorithm with many applications. In this assignment, we will use SGD in the TinyGrad framework as polynomial solver - to find the degree and coefficients.

### Usage:
The application will be used as follows:
```
python3 solver.py
```

The solver will use data_train.csv to estimate the degree and coefficients of a polynomial. To test the generalization of the learned function, it should have small test error on data_test.csv.

The function should be modeled using tinygrad : https://github.com/geohot/tinygrad

Use SGD to learn the polynomial coefficients.