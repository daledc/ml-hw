## Table of Contents 
- [Program Requirements](#program-requirements)
- [Usage Example](#usage-example)
- [Usage Instructions](#usage-instructions)
- [Homework 1: Removing Projective Distortion on Images](#homework-1-removing-projective-distortion-on-images)
  - [Learning Objective](#learning-objective)
  - [Problem](#problem)
  - [Solution](#solution)
- [License](#license)

## Program Requirements
The minimum requirements to run the program are: `python>=3.2`, `numpy`, and `opencv-python`.

Assuming you already have `python` and `pip` installed, you may run the command below:
```
pip install -r requirements.txt
```

The program has been tested to work on Windows 10 (Build 19044) and PopOS 22.04 (Ubuntu-based distribution) using the specific versions below:
- `python==3.8.13`
- `numpy==1.23.4`
- `opencv-python==4.6.0.66`

## Usage Example
You can run the program in the terminal by specifying an input image path option and output image path option. An example is given below:
```
python main.py -i input.jpg -o output.jpg
```
Note: *Omitting an option will use the default values provided.*

To check other (GUI-related) options, access the program help:
```
python main.py --help
```

## Usage Instructions
1. Two display windows will popup immediately after starting the program. The left window will contain the input image while the right window will be initially empty. *You may press the ESC key or the Q key (while focused on the windows) to terminate the program at any time.*
2. On the left window containing the input image, select four (4) points of the quadrilateral region to be transformed by double clicking each desired point. Once four (4) points have been selected, the quadrilateral region to be transformed will be shown by lines. Press the ENTER key to confirm transformation of the highlighted region. 
3. The transformed output image will be displayed on the right window. Press any key to terminate the program.


## Homework 1: Removing Projective Distortion on Images

### Learning Objective
1) Learn how to apply linear mapping concepts on removing projective distortion

### Problem
- When an image is taken by the camera, there is an inherent projective distortion. This makes rectangular objects appear distorted. However, this projective distortion can be reversed if we know the 3x3 matrix that maps from projective to affine (frontal no distortion). 

- Using our knowledge on linear mapping and least squares estimation, develop a program that will remove the projective distortion on a given image. Note that at least 4 pts on the target undistorted image must be known.

### Solution
Let the affine coordinate space of an *m*-by-*n* input image be expressed as $X$ defined below.

$$X = \\{(u, v, 1) | u,v\in\mathbb{Z} \cap 0\leq u\lt n \cap 0\leq v\lt m \\}$$

Let the subspace $W\subseteq X$ be defined by the quadrilateral region formed by connecting four (4) non-collinear points $p_1,...,p_4 \in X$.

Let the affine coordinate space of the *h*-by-*w* output image be expressed as $Y$ defined below.

$$Y = \\{(x, y, 1) | x,y\in\mathbb{Z} \cap 0\leq x\lt w \cap 0\leq y\lt h \\}$$

We want to find a mapping, $A$, that maps the subspace $W$ to the space $Y$.
$$A : W \to  Y $$

To be able to compute the mapping $A$, we assume that the four (4) corner points defining $W$ maps to the corner points of the output image, i.e. $q_1,...q_4 \in Y$; where, $q_1 = [0, 0, 1]$, $q_2=[w-1, 0, 1]$, $q_3=[w-1, h-1, 1]$, and $q_4=[0, h-1, 1]$.

To simplify the solution, we assume $W$ and $Y$ are continuous spaces. The transformation matrix $A$ is then determined by solving the equation below.

$$[p_1, p_2, p_3, p_4]^T A = [q_1, q_2, q_3, q_4]^T$$

The equation above can be solved using linear least squares regression. Suppose $P=[p_1, p_2, p_3, p_4]^T$, and $Q=[q_1, q_2, q_3, q_4]^T$, the value of the transformation matrix $A$ can be computed below.
$$A = (P^T P)^{-1}P^T Q$$

The image transformation can then be performed by copying the pixel values in the input image affine coordinates $[u,v,1] \in X$ to the output image affine coordinates $\lfloor [x,y,1] A \rfloor \in Y$. Note: *Instead of mapping pixel values in* $W$*, we map pixel values in* $X$ *and copy the pixel value as long as its mapped coordinate belongs to the output space* $Y$*.*

## LICENSE
