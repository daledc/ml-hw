# Program Usage
## Program Requirements
The program minimum requirements are: `python>=3.2`, `numpy`, and `opencv-python`.

If you have `python` and `pip` installed you can run the command below:
```
pip install -r requirements.txt
```

The program has been tested to work on Windows 10 and PopOS (Ubuntu-based Distro) using the specific versions:
- `python==3.8.13`
- `numpy==1.23.4`
- `opencv-python==4.6.0.66`


## Running The Program
You can run the program in the command line by specifying an input image path and output image path as below:
```
python main.py -i book.jpg -o output.jpg
```

Note: *Omitting the options will use the default input and output image path provided.*


## Program Usage
1. A window containing the input image will pop-up.
2. Select four (4) corner points of the region to be transformed into a rectangular image by double clicking each desired corner point.
3. Once four (4) corner points have been selected, the quadrilateral region to be transformed will be bounded by lines.
4. Press the ENTER key to confirm transformation of the highlighted region. Otherwise, press the ESC key or the Q key to exit the program.


# Background
## Homework 1: Removing Projective Distortion on Images

### Learning Objective:
1) Learn how to apply linear mapping concepts on removing projective distortion

### Problem:
- When an image is taken by the camera, there is an inherent projective distortion. This makes rectangular objects appear distorted. However, this projective distortion can be reversed if we know the 3x3 matrix that maps from projective to affine (frontal no distortion). 

- Using our knowledge on linear mapping and least squares estimation, develop a program that will remove the projective distortion on a given image. Note that at least 4 pts on the target undistorted image must be known.

# Solution
Let the affine coordinate space of an $m$-by-$n$ input image be expressed as $X$ defined below.

$$X = \{(u, v, 1) | u,v\in\mathbb{Z} \cap 0\leq u\lt n \cap 0\leq v\lt m \}$$

Let the subspace $W\subseteq X$ be defined by the quadrilateral region formed by connecting four non-collinear points $p_1,...,p_4 \in X$.

Let the affine coordinate space of the $h$-by-$w$ output image be expressed as $Y$ defined below.

$$Y = \{(x, y, 1) | x,y\in\mathbb{Z} \cap 0\leq x\lt w \cap 0\leq y\lt h \}$$

We want to find a mapping, $A$, that maps the subspace $W$ to the space $Y$.
$$A : W \to  Y $$

To do this, we assume that the four (4) corner points defining $W$ maps to the corner points of the output image, $q_1,...q_4 \in Y$;

i.e., $q_1 = <0, 0, 1>, q_2=<w-1, 0, 1>, q_3=<w-1, h-1, 1>, and q_4=<0, h-1, 1>$

For simplicity, we assume the input subspace $W$ and output space $Y$ are continuous. The projection matrix $A$ is then determined by solving the equation below.

$$[p_1, p_2, p_3, p_4]^T A = [q_1, q_2, q_3, q_4]^T$$


We then copy the pixel values in the affine image coordinates $(u,v,1) \in W$ to $\lfloor (u,v,1) \cdot A \rfloor \in Y$.