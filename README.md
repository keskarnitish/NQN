# A Limited-Memory Quasi-Newton Algorithm for Bound-Constrained Nonsmooth Optimization
by [Nitish Shirish Keskar](http://users.iems.northwestern.edu/~nitish) and [Andreas Waechter](http://users.iems.northwestern.edu/~andreasw/)

Paper link: [arXiv preprint](https://arxiv.org/abs/1612.07350)

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Disclaimer and Known Issues](#disclaimer-and-known-issues)
0. [Usage](#usage)

### Introduction

The proposed method (NQN) is a limited-memory quasi-Newton method for bound-constrained nonsmooth optimization. It is an active-set method in that it operates iteratively in a two-phase approach of predicting the optimal active-set and computing steps in the identified subspace. The code is written in pure-Python and aims to mimic the calling syntax of [`scipy.optimize.fmin_l_bfgs_b`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html). In order to replace an existing call to `fmin_l_bfgs_b` with our code, simply import `NQN.py` and replace `scipy.optimize.fmin_l_bfgs_b` to `NQN.fmin_l_bfgs_b`. In most cases, no changes to the calling arguments should be necessary. We include additional usage details at the end of this README.

Please contact us if you have any questions, suggestions, requests or bug-reports.

### Citation

If you use this code or our results in your research, please cite:

	@article{Keskar2016,
		author = {Nitish Shirish Keskar and Andreas Waechter},
		title = {A Limited-Memory Quasi-Newton Algorithm for Bound-Constrained Nonsmooth Optimization},
		journal = {arXiv:1612.07350},
		year = {2016}
	}

### Disclaimer and Known Issues

0. Our code is written in pure-Python. Circumstantially, this can be slower than SciPy's L-BFGS-B implementation which is written in Fortran. 
1. If you set the value of the gradient-sampling memory (M) to be greater than 1, you need the [CVXOPT](https://github.com/cvxopt/cvxopt) package for solving the QP. This can be easily installed via `pip`. However, we do not recommend setting M to be greater than 1 unless you need extremely high-precision solutions.
2. Please write to us if you need the Python code for the test problems, other solvers/wrappers or scripts for generating figures. 

	
### Usage

In order to use our code, simply import the `NQN.py` file and call `NQN.fmin_l_bfgs_b` with the calling options similar to [`scipy.optimize.fmin_l_bfgs_b`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html). Concretely, the function signature is

``fmin_l_bfgs_b(funObj, x0, gradObj, bounds=None, m=20, M=1, pgtol=1e-5, iprint=-1, maxfun=15000, maxiter=15000, callback=None, factr=0.)``

where:

```
funObj   => Function to minimize 
x0       => Initial/starting point
gradObj  => Gradient of objective function. Unlike L-BFGS-B, for NQN, this *must* be provided 
bounds   => A tuple of bound constraints. Use (-numpy.inf, numpy.inf) if a a variable in unconstrained
m        => L-BFGS memory
M        => Gradient sampling memory
pgtol    => Scaled 2-norm tolerance for convergence
iprint   => Controls output; iprint>0 prints to STDOUT
maxfun   => Max. number of function evaluations
maxiter  => Max. number of iterations
callback => Any function, called before every iteration
factr    => The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * 1E-16
```

The only additional option we employ over SciPy's L-BFGS-B implementation is the size of the gradient-sampling memory (M) which is initialized to 1. Further, the user can choose to employ a BFGS variant of our code (i.e., retaining BFGS matrices instead of limited-memory curvature pairs) by setting `m=numpy.inf`.

A sample usage of our code (on the popular Rosenbrock function) is:
```
import scipy.optimize
import NQN
import numpy

n = 100
funObj = scipy.optimize.rosen
gradObj = func = scipy.optimize.rosen_der
bounds = [(-0.5,0.5) for i in range(n)]
x0 = numpy.zeros(n)

scipy_output = scipy.optimize.fmin_l_bfgs_b(func,x0,fprime=grad,bounds=bounds)
NQN_output = NQN.fmin_l_bfgs_b(func,x0,fprime=grad,bounds=bounds)
```

Similar to the SciPy L-BFGS-B implementation, we return the final iterate, corresponding function value and an _information dictionary_. We use different termination flags from SciPy's L-BFGS-B implementation. They are
```
Termination Flag:
0 => Converged
1 => Reached Maximum Iterations
2 => Reached Maximum Function Evaluations
3 => Converged to Nonstationary Point
4 => Abnormal Termination in Line Search
5 => Iterate Has NaN values
6 => Numerical Issues
7 => FACTR Convergence
 ```
