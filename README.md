# Performance Estimation Problem for SGD

1. [Introduction](#Introduction)
2. [Getting Started](#Getting-Started)
3. [Functionality](#Functionality)
4. [Example](#Example)
5. [Customization](#Customization)
6. [Contributions](#Contributions)
7. [References](#References)

## Introduction

This code aims to implement a performance estimation problem for Stochastic Gradient Descent (SGD) with or without
bounded variance assumption. The main results, being without bounded variance assumption, are summarized in [[1]](#1).
The problem model and methodology is inspired by [[2]](#2).

## Getting Started

The code is written in Python 3.13. To install the required packages with pip, run the following command:

```bash
python3 -m pip install -r requirements.txt
```

Note that you will need to obtain a license for Mosek in the first place.

The recommended practice is to create a virtual environment dedicated to the project, which can be done with

```bash
python3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Instead of pip one can use the faster alternative [uv](https://github.com/astral-sh/uv).

```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Functionality

The main functionality of the code is to estimate the worst instance of SGD for a given problem.

```python
sgd = get_worst_instance(gamma=0.5, T=2, mu=0, L=1)
```

#### Parameters

- `gamma` (_float_): The step-size of the SGD algorithm.
- `T` (_int_): The number of iterations.
- `mu` (_float_): The strong convexity parameter. If set to 0, the problem is not strongly convex.
- `L` (_float_): The smoothness parameter.
- `objective` (_str_): The objective function. Possible values are:
    - "bias": minimize the rho term (in front of the sum) in the Lyapunov function (for non-strongly convex problems)
    - "variance": minimizes the sum of (e_k), by fixing d to be its maximal value (up to a factor 1-epsilon)
      Defaults to "bias".
- `additional_constraints` (_function_): (Optional) Additional conditions to add to the SGD class. Should be a function
  that takes in the SGD class object and modifies it in place.
- `solver` (_str_): The solver to use. Possible values are:
    - "MOSEK": Use the MOSEK solver [[3]](#3) (default).
    - "CLARABEL": Use the Clarabel solver [[4]](#4). This solver has purposefully been degraded with a much lower
      tolerance to reach different solutions. It is not recommended to use this solver for real problems.

#### Returns

- `sgd` (_SGD_): The instance of SGD after solving. This instance allows to access all the dual variables through:
    - `sgd.value`: The objective value of the PEP.
    - `sgd.rho`: The value of the rho term in the Lyapunov function.
    - `sgd.vars[k]`: An array containing the values of the Lyapunov variables at iteration k. Come as a tuple (a_k,
      e_k).
    - `sgd.lambs[k]`: A matrix containing the values of the dual variables lambda at iteration k.
    - `sgd.lamb1s[k]`: An array containing the values of the dual variables lambda_1 at iteration k.
    - `sgd.lamb2s[k]`: An array containing the values of the dual variables lambda_2 at iteration k.
    - `sgd.tau[k]`: The values of the dual variable tau at iteration k.

## Example

```python
# Imports
import sys

sys.path.append("../")
from src import get_worst_instance
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
GAMMAS = np.linspace(0.1, 3, 100)
Ts = [2, 5, 10]

# Compute all rates
rates = [[get_worst_instance(gamma, N, mu=0, L=1).value for gamma in tqdm(GAMMAS)] for N in Ns]

# Plot obtained rates
fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

for i, (T, ax) in enumerate(zip(Ts, axs.flatten())):
    ax.plot(GAMMAS, rates[i])
    ax.set_title(f"T={T} Iteration{'s' if T > 1 else ''}")
    ax.set_xlabel(r"Normalized Step-Size ($\gamma L$)")
    if i == 0: ax.set_ylabel(r"Optimal Bias ($\rho_\text{opt}$)")

plt.tight_layout()
plt.show()
```

The code above computes the worst rate of convergence of SGD for different step-sizes and number of iterations. The
result is shown below:

![](https://github.com/DanielCortild/PEP-for-SGD-without-Variance/blob/main/example.png?raw=true)

## Customization

The code is designed to be easily customizable. Additional variance assumptions or objectives to the PEP may be
integrated by modifying the `src/SGD/SGD.py` file. Some experiments are provided in `tests/` to illustrate the
functionality, and may be used as a template for further experiments.

## Contributions

Contributions are welcome. Please open an issue to discuss the changes you would like to make.

## References

<a id="1">[1]</a> Yet unpublished.
<br/>
<a id="2">[2]</a> Taylor, A., & Bach, F. (2019). Stochastic first-order methods: Non-asymptotic and computer-aided
analyses via potential functions. Proceedings of the 32nd Conference on Learning Theory,
2934–2992. https://proceedings.mlr.press/v99/taylor19a.html
<br/>
<a id="3">[3]</a> MOSEK, A. (2025). MOSEK Optimizer API for Python. Release 11.0.20. http://www.mosek.com
<br/>
<a id="4">[4]</a> Goulart, P. J., & Chen, Y. (2024). Clarabel: An interior-point solver for conic programs with
quadratic objectives (No. arXiv:2405.12762). arXiv. https://doi.org/10.48550/arXiv.2405.12762

