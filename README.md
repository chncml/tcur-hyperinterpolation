# TCUR – Tensor Cross‑Approximation for Hyperinterpolation

This repository contains a Python implementation of the **TCUR** (Tensor Cross‑Approximation) algorithms for low‑rank approximation of hyperinterpolation operators, as described in the paper

> *“Greedy Tensor Cross‑Approximation for Hyperinterpolation”*  
> (Author names and reference details)

The code reproduces all numerical experiments from the paper, including:

- Greedy index selection algorithms (Algorithms 5.1, 5.2, 5.4)
- TCUR‑to‑Tucker recompression (Algorithm A.1)
- Error bounds (Theorem 4.4 and Remark 4.9)
- Rank scaling with polynomial degree and tolerance
- Performance comparisons on three test functions

## Features

- **Orthonormal Legendre basis** on `[-1,1]^d`
- **Gauss‑Legendre quadrature** for discrete inner products
- **Full coefficient tensor** computation (for small degrees)
- **TCUR factor‑based and core‑based greedy algorithms**
- **Fiber‑type TCUR** (Algorithm 5.4)
- **TCUR → Tucker recompression** via QR + ST‑HOSVD
- **Evaluation** of hyperinterpolation and Tucker/TCUR approximations at arbitrary points
- **Theoretical error bound** computation (Theorem 4.4)
- **ε‑Tucker rank search** and theoretical rank bounds (Theorem 4.1)
- **Plotting scripts** for all figures in the paper

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/chncml/tcur-hyperinterpolation
cd tcur-hyperinterpolation
pip install -r requirements.txt

## Structure

tcur-hyperinterpolation/
├── tcur/                          # Core library
│   ├── __init__.py                # Exports all public functions
│   ├── basis.py                   # LegendreBasis class
│   ├── quadrature.py              # Gauss‑Legendre quadrature
│   ├── tensor_ops.py              # Unfolding, Tucker reconstruction, ST‑HOSVD
│   ├── coefficient.py             # Inner products, coefficient tensor, subtensor extraction
│   ├── algorithms.py              # Algorithms 5.1, 5.2, 5.4 and TCUR→Tucker
│   ├── evaluation.py              # Evaluation of approximations
│   ├── error_bounds.py            # Theoretical error bounds and rank search
│   ├── utils.py                   # Helper functions
│   └── test_functions.py          # f1, f2, f3 (and f3_nd)
├── experiments/                   # All validation and plotting scripts
│   ├── cross_function_performance.py
│   ├── cross_function_plot.py
│   ├── greedy_performance.py
│   ├── greedy_convergence_plot.py
│   ├── recompression_validation.py
│   ├── error_bounds_validation.py
│   ├── tucker_rank_validation.py
│   ├── rank_scaling_plot.py
│   ├── rank_eps_plot.py
│   └── low_rank_comparison.py     # (uses TensorLy)
├── README.md
└── requirements.txt
