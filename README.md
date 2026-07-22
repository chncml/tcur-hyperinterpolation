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
git clone https://github.com/yourusername/tcur-hyperinterpolation.git
cd tcur-hyperinterpolation
pip install -r requirements.txt
