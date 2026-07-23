# TCUR – Tensor Cross Approximation and Tucker Recompression

This repository provides a Python implementation of the **TCUR (Tensor Cross Approximation with Uniform Ranks)** algorithms introduced in the paper *"Intrinsic Low-Tucker-Rank Theory and Unified Tensor CUR Decomposition for High-Dimensional Hyperinterpolation"* ([reference details to be added](https://arxiv.org/pdf/2607.19741)). The package includes:

- Three greedy algorithms for constructing a **TCUR decomposition**:
  - **Algorithm 5.1** (factor‑based)
  - **Algorithm 5.2** (core‑based)
  - **Algorithm 5.4** (fiber‑based)
- **Recompression** of a TCUR core into a Tucker format (Algorithm A.1).
- **Error bounds** (Theorem 4.4 and Remark 4.9) and **ε‑Tucker rank** estimation.
- Validation scripts that reproduce numerical experiments from the paper, including error tables, convergence plots, and slice comparisons.

The code is built on top of **orthonormal Legendre polynomials** and **Gauss‑Legendre quadrature**, but the algorithms are general and can be adapted to other bases.

---

## Features

- **Full coefficient tensor** computation via quadrature.
- **Sub‑tensor extraction**, mode‑n unfolding, and Tucker operations.
- **Greedy index set expansion** with stopping criteria based on smallest singular values.
- **TCUR‑to‑Tucker** recompression with error bounds.
- **End‑to‑end validation** for test functions \( f_1, f_2, f_3 \) (sharply peaked, smooth, oscillatory).
- **Visualisation** of approximation slices, rank scaling, and error convergence.

---

## Requirements

- Python 3.7+
- NumPy
- SciPy (for Legendre polynomials; falls back to `numpy.polynomial` if unavailable)
- Matplotlib (for plotting scripts)

All dependencies are listed in `requirements.txt`.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/tcur-tensor.git
cd tcur-tensor
pip install -r requirements.txt

.
├── TCUR/                         # Core package
│   ├── __init__.py
│   ├── algorithms.py             # Algorithms 5.1, 5.2, 5.4 and tcur_to_tucker
│   ├── basis.py                  # LegendreBasis class
│   ├── coefficient.py            # Coefficient tensor computation
│   ├── error_bounds.py           # Error bounds and ε‑rank search
│   ├── evaluation.py             # Evaluation of approximations on points
│   ├── quadrature.py             # Gauss‑Legendre quadrature
│   ├── tensor_ops.py             # Unfolding, reconstruction, ST‑HOSVD
│   └── utils.py                  # Helper functions
├── Cross_Function_Performance.py            # End‑to‑end error chain (Table 6.10 style)
├── Cross_Function_Performance_plot.py       # Slice plots for f2
├── Greedy_TCUR_Algorithm_Performance.py     # Compares algorithms with different b, τ
├── Greedy_TCUR_Algorithm_Performance_plot.py # Convergence plot of tol
├── TCUR_to_Tucker_Recompression.py          # Validates Theorem A.1 with plots
├── Unified_TCUR_Error_Bounds.py             # Validates Theorem 4.4 and Remark 4.9
├── validation_tucker_rank.py                # Rank scaling vs degree, ε, dimension
├── validation_tucker_rank_plot1.py          # Rank scaling plot for f2
├── validation_tucker_rank_plot2.py          # Rank vs ε for multiple functions
└── requirements.txt
