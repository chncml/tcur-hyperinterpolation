import numpy as np
from numpy.linalg import svd

def mode_n_unfold(tensor, n):
    """Mode-n unfolding of a tensor."""
    shape = tensor.shape
    perm = [n] + list(range(n)) + list(range(n+1, tensor.ndim))
    return np.transpose(tensor, perm).reshape(shape[n], -1)

def reconstruct_tucker(core, factors):
    """Reconstruct full tensor from Tucker core and factors."""
    rec = core.copy()
    for n, factor in enumerate(factors):
        rec = np.tensordot(rec, factor, axes=([n], [1]))
        axes_order = list(range(rec.ndim))
        axes_order.insert(n, axes_order.pop())
        rec = np.transpose(rec, axes_order)
    return rec

def st_hosvd(A, ranks):
    """
    Sequential Truncated HOSVD.
    Returns (core, factors) with orthonormal factors.
    """
    core = A.copy()
    factors = []
    for n in range(A.ndim):
        unfold = mode_n_unfold(core, n)
        U, _, _ = svd(unfold, full_matrices=False)
        r = min(ranks[n], core.shape[n])
        U_trunc = U[:, :r]
        factors.append(U_trunc)
        core = np.tensordot(core, U_trunc.T, axes=([n], [1]))
        axes_order = list(range(core.ndim))
        axes_order.insert(n, axes_order.pop())
        core = np.transpose(core, axes_order)
    return core, factors

def column_indices_for_mode(shape, I_sets, mode):
    """
    Linear indices for columns of mode-n unfolding corresponding to
    the Cartesian product of selected index sets in other modes.
    """
    from itertools import product
    other_modes = [m for m in range(len(shape)) if m != mode]
    other_sets = [I_sets[m] for m in other_modes]
    if not other_sets:
        return [0]
    dims = [shape[m] for m in other_modes]
    indices = []
    for comb in product(*other_sets):
        linear = 0
        for val, dim in zip(comb, dims):
            linear = linear * dim + val
        indices.append(linear)
    return indices
