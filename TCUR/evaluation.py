import numpy as np

def eval_full_hyperinterpolation(points, full_tensor, basis_list):
    P = points.shape[0]
    N = len(basis_list)
    approx = np.zeros(P)
    for p in range(P):
        phi_vals = [
            np.array([basis_list[n](points[p,n], i)
                      for i in range(full_tensor.shape[n])])
            for n in range(N)
        ]
        temp = full_tensor
        for n in range(N):
            temp = np.tensordot(temp, phi_vals[n], axes=([0], [0]))
        approx[p] = temp
    return approx

def eval_tucker_approximation(points, S_core, V_final, basis_list):
    P = points.shape[0]
    N = len(basis_list)
    approx = np.zeros(P)
    for p in range(P):
        new_vals = []
        for n in range(N):
            x = points[p, n]
            phi = np.array([basis_list[n](x, i)
                            for i in range(basis_list[n].degree+1)])
            new_vals.append(phi @ V_final[n])
        temp = S_core
        for n in range(N):
            temp = np.tensordot(temp, new_vals[n], axes=([0], [0]))
        approx[p] = temp
    return approx

def eval_tcur_approximation(points, G, factors, basis_list):
    P = points.shape[0]
    N = G.ndim
    approx = np.zeros(P)
    for p in range(P):
        new_vals = []
        for n in range(N):
            x = points[p, n]
            phi = np.array([basis_list[n](x, i)
                            for i in range(basis_list[n].degree+1)])
            new_vals.append(phi @ factors[n])
        temp = G
        for n in range(N):
            temp = np.tensordot(temp, new_vals[n], axes=([0], [0]))
        approx[p] = temp
    return approx
