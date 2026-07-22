import numpy as np

def f1(x, y, z):
    r2 = x**2 + y**2 + z**2
    return np.exp(-1.0 / (r2 + 1e-15))

def f2(x, y, z):
    return 1.0 / (np.cosh(3*(x+y+z)))**2

def f3(x, y, z):
    return np.cos(2*np.pi*x)**2 + np.cos(2*np.pi*y)**2 + np.cos(2*np.pi*z)**2

def f3_nd(*args):
    return sum(np.cos(2*np.pi*arg)**2 for arg in args)
