import os
import scipy.linalg
import scipy.optimize
import scipy.stats
import typing
import warnings
import gpflow
import numpy as np
import numpy.ma as ma
from gpflow.kernels import RBF, White
from gpflow.models import GPR
from numpy import eye, sqrt, trace, diag, zeros
from scipy.stats import chi2, gamma
from typing import Union, List
from sklearn.metrics import euclidean_distances
from numpy import diag, exp, sqrt

"""
Kernel Functions Adapted from https://github.com/sanghack81/SDCIT/tree/master
"""
def columnwise_normalizes(*Xs) -> typing.List[Union[None, np.ndarray]]:
    """normalize per column for multiple data"""
    return [columnwise_normalize(X) for X in Xs]


def columnwise_normalize(X: np.ndarray) -> Union[None, np.ndarray]:
    """normalize per column"""
    if X is None:
        return None
    return (X - np.mean(X, 0)) / np.std(X, 0)  # broadcast
def ensure_symmetric(x: np.ndarray) -> np.ndarray:
    return (x + x.T) / 2

def truncated_eigen(eig_vals, eig_vecs=None, relative_threshold=1e-5):
    """Retain eigenvalues and corresponding eigenvectors where an eigenvalue > max(eigenvalues)*relative_threshold"""
    indices = np.where(eig_vals > max(eig_vals) * relative_threshold)[0]
    if eig_vecs is not None:
        return eig_vals[indices], eig_vecs[:, indices]
    else:
        return eig_vals[indices]


def eigdec(X: np.ndarray, top_N: int = None):
    """Eigendecomposition with top N descending ordered eigenvalues and corresponding eigenvectors"""
    if top_N is None:
        top_N = len(X)

    X = ensure_symmetric(X)
    M = len(X)

    # ascending M-1-N <= <= M-1
    w, v = scipy.linalg.eigh(X, eigvals=(M - 1 - top_N + 1, M - 1))

    # descending
    return w[::-1], v[:, ::-1]

def centering(M: np.ndarray) -> Union[None, np.ndarray]:
    """Matrix Centering"""
    if M is None:
        return None
    n = len(M)
    H = np.eye(n) - 1 / n
    return H @ M @ H

def pdinv(x: np.ndarray) -> np.ndarray:
    """Inverse of a positive definite matrix"""
    U = scipy.linalg.cholesky(x)
    Uinv = scipy.linalg.inv(U)
    return Uinv @ Uinv.T

def residual_kernel(K_Y: np.ndarray, K_X: np.ndarray, use_expectation=True, with_gp=True, sigma_squared=1e-3, return_learned_K_X=False):
    """Kernel matrix of residual of Y given X based on their kernel matrices, Y=f(X)"""
    import gpflow
    from gpflow.kernels import White, Linear
    from gpflow.models import GPR

    K_Y, K_X = centering(K_Y), centering(K_X)
    T = len(K_Y)

    if with_gp:
        eig_Ky, eiy = truncated_eigen(*eigdec(K_Y, min(100, T // 4)))
        eig_Kx, eix = truncated_eigen(*eigdec(K_X, min(100, T // 4)))

        X = eix @ diag(sqrt(eig_Kx))  # X @ X.T is close to K_X
        Y = eiy @ diag(sqrt(eig_Ky))
        n_feats = X.shape[1]

        linear = Linear(n_feats, ARD=True)
        white = White(n_feats)
        gp_model = GPR(X, Y, linear + white)
        gpflow.train.ScipyOptimizer().minimize(gp_model)

        K_X = linear.compute_K_symm(X)
        sigma_squared = white.variance.value

    P = pdinv(np.eye(T) + K_X / sigma_squared)  # == I-K @ inv(K+Sigma) in Zhang et al. 2011
    if use_expectation:  # Flaxman et al. 2016 Gaussian Processes for Independence Tests with Non-iid Data in Causal Inference.
        RK = (K_X + P @ K_Y) @ P
    else:  # Zhang et al. 2011. Kernel-based Conditional Independence Test and Application in Causal Discovery.
        RK = P @ K_Y @ P

    if return_learned_K_X:
        return RK, K_X
    else:
        return RK


def rbf_kernel_median(data: np.ndarray, *args, without_two=False):
    """A list of RBF kernel matrices for data sets in arguments based on median heuristic"""
    if args is None:
        args = []

    outs = []
    for x in [data, *args]:
        D_squared = euclidean_distances(x, squared=True)
        # masking upper triangle and the diagonal.
        mask = np.triu(np.ones(D_squared.shape), 0)
        median_squared_distance = ma.median(ma.array(D_squared, mask=mask))
        if without_two:
            kx = exp(-D_squared / median_squared_distance)
        else:
            kx = exp(-0.5 * D_squared / median_squared_distance)
        outs.append(kx)

    if len(outs) == 1:
        return outs[0]
    else:
        return outs