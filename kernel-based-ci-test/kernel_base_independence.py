import numpy as np
from numpy import eye, sqrt, trace, diag, zeros
from scipy.stats import chi2, gamma
from numpy import diag, exp, sqrt
from kernel_functions import centering, pdinv, truncated_eigen, eigdec, columnwise_normalizes, residual_kernel, \
    rbf_kernel_median
"""
Author: Torin Perkins
Contains a method describing the kernel based conditional independence test described in 
NOTE: This is run in PyCharm with python 3.8
https://arxiv.org/ftp/arxiv/papers/1202/1202.3775.pdf

Sources:
https://arxiv.org/ftp/arxiv/papers/1202/1202.3775.pdf
https://conditional-independence.readthedocs.io/en/latest/_modules/conditional_independence/ci_tests/nonparametric/kci.html#kci_test
https://en.wikipedia.org/wiki/Radial_basis_function_kernel
https://www.degruyter.com/document/doi/10.1515/jci-2018-0017/html?lang=en
https://github.com/sanghack81/SDCIT/tree/master
"""

def kci_test(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, alpha=0.05, lam=1e-3, kern=rbf_kernel_median):
    """
    X :param sample of continuous r.v. X
    Y :param sample of continuous r.v. Y
    Z :param sample of continuous r.v. Z
    alpha :param statistic significance value for p-value
    lam :param small positive regularization variable
    kern :param kernel function to be used rbf is the guassian kernel described in source paper
    :returns a dictionary containing the test statistic, the p-value, the crit value, and the result
    """
    n = len(Y)

    # create centered kernels Kx, Ky, Kz
    Kx = centering(kern(np.hstack([X, Z])))
    Ky = centering(kern(Y))
    Kz = centering(kern(Z))
    print("Kx: ")
    print(Kx)
    print("Ky: ")
    print(Ky)
    print("Kz: ")
    print(Kz)

    # create dependent kernels defined in (11) and (12)
    rz = eye(n) - Kz @ pdinv(Kz + lam * eye(n))
    print("Rz: ")
    print(rz)
    Kxz = rz @ Kx @ rz.T  # (11)
    print("Kxz: ")
    print(Kxz)
    Kyz = rz @ Ky @ rz.T  # (12)
    print("Kyz: ")
    print(Kyz)

    # generate test statistic based on (13)
    test_stat = (Kxz * Kyz).sum()
    print("Test Statistic:")
    print(test_stat)

    # find w based on Proposition 5
    # eigenvalues and eigenvectors
    eig_Kxz, eivx = truncated_eigen(*eigdec(Kxz))
    eig_Kyz, eivy = truncated_eigen(*eigdec(Kyz))

    # generate diagonal matrices of non-negative eigenvalues
    eiv_prodx = eivx @ diag(sqrt(eig_Kxz))
    eiv_prody = eivy @ diag(sqrt(eig_Kyz))

    num_eigx = eiv_prodx.shape[1]
    num_eigy = eiv_prody.shape[1]

    # generate ww via description in Proposition 5
    w_size = num_eigx * num_eigy
    w = zeros((n, w_size))
    for i in range(num_eigx):
        for j in range(num_eigy):
            w[:, i * num_eigy + j] = eiv_prodx[:, i] * eiv_prody[:, j]

    ww = w @ w.T if w_size else w.T @ w
    print("WW: ")
    print(ww)
    # Approximation of null distribution by a gamma distribution
    # Section 3.4

    # calculate mean and variance as described in proposition 6.ii
    mean_approx = trace(ww)
    print("Mean: ")
    print(mean_approx)
    var_approx = 2 * trace(ww ** 2)
    print("Var: ")
    print(var_approx)

    k_approx = mean_approx ** 2 / var_approx
    print("k: ")
    print(k_approx)

    theta_approx = var_approx / mean_approx
    print("theta: ")
    print(theta_approx)

    # calculate the crit val and p_value
    critical_val = gamma.ppf(1 - alpha, k_approx, theta_approx)
    p_value = 1 - gamma.cdf(test_stat, k_approx, theta_approx)

    return dict(statistic=test_stat, critval=critical_val, p_value=p_value, reject=p_value < alpha)


if __name__ == "__main__":
    # if running from command line remove if __name__ == "__main__":
    array_size = 5
    # Arrays should be in shape (array_size, 1)
    # Conditionally independent data
    X = np.random.rand(array_size, 1)
    Z = 2 * X + 2
    Y = 3 * Z + 4



    print("X:")
    print(X)

    print("Y:")
    print(Y)

    print("Z:")
    print(Z)
    # Conditionally dependent data
    #Z = np.random.rand(array_size, 1)
    #X = np.random.uniform(low=0, high=1, size=(array_size, 1))
    #Y = 2 * X + 3 * Z


    res = kci_test(X, Y, Z)
    print(res)
    if res['reject']:
        print("Reject null hypothesis: X and Y are not conditionally independent given Z.")
    else:
        print("Fail to reject null hypothesis: X and Y are conditionally independent given Z.")
