import numpy as np


def compute_squared_distance(x1, x2):
    """Compute the squared distance between two sets of inputs.

    Args:
        x1: First set of inputs (n1, d)
        x2: Second set of inputs (n2, d)

    Returns:
        Squared distance matrix of shape (n1, n2)
    """

    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2)


def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF (Gaussian) kernel.

    K(x1, x2) = sigma_f^2 * exp(-0.5 * ||x1 - x2||^2 / length_scale^2)

    Args:
        x1: First set of inputs (n1, d)
        x2: Second set of inputs (n2, d)
        length_scale: Length scale parameter (controls smoothness)
        sigma_f: Signal variance parameter

    Returns:
        Covariance matrix of shape (n1, n2)
    """

    dist_sq = compute_squared_distance(x1, x2)
    K = sigma_f**2 * np.exp(-0.5 * dist_sq / length_scale**2)

    return K


def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matern kernel.

    For nu=1.5, the Matern kernel has the form:
    K(x1, x2) = sigma_f^2 * (1 + sqrt(3) * r / l) * exp(-sqrt(3) * r / l)
    where r = ||x1 - x2|| and l = length_scale

    Args:
        x1: First set of inputs (n1, d)
        x2: Second set of inputs (n2, d)
        length_scale: Length scale parameter (controls smoothness)
        sigma_f: Signal variance parameter
        nu: Smoothness parameter (default 1.5)

    Returns:
        Covariance matrix of shape (n1, n2)
    """
    dist_sq = compute_squared_distance(x1, x2)
    dist = np.sqrt(dist_sq)

    assert nu == 1.5, "Only nu = 1.5 is supported for Matern kernel"

    r = dist / length_scale
    K = sigma_f**2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)

    return K


def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel.

    K(x1, x2) = sigma_f^2 * (1 + ||x1 - x2||^2 / (2*alpha*length_scale^2))^(-alpha)

    Args:
        x1: First set of inputs (n1, d)
        x2: Second set of inputs (n2, d)
        length_scale: Length scale parameter (controls smoothness)
        sigma_f: Signal variance parameter
        alpha: Shape parameter (controls differentiability)

    Returns:
        Covariance matrix of shape (n1, n2)
    """

    dist_sq = compute_squared_distance(x1, x2)
    K = sigma_f**2 * (1 + dist_sq / (2 * alpha * length_scale**2)) ** (-alpha)

    return K
