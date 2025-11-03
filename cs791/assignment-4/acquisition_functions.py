import numpy as np
from scipy.special import erf

ZETA = 0.01


def _standard_normal_cdf(z):
    """
    Compute the cumulative distribution function (CDF) of standard normal distribution.
    Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    """
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def _standard_normal_pdf(z):
    """
    Compute the probability density function (PDF) of standard normal distribution.
    ϕ(z) = (1 / sqrt(2π)) * exp(-z²/2)
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)


def probability_of_improvement(mu, sigma, f_best):
    """
    Compute the Probability of Improvement acquisition function.
    PI(x) = Φ((mu(x) - f(x_best) - zeta) / sigma(x))
    where Φ is the CDF of the standard normal distribution

    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value

    Returns:
    pi: Probability of Improvement values (num_test_samples, 1)
    """

    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)

    z = (mu - f_best - ZETA) / sigma
    pi = _standard_normal_cdf(z)
    return pi


def expected_improvement(mu, sigma, f_best):
    """
    Compute the Expected Improvement acquisition function.
    EI(x) = (mu(x) - f(x_best) - zeta)Φ(z) + sigma(x)ϕ(z)
    where z = (mu(x) - f(x_best) - zeta) / sigma(x)

    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value

    Returns:
    ei: Expected Improvement values (num_test_samples, 1)
    """

    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)

    z = (mu - f_best - ZETA) / sigma

    phi_z = _standard_normal_cdf(z)
    p_z = _standard_normal_pdf(z)

    improvement = mu - f_best - ZETA
    ei = improvement * phi_z + sigma * p_z

    # Ensure non-negative EI
    ei = np.maximum(ei, 0)
    return ei
