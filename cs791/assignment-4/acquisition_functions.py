### Add necessary imports ###

# Implement acquisition functions here

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Compute the Expected Improvement acquisition function.
    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value
    
    Returns:
    ei: Expected Improvement values (num_test_samples, 1)
    """
    pass

def probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """
    Compute the Probability of Improvement acquisition function.
    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value
    xi: Exploration-exploitation trade-off parameter

    Returns:
    pi: Probability of Improvement values (num_test_samples, 1)
    """
    pass