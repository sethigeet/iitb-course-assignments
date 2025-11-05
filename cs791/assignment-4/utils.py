import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed):
    """Set random seed for all libraries to ensure reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)


def get_candidates(X_evaluated, hyperparameter_utils, num_candidates):
    """Get candidates from the search space."""

    candidates = []
    evaluated_set = set()
    for vec in X_evaluated:
        evaluated_set.add(tuple(vec))

    attempts = 0
    max_attempts = num_candidates * 2
    while len(candidates) < num_candidates and attempts < max_attempts:
        candidate_vec = hyperparameter_utils["hyperparams_to_vector"](
            hyperparameter_utils["sample_random_hyperparams"](
                hyperparameter_utils["space"]
            )
        )
        candidate_tuple = tuple(candidate_vec)
        if candidate_tuple not in evaluated_set:
            candidates.append(candidate_vec)
            evaluated_set.add(candidate_tuple)
        attempts += 1

    # If we didn't get enough candidates, fill with random samples
    while len(candidates) < num_candidates:
        candidates.append(
            hyperparameter_utils["hyperparams_to_vector"](
                hyperparameter_utils["sample_random_hyperparams"](
                    hyperparameter_utils["space"]
                )
            )
        )

    return np.array(candidates)


def log_marginal_likelihood(
    x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4
):
    """
    Compute the log-marginal likelihood.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)

    Returns:
    log_likelihood: Scalar log-marginal likelihood value
    """
    n = len(x_train)
    K = kernel_func(x_train, x_train, length_scale=length_scale, sigma_f=sigma_f)

    # Add noise to training kernel (for numerical stability)
    # K += noise * np.eye(n)

    K_inv = np.linalg.inv(K)
    log_likelihood = (
        -0.5 * y_train.T @ K_inv @ y_train
        - np.sum(np.log(np.linalg.norm(K_inv, ord=1)))
        - 0.5 * n * np.log(2 * np.pi)
    )

    return log_likelihood


def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """
    Optimize hyperparameters using grid search.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)

    Returns:
    best_length_scale: Optimized length scale
    best_sigma_f: Optimized signal variance
    """
    length_scales = np.logspace(-2, 1, 20)  # 0.01 to 10
    sigma_fs = np.logspace(-1, 1, 20)  # 0.1 to 10

    best_log_likelihood = -np.inf
    best_length_scale = 1.0
    best_sigma_f = 1.0

    for length_scale in length_scales:
        for sigma_f in sigma_fs:
            try:
                log_likelihood = log_marginal_likelihood(
                    x_train, y_train, kernel_func, length_scale, sigma_f, noise
                )
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_length_scale = length_scale
                    best_sigma_f = sigma_f
            except np.linalg.LinAlgError:
                # Skip if inverse fails
                continue

    return best_length_scale, best_sigma_f


def gaussian_process_predict(
    x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4
):
    """
    Perform GP prediction. Return mean and standard deviation of predictions.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)
    x_test: Test inputs (num_test_samples, num_hyperparameters)
    kernel_func: Kernel function to use

    Returns:
    mu_s: Predicted means (num_test_samples, 1)
    sigma_s: Predicted standard deviations (num_test_samples, 1)
    """
    K_XX = kernel_func(x_train, x_train, length_scale=length_scale, sigma_f=sigma_f)
    k_X = kernel_func(x_test, x_train, length_scale=length_scale, sigma_f=sigma_f)
    k = kernel_func(x_test, x_test, length_scale=length_scale, sigma_f=sigma_f)

    # Add noise to training kernel (for numerical stability)
    # K_XX += noise * np.eye(len(x_train))

    K_XX_inv = np.linalg.inv(K_XX)

    # Predictive mean
    mu_s = k_X @ K_XX_inv @ y_train

    # Predictive variance
    middle = k_X @ K_XX_inv @ k_X.T
    var_s = np.diag(k) - np.diag(middle)
    var_s = np.maximum(var_s, 0)  # Ensure non-negative variance
    sigma_s = np.sqrt(var_s)

    # Reshape to column vectors
    mu_s = mu_s.reshape(-1, 1)
    sigma_s = sigma_s.reshape(-1, 1)

    return mu_s, sigma_s


def plot_progression(accuracies_history, save_path, title):
    """Plot validation accuracy progression."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(accuracies_history) + 1),
        accuracies_history,
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
