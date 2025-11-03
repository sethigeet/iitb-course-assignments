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


HYPERPARAMETER_SPACE = {
    "hidden_size": [100, 200, 300, 400, 500],  # Discrete
    "epochs": np.linspace(1, 10, 10, dtype=int).tolist(),  # Linear: [1, 2, ..., 10]
    "learning_rate": np.logspace(-5, -1, 20).tolist(),  # Log: 10^-5 to 10^-1
    "batch_size": [16, 32, 64, 128, 256],  # Discrete
    "dropout_rate": np.linspace(0, 0.5, 20).tolist(),  # Linear: [0, 0.026, ..., 0.5]
    "weight_decay": np.logspace(-6, -2, 20).tolist(),  # Log: 10^-6 to 10^-2
}


def sample_random_hyperparams(space):
    """Sample random hyperparameters from the space."""
    return {
        "hidden_size": int(np.random.choice(space["hidden_size"])),
        "epochs": int(np.random.choice(space["epochs"])),
        "learning_rate": float(np.random.choice(space["learning_rate"])),
        "batch_size": int(np.random.choice(space["batch_size"])),
        "dropout_rate": float(np.random.choice(space["dropout_rate"])),
        "weight_decay": float(np.random.choice(space["weight_decay"])),
    }


def hyperparams_to_vector(hyperparams):
    """Convert hyperparameters to a normalized vector for GP."""
    # Normalize hyperparameters to [0, 1] range
    # Hidden size: map [100, 500] to [0, 1]
    # Epochs: map [1, 10] to [0, 1]
    # Learning rate: log scale already, map log10([1e-5, 1e-1]) = [-5, -1] to [0, 1]
    # Batch size: map [16, 256] to [0, 1] (log scale)
    # Dropout: map [0, 0.5] to [0, 1]
    # Weight decay: log scale already, map log10([1e-6, 1e-2]) = [-6, -2] to [0, 1]

    vec = np.array(
        [
            (hyperparams["hidden_size"] - 100) / 400.0,  # [0, 1]
            (hyperparams["epochs"] - 1) / 9.0,  # [0, 1]
            (np.log10(hyperparams["learning_rate"]) + 5) / 4.0,  # [0, 1]
            (np.log2(hyperparams["batch_size"]) - 4) / 4.0,  # [0, 1] (log2)
            hyperparams["dropout_rate"] / 0.5,  # [0, 1]
            (np.log10(hyperparams["weight_decay"]) + 6) / 4.0,  # [0, 1]
        ]
    )
    return vec


def vector_to_hyperparams(vec, space):
    """Convert normalized vector back to hyperparameters (snap to nearest discrete value)."""
    # Denormalize
    hidden_size_val = vec[0] * 400 + 100
    epochs_val = vec[1] * 9 + 1
    lr_val = 10 ** (vec[2] * 4 - 5)
    batch_size_val = 2 ** (vec[3] * 4 + 4)
    dropout_val = vec[4] * 0.5
    weight_decay_val = 10 ** (vec[5] * 4 - 6)

    # Snap to nearest discrete values
    hidden_size = min(space["hidden_size"], key=lambda x: abs(x - hidden_size_val))
    epochs = int(min(space["epochs"], key=lambda x: abs(x - epochs_val)))
    learning_rate = min(space["learning_rate"], key=lambda x: abs(x - lr_val))
    batch_size = min(space["batch_size"], key=lambda x: abs(x - batch_size_val))
    dropout_rate = min(space["dropout_rate"], key=lambda x: abs(x - dropout_val))
    weight_decay = min(space["weight_decay"], key=lambda x: abs(x - weight_decay_val))

    return {
        "hidden_size": int(hidden_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "dropout_rate": float(dropout_rate),
        "weight_decay": float(weight_decay),
    }


def get_candidates(X_evaluated, space, num_candidates):
    """Get candidates from the search space."""

    candidates = []
    evaluated_set = set()
    for vec in X_evaluated:
        evaluated_set.add(tuple(vec))

    attempts = 0
    max_attempts = num_candidates * 2
    while len(candidates) < num_candidates and attempts < max_attempts:
        candidate_vec = hyperparams_to_vector(sample_random_hyperparams(space))
        candidate_tuple = tuple(candidate_vec)
        if candidate_tuple not in evaluated_set:
            candidates.append(candidate_vec)
            evaluated_set.add(candidate_tuple)
        attempts += 1

    # If we didn't get enough candidates, fill with random samples
    while len(candidates) < num_candidates:
        candidates.append(hyperparams_to_vector(sample_random_hyperparams(space)))

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
