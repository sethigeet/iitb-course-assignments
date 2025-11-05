import numpy as np
from tqdm import tqdm

from utils import (
    gaussian_process_predict,
    get_candidates,
    optimize_hyperparameters,
)

MAX_CANDIDATES = 1000


def bayesian_optimization(
    black_box_function,
    kernel_func,
    acquisition_func,
    hyperparameter_utils,
    max_budget=25,
    init_points=10,
):
    """Run Bayesian Optimization loop."""

    space = hyperparameter_utils["space"]
    num_candidates = min(MAX_CANDIDATES, hyperparameter_utils["num_candidates"])

    # Store all evaluated hyperparameters and their accuracies
    X_evaluated = []  # Normalized hyperparameter vectors
    y_evaluated = []  # Validation accuracies
    hyperparams_history = []  # Actual hyperparameter dicts
    accuracies_history = []  # Validation accuracies over iterations

    best_accuracy = -np.inf
    best_hyperparams = None

    print(f"Starting Bayesian Optimization with {init_points} warm-up points...")

    # Warm-up phase: Random sampling
    pbar = tqdm(range(init_points), desc="Warm-up phase")
    for i in pbar:
        hyperparams = hyperparameter_utils["sample_random_hyperparams"](space)
        hyperparams_history.append(hyperparams)

        # Evaluate
        accuracy = black_box_function(hyperparams=hyperparams)
        accuracies_history.append(accuracy)

        # Update best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparams = hyperparams.copy()

        X_evaluated.append(hyperparameter_utils["hyperparams_to_vector"](hyperparams))
        y_evaluated.append(accuracy)
        pbar.set_postfix(
            accuracy=f"Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f})"
        )

    # Bayesian Optimization phase
    print(f"\nStarting BO phase with {max_budget - init_points} iterations...")
    pbar = tqdm(range(max_budget - init_points), desc="BO phase")
    for i in pbar:
        # Convert evaluated history to numpy arrays for GP
        X_train = np.array(X_evaluated)
        y_train = np.array(y_evaluated).reshape(-1, 1)

        # Use fixed kernel hyperparameters (length_scale, sigma_f)
        length_scale, sigma_f = optimize_hyperparameters(X_train, y_train, kernel_func)

        # NOTE: Ideally, we want to get the argmax over the acquisition function
        # for all possible candidates, but this is computationally infeasible.
        # Instead, we sample a subset of the candidates and use the GP to
        # predict the acquisition function values for the subset.
        X_candidates = get_candidates(X_evaluated, hyperparameter_utils, num_candidates)

        # Predict the means and standard deviations for the candidates
        mu_s, sigma_s = gaussian_process_predict(
            X_train, y_train, X_candidates, kernel_func, length_scale, sigma_f
        )

        # Compute acquisition function values
        f_best = best_accuracy
        acquisition_values = acquisition_func(mu_s, sigma_s, f_best).flatten()

        # Select candidate with highest acquisition value
        best_candidate_idx = np.argmax(acquisition_values)
        next_hyperparams = hyperparameter_utils["vector_to_hyperparams"](
            X_candidates[best_candidate_idx], hyperparameter_utils["space"]
        )

        hyperparams_history.append(next_hyperparams)

        # Evaluate
        accuracy = black_box_function(hyperparams=next_hyperparams)
        accuracies_history.append(accuracy)

        # Update best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparams = next_hyperparams.copy()

        X_evaluated.append(
            hyperparameter_utils["hyperparams_to_vector"](next_hyperparams)
        )
        y_evaluated.append(accuracy)

        pbar.set_postfix(
            accuracy=f"Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f})"
        )

    return {
        "best_hyperparams": best_hyperparams,
        "best_accuracy": best_accuracy,
        "hyperparams_history": hyperparams_history,
        "accuracies_history": accuracies_history,
    }
