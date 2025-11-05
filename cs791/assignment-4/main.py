import argparse
import json
import os
from functools import partial

import torch
from torchvision import datasets, transforms

from acquisition_functions import expected_improvement, probability_of_improvement
from kernels import matern_kernel, rational_quadratic_kernel, rbf_kernel
from optimize import bayesian_optimization
from space_utils import HYPERPARAMETER_UTILS_CNN, HYPERPARAMETER_UTILS_NN
from train_test import train_and_test_CNN, train_and_test_NN
from utils import plot_progression, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and Test Models with Hyperparameters"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["nn", "cnn"],
        default="nn",
        help="Type of model to use",
    )
    parser.add_argument(
        "--acquisition_function",
        type=str,
        choices=["ei", "pi"],
        default="ei",
        help="Acquisition function to use",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["rbf", "matern", "rational_quadratic"],
        default="rbf",
        help="Kernel function to use",
    )
    parser.add_argument(
        "--max_budget",
        type=int,
        default=25,
        help="Maximum budget for hyperparameter optimization",
    )
    parser.add_argument(
        "--init_points",
        type=int,
        default=10,
        help="Number of initial random points for hyperparameter optimization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    assert args.max_budget >= args.init_points, (
        "max_budget should be greater than init_points"
    )

    transform = transforms.Compose([transforms.ToTensor()])

    train_validation_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_size = int(0.8 * len(train_validation_dataset))
    validation_size = len(train_validation_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_validation_dataset, [train_size, validation_size]
    )
    train_val_datasets = (train_dataset, validation_dataset)

    if args.model_type == "nn":
        black_box_function = partial(train_and_test_NN, datasets=train_val_datasets)
        hyperparameter_utils = HYPERPARAMETER_UTILS_NN
    elif args.model_type == "cnn":
        black_box_function = partial(train_and_test_CNN, datasets=train_val_datasets)
        hyperparameter_utils = HYPERPARAMETER_UTILS_CNN
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    if args.kernel == "rbf":
        kernel_func = rbf_kernel
    elif args.kernel == "matern":
        kernel_func = matern_kernel
    elif args.kernel == "rational_quadratic":
        kernel_func = rational_quadratic_kernel
    else:
        raise ValueError(f"Invalid kernel: {args.kernel}")

    if args.acquisition_function == "ei":
        acquisition_func = expected_improvement
    elif args.acquisition_function == "pi":
        acquisition_func = probability_of_improvement
    else:
        raise ValueError(f"Invalid acquisition function: {args.acquisition_function}")

    results = bayesian_optimization(
        black_box_function,
        kernel_func,
        acquisition_func,
        hyperparameter_utils,
        max_budget=args.max_budget,
        init_points=args.init_points,
    )
    os.makedirs("results", exist_ok=True)
    config_name = f"model_{args.model_type}_kernel_{args.kernel}_acquisition_function_{args.acquisition_function}_max_budget_{args.max_budget}_init_points_{args.init_points}"
    plot_progression(
        results["accuracies_history"],
        f"results/{config_name}_progression.png",
        f"Validation Accuracy Progression for {config_name}",
    )

    with open(f"results/{config_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Best Hyperparameters: {results['best_hyperparams']}")
    print(f"Best Validation Accuracy: {results['best_accuracy']:.4f}")
