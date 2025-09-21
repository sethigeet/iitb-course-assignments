import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from d3pm import sample as sample_d3pm
from d3pm_cond import sample as sample_d3pm_cond
from models import D3PM, ConditionalD3PM
from utils import compute_fid, seed_everything


def load_test_data(batch_size: int = 64, num_samples: int = 1000) -> torch.Tensor:
    """
    Load test data from MNIST dataset.

    Args:
        batch_size: Batch size for data loading
        num_samples: Number of samples to load

    Returns:
        Tensor of shape (num_samples, 1, 28, 28) with values in [0, 1]
    """
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Load data in batches to get exactly num_samples
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    all_data = []
    for data, _ in test_loader:
        all_data.append(data)
        if len(all_data) * batch_size >= num_samples:
            break

    # Concatenate and take exactly num_samples
    test_data = torch.cat(all_data, dim=0)[:num_samples]
    return test_data


def compute_fid_efficient(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    num_samples_per_batch: int = 64,
    num_batches: int = 5,
) -> float:
    """
    Compute FID score efficiently by sampling multiple batches and averaging.

    Args:
        real_images: Real images tensor
        fake_images: Generated images tensor
        num_samples_per_batch: Number of samples per FID calculation
        num_batches: Number of batches to sample and average

    Returns:
        Average FID score
    """
    fid_scores = []

    print(
        f"Computing FID with {num_batches} batches of {num_samples_per_batch} samples each..."
    )

    for i in tqdm(range(num_batches)):
        # Randomly sample indices
        real_indices = torch.randperm(len(real_images))[:num_samples_per_batch]
        fake_indices = torch.randperm(len(fake_images))[:num_samples_per_batch]

        # Get sampled images
        real_batch = real_images[real_indices]
        fake_batch = fake_images[fake_indices]

        # Compute FID for this batch
        start_time = time.time()
        fid_score = compute_fid(real_batch, fake_batch)
        elapsed_time = time.time() - start_time

        fid_scores.append(fid_score.item())
        print(
            f"Batch {i + 1}/{num_batches}: FID = {fid_score.item():.4f} (took {elapsed_time:.1f}s)"
        )

    avg_fid = float(np.mean(fid_scores))
    std_fid = float(np.std(fid_scores))

    print(f"Average FID: {avg_fid:.4f} ± {std_fid:.4f}")
    return avg_fid


def evaluate_model(
    model_path: str,
    device: torch.device,
    num_samples: int = 1000,
    num_steps: int = 1000,
    num_fid_batches: int = 5,
    num_samples_per_fid: int = 64,
) -> Dict:
    """
    Evaluate a single model and return results.

    Args:
        model_path: Path to the model file
        device: Device to run on
        num_samples: Number of samples to generate
        num_steps: Number of diffusion steps
        num_fid_batches: Number of FID calculation batches
        num_samples_per_fid: Number of samples per FID calculation

    Returns:
        Dictionary containing evaluation results
    """
    print(f"\nEvaluating model: {model_path}")

    # Load model
    if "d3pm_cond" in model_path:
        model = ConditionalD3PM(num_classes=10)
    else:
        model = D3PM()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load test data
    print("Loading test data...")
    real_images = load_test_data(
        batch_size=num_samples_per_fid, num_samples=num_samples
    )

    # Generate samples
    print("Generating samples...")
    if "d3pm_cond" in model_path:
        fake_images = sample_d3pm_cond(model, device, num_samples, num_steps)
    else:
        fake_images = sample_d3pm(model, device, num_samples, num_steps)

    # Compute FID
    print("Computing FID score...")
    fid_score = compute_fid_efficient(
        real_images, fake_images, num_samples_per_fid, num_fid_batches
    )

    results = {
        "model_type": "d3pm_cond" if "d3pm_cond" in model_path else "d3pm",
        "model_path": model_path,
        "parameters": {
            "num_samples": num_samples,
            "num_steps": num_steps,
            "num_fid_batches": num_fid_batches,
            "num_samples_per_fid": num_samples_per_fid,
        },
        "fid_score": fid_score,
    }

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate D3PM models with FID scores")
    parser.add_argument(
        "model_path",
        type=str,
        help="Full path to the model file (e.g., './exps_d3pm/10ep_64bs_0.0001lr_1000steps_cosinescheduler/model.pth')",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate for evaluation",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of diffusion steps for sampling",
    )
    parser.add_argument(
        "--num_fid_batches",
        type=int,
        default=5,
        help="Number of FID calculation batches",
    )
    parser.add_argument(
        "--num_samples_per_fid",
        type=int,
        default=64,
        help="Number of samples per FID calculation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from the file
    print(f"Loading model from file: {args.model_path}")

    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found or invalid!")
        exit(1)

    results = evaluate_model(
        args.model_path,
        device,
        args.num_samples,
        args.num_steps,
        args.num_fid_batches,
        args.num_samples_per_fid,
    )

    # Set default output file if not provided
    experiment_name = os.path.splitext(args.model_path)[0]
    output_file = f"{experiment_name}_results.json"

    # Print and save results
    print("\nResults:")
    print("-" * 50)
    print(f"Model Path: {results['model_path']}")
    print(f"FID Score: {results['fid_score']:.4f}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
