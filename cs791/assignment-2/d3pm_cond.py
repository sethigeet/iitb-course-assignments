import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import ConditionalD3PM
from scheduler import MaskSchedulerD3PM
from utils import seed_everything


def discretize_mnist(x, num_classes=10):
    """
    Convert MNIST images to discrete values.
    Args:
        x: Tensor of shape (batch_size, 1, 28, 28) with values in [0, 1]
        num_classes: Number of discrete classes (including absorbing state)
    Returns:
        discrete_x: Tensor with values in [0, num_classes-2] (excluding absorbing state)
    """
    # Scale to [0, num_classes-2] and round to nearest integer
    discrete_x = torch.round(x * (num_classes - 2)).long()
    return discrete_x


def train(
    model,
    train_loader,
    test_loader,
    run_name,
    learning_rate,
    start_epoch,
    epochs,
    num_steps,
    scheduler_type,
    device,
):
    """
    Training loop for ConditionalD3PM model.
    """
    # Initialize scheduler
    scheduler = MaskSchedulerD3PM(
        num_timesteps=num_steps, mask_type=scheduler_type, device=device
    )

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(device)
            targets = targets.to(device)

            # Discretize the data
            discrete_data = discretize_mnist(data, num_classes=10)

            # Sample random timesteps
            timesteps = torch.randint(
                0, scheduler.num_timesteps, (data.shape[0],), device=device
            )

            # Add noise according to mask schedule
            noisy_data = scheduler.add_noise(discrete_data, timesteps, num_classes=10)

            # Forward pass
            optimizer.zero_grad()

            # Get model predictions with class conditioning
            logits = model(noisy_data.float(), timesteps, targets)

            # Compute loss
            # Reshape for cross entropy: (batch_size, num_classes, height, width) -> (batch_size * height * width, num_classes)
            logits_flat = (
                logits.permute(0, 2, 3, 1).contiguous().view(-1, logits.shape[1])
            )
            targets_flat = discrete_data.view(-1)

            loss = criterion(logits_flat, targets_flat)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{run_name}/model_epoch_{epoch + 1}.pth")

    # Save final model
    torch.save(model.state_dict(), f"{run_name}/model.pth")
    print(f"Training completed. Model saved to {run_name}/model.pth")


def sample(model, class_label, device, num_samples=16, num_steps=1000):
    """
    Sample from the ConditionalD3PM model using the reverse diffusion process.

    Args:
        model: Trained ConditionalD3PM model
        class_label: Class label to generate (0-9)
        device: Device to run on
        num_samples: Number of samples to generate
        num_steps: Number of diffusion steps

    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28) with values in [0, 1]
    """
    model.eval()

    # Start with all absorbing states
    x = torch.full(
        (num_samples, 1, 28, 28), 9, device=device, dtype=torch.long
    )  # 9 is absorbing state

    # Create class labels for all samples
    class_labels = torch.full(
        (num_samples,), class_label, device=device, dtype=torch.long
    )

    # Reverse diffusion process
    with torch.no_grad():
        for t in tqdm(reversed(range(num_steps)), desc=f"Sampling class {class_label}"):
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Get model predictions with class conditioning
            logits = model(x.float(), timestep, class_labels)

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)

            # Sample from the predicted distribution
            # For each pixel, sample from the categorical distribution
            x_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, probs.shape[1])
            sampled_flat = torch.multinomial(x_flat, 1).squeeze(-1)
            x = sampled_flat.view(num_samples, 1, 28, 28)

    # Convert from discrete values back to [0, 1] range
    x_continuous = x.float() / 8.0  # Scale from [0, 8] to [0, 1]
    x_continuous = torch.clamp(x_continuous, 0, 1)

    return x_continuous


def parse_args():
    parser = argparse.ArgumentParser(description="D3PM Conditional Model Template")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=1000, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of samples to generate"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Scheduler type: cosine or linear",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "sample"],
        help="Mode: train or sample",
    )
    parser.add_argument(
        "--continue_training",
        type=bool,
        default=False,
        help="Continue training from the last checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ### Data Preprocessing Start ### (Do not edit this)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ### Data Preprocessing End ### (Do not edit this)

    model = ConditionalD3PM(num_classes=10)
    model.to(device)

    run_name = f"exps_conditional_d3pm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps_{args.scheduler_type}scheduler"
    os.makedirs(run_name, exist_ok=True)

    if args.mode == "train":
        start_epoch = 0
        if args.continue_training:
            # Choose the latest checkpoint
            checkpoint_files = [f for f in os.listdir(run_name) if f.endswith(".pth")]
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_checkpoint = checkpoint_files[-1]
            start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
            model.load_state_dict(torch.load(f"{run_name}/{latest_checkpoint}"))
        model.train()
        train(
            model,
            train_loader,
            test_loader,
            run_name,
            args.learning_rate,
            start_epoch,
            args.epochs,
            args.num_steps,
            args.scheduler_type,
            device,
        )
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/model.pth"))
        model.eval()
        for class_num in range(10):
            samples = sample(model, class_num, device, args.num_samples, args.num_steps)
            torch.save(
                samples,
                f"{run_name}/{class_num}class_{args.num_samples}samples_{args.num_steps}steps.pt",
            )
