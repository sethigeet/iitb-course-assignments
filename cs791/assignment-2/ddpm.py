import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import DDPM
from scheduler import NoiseSchedulerDDPM
from utils import seed_everything

# Add any extra imports you want here


def _scale_to_model_range(x):
    # From [0,1] to [-1,1]
    return x * 2.0 - 1.0


def _scale_to_image_range(x):
    # From [-1,1] to [0,1]
    return (x + 1.0) / 2.0


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
    # Ho et al. training: predict epsilon with MSE over q(x_t|x_0)
    scheduler = NoiseSchedulerDDPM(
        num_timesteps=num_steps, type=scheduler_type, device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for data, _ in pbar:
            data = data.to(device)
            x0 = _scale_to_model_range(data)

            bsz = x0.shape[0]
            t = torch.randint(
                0, scheduler.num_timesteps, (bsz,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, t, noise)

            optimizer.zero_grad()
            eps_pred = model(x_t, t)
            loss = criterion(eps_pred, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{run_name}/model_epoch_{epoch + 1}.pth")

    torch.save(model.state_dict(), f"{run_name}/model.pth")
    print(f"Training completed. Model saved to {run_name}/model.pth")


def sample(model, device, num_samples=16, num_steps=1000, scheduler_type="cosine"):
    """
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    """
    scheduler = NoiseSchedulerDDPM(
        num_timesteps=num_steps, type=scheduler_type, device=device
    )
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(num_samples, 1, 28, 28, device=device)
        for t in tqdm(reversed(range(num_steps)), desc="Sampling"):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps_pred = model(x_t, t_batch)
            x_t = scheduler.step(x_t, t_batch, eps_pred)

        x0 = _scale_to_image_range(x_t)
        x0 = torch.clamp(x0, 0.0, 1.0)
        return x0


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Model Template")
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
    # Add any other arguments you want here
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

    model = DDPM()
    model.to(device)

    run_name = f"exps_ddpm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps_{args.scheduler_type}scheduler"
    os.makedirs(run_name, exist_ok=True)

    if args.mode == "train":
        start_epoch = 0
        if args.continue_training:
            checkpoint_files = [f for f in os.listdir(run_name) if f.endswith(".pth")]
            if checkpoint_files:
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
        samples = sample(
            model, device, args.num_samples, args.num_steps, args.scheduler_type
        )
        torch.save(
            samples, f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.pt"
        )
