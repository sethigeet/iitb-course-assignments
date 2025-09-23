import argparse
import re
from pathlib import Path

import torch
from tqdm import tqdm

from d3pm import sample as sample_d3pm
from d3pm_cond import sample as sample_d3pm_cond
from ddpm import sample as sample_ddpm
from ddpm_cond import sample as sample_ddpm_cond
from models import D3PM, DDPM, ConditionalD3PM, ConditionalDDPM
from utils import seed_everything


def parse_model_filename(filename):
    name = filename.replace(".pth", "")

    # Extract model type
    if "ddpm_cond" in name:
        model_type = "ddpm_cond"
    elif "d3pm_cond" in name:
        model_type = "d3pm_cond"
    elif "ddpm" in name:
        model_type = "ddpm"
    elif "d3pm" in name:
        model_type = "d3pm"
    else:
        raise ValueError(f"Unknown model type in filename: {filename}")

    # Extract parameters using regex
    patterns = {
        "steps": r"(\d+)steps",
        "scheduler": r"([a-z]+)scheduler",
    }

    params = {"model_type": model_type}
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        assert match is not None
        params[key] = match.group(1)

    return params


def load_model(model_path, model_type, device):
    if model_type == "ddpm":
        model = DDPM()
    elif model_type == "d3pm":
        model = D3PM()
    elif model_type == "ddpm_cond":
        model = ConditionalDDPM(num_classes=10)
    elif model_type == "d3pm_cond":
        model = ConditionalD3PM(num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def generate_samples_unconditional(
    model, model_type, device, num_samples=64, num_steps=1000, scheduler_type="cosine"
):
    if model_type == "ddpm":
        return sample_ddpm(model, device, num_samples, num_steps, scheduler_type)
    elif model_type == "d3pm":
        return sample_d3pm(model, device, num_samples, num_steps)
    else:
        raise ValueError(f"Unconditional model type expected, got: {model_type}")


def generate_samples_conditional(
    model, model_type, device, num_samples=64, num_steps=1000, scheduler_type="cosine"
):
    samples = {}

    for class_label in tqdm(range(10), desc=f"Generating samples for {model_type}"):
        if model_type == "ddpm_cond":
            class_samples = sample_ddpm_cond(
                model, class_label, device, num_samples, num_steps, scheduler_type
            )
        elif model_type == "d3pm_cond":
            class_samples = sample_d3pm_cond(
                model, class_label, device, num_samples, num_steps
            )
        else:
            raise ValueError(f"Conditional model type expected, got: {model_type}")

        samples[class_label] = class_samples

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from trained models")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="Directory to save generated samples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples to generate per class (for conditional) or total (for unconditional)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if models directory exists
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models directory '{args.models_dir}' does not exist!")
        return

    # Get all model files
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print(f"No .pth files found in '{args.models_dir}'")
        return

    print(f"Found {len(model_files)} model files")

    for model_file in tqdm(model_files, desc="Processing models"):
        try:
            params = parse_model_filename(model_file.name)
            model_type = params["model_type"]
            scheduler_type = params["scheduler"]
            num_steps = int(params["steps"])

            print(
                f"\nProcessing {model_file.name} with {num_steps} steps and {scheduler_type} scheduler"
            )

            model = load_model(str(model_file), model_type, device)

            experiment_name = model_file.stem  # Remove .pth extension
            output_experiment_dir = Path(args.output_dir) / experiment_name
            output_experiment_dir.mkdir(parents=True, exist_ok=True)

            if model_type in ["ddpm", "d3pm"]:
                print(f"Generating {args.num_samples} samples...")
                samples = generate_samples_unconditional(
                    model,
                    model_type,
                    device,
                    args.num_samples,
                    num_steps,
                    scheduler_type,
                )

                output_file = output_experiment_dir / f"samples_{model_type}.pt"
                torch.save(samples, output_file)
                print(f"Saved {args.num_samples} samples to {output_file}")

            else:
                print(
                    f"Generating {args.num_samples} samples per class (10 classes)..."
                )
                samples_dict = generate_samples_conditional(
                    model,
                    model_type,
                    device,
                    args.num_samples,
                    num_steps,
                    scheduler_type,
                )

                for class_label, class_samples in samples_dict.items():
                    output_file = (
                        output_experiment_dir / f"samples_{model_type}_{class_label}.pt"
                    )
                    torch.save(class_samples, output_file)
                    print(
                        f"Saved {len(class_samples)} samples for class {class_label} to {output_file}"
                    )

            print(f"Completed processing {model_file.name}")

        except Exception as e:
            print(f"Error processing {model_file.name}: {str(e)}")
            continue

    print(f"\nAll samples saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
