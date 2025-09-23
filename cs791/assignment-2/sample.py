import argparse
import os
import re

import torch

from d3pm import sample as sample_d3pm
from d3pm_cond import sample as sample_d3pm_cond
from ddpm import sample as sample_ddpm
from ddpm_cond import sample as sample_ddpm_cond
from models import D3PM, DDPM, ConditionalD3PM, ConditionalDDPM
from utils import seed_everything, visualize_samples

seed_everything(42)


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


parser = argparse.ArgumentParser(description="DDPM Model Template")
parser.add_argument(
    "model_path",
    type=str,
    help="Full path to the model file (e.g., './exps_d3pm/10ep_64bs_0.0001lr_1000steps_cosinescheduler/model.pth')",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = parse_model_filename(args.model_path)
model = load_model(args.model_path, params["model_type"], device)

model_type = params["model_type"]
if model_type == "ddpm":
    samples = sample_ddpm(model, device, 16, int(params["steps"]), params["scheduler"])
elif model_type == "d3pm":
    samples = sample_d3pm(model, device, 16, int(params["steps"]))
elif model_type == "ddpm_cond":
    samples = sample_ddpm_cond(
        model, 3, device, 16, int(params["steps"]), params["scheduler"]
    )
elif model_type == "d3pm_cond":
    samples = sample_d3pm_cond(model, 3, device, 16, int(params["steps"]))
else:
    raise ValueError(f"Unknown model type: {model_type}")

os.makedirs("samples", exist_ok=True)
visualize_samples(samples, f"samples/samples_{os.path.basename(args.model_path)}.png")
