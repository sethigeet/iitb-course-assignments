import matplotlib.pyplot as plt
import numpy as np

from utils import load_jsonl


def main():
    data = load_jsonl("data/outputs_task1_IS.jsonl")

    normalized_weights = [
        (block["prompt_id"], block["continuations"][0]["normalized_weights"])
        for block in data
    ]

    num_prompts = len(normalized_weights)
    n_cols = int(np.ceil(np.sqrt(num_prompts)))
    n_rows = int(np.ceil(num_prompts / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    bins = np.linspace(0, 1, 11)
    for idx, (prompt_id, weights) in enumerate(normalized_weights):
        ax = axes[idx]
        ax.hist(weights, bins=bins)
        ax.set_title(f"Prompt {prompt_id}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized weight")
        ax.set_ylabel("Count")
        ax.set_xticks(bins)
        ax.set_xticklabels([f"{b:.1f}" for b in bins])

    for idx in range(num_prompts, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    plt.savefig("data/outputs_task1_IS_histogram.png")


if __name__ == "__main__":
    main()
