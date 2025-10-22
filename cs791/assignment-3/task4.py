#!/usr/bin/env python3
"""Task 4: Power Distribution Sampling with Metropolis-Hastings MCMC — CLI.

Uses fixed model: meta-llama/Meta-Llama-3-8B-Instruct

Input test row (JSONL):
    {
        "prompt_id": int,
        "prefix": str,
        "instruction": str,          # optional; default "Continue the text."
        "max_output_tokens": int
    }

Output row (JSONL), one per prompt (minimal fields used by eval):
    {
        "prompt_id": int,
        "prefix": str,
        "continuations": [{
            "method": "PowerMCMC[alpha=alpha; steps=S; k=K]",
            "samples": [
                {"text": str, "weight": float}, ...
            ],
            "normalized_weights": [float, ...]
        }]
    }

Usage:
    python task4.py --hf-token <token> --A <num_prompts> --B <num_samples>
                    [--alpha <alpha>] [--steps <steps>] [--k <topk>] [--out <file>]
"""

import argparse
import time

from generate_task4_power import load_model, power_mcmc_for_prompt
from utils import ensure_dir, load_jsonl, save_jsonl, set_seed


def parse_args():
    """Parse CLI options for Task 4 (Power Distribution MCMC).

    Returns:
        argparse.Namespace with:
            - hf_token, device
            - alpha (power parameter for p^alpha(x))
            - steps (number of MCMC steps)
            - k (top-k sampling parameter for proposal)
            - test_file, A, B (samples), seed
            - out
    """
    p = argparse.ArgumentParser(
        description="Task 4: Power Distribution Sampling with Metropolis-Hastings MCMC"
    )

    # HF model
    p.add_argument("--hf-token", type=str, required=False)
    p.add_argument("--device", type=str, default="cuda:0")

    # Power distribution parameters
    p.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="Power parameter alpha for p^alpha(x) distribution",
    )

    # MCMC parameters
    p.add_argument(
        "--steps", type=int, default=10, help="Number of MCMC steps for sampling"
    )

    # Proposal distribution parameters
    p.add_argument(
        "--k", type=int, default=10, help="Top-k parameter for proposal sampling"
    )

    # dataset/run
    p.add_argument("--test-file", type=str, default="data/test_prompts.jsonl")
    p.add_argument("--A", type=int, required=True, help="Number of prompts")
    p.add_argument("--B", type=int, required=True, help="Samples per prompt")
    p.add_argument("--seed", type=int, default=123)

    # output
    p.add_argument("--out", type=str, default="data/outputs_task4_PowerMCMC.jsonl")
    return p.parse_args()


def _method_tag(args) -> str:
    """Generate method identification string for output tracking.

    Args:
        args: Parsed command line arguments

    Returns:
        str: Method identification string with key hyperparameters
    """
    return f"PowerMCMC[alpha={args.alpha}; steps={args.steps}; k={args.k}]"


def main():
    # Parse command line arguments and initialize environment
    args = parse_args()
    set_seed(args.seed)  # Ensure reproducible results
    ensure_dir("data")  # Create output directory if needed

    # Load test prompts and model components
    rows = load_jsonl(args.test_file)[: args.A]  # Load first A prompts
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok, model, eos_id = load_model(model_name, args.hf_token, args.device)

    out_rows = []
    for row in rows:
        prompt_id = row["prompt_id"]
        prefix = row["prefix"]
        instruction = row.get("instruction", "Continue the text.")
        full_prompt = f"{instruction} {prefix}"
        max_new = int(row.get("max_output_tokens", 50))

        print(
            f"[Task 4] Processing prompt {prompt_id}: '{prefix[:50]}{'...' if len(prefix) > 50 else ''}'"
        )
        start_time = time.time()

        result = power_mcmc_for_prompt(
            tok,
            model,
            prefix=full_prompt,
            B=args.B,
            max_new_tokens=max_new,
            eos_id=eos_id,
            alpha=args.alpha,
            steps=args.steps,
            k=args.k,
        )

        end_time = time.time()
        print(
            f"[Task 4] Completed prompt {prompt_id} in {end_time - start_time} seconds"
        )

        out_rows.append(
            {
                "prompt_id": prompt_id,
                "prefix": prefix,
                "continuations": [
                    {
                        "method": _method_tag(args),
                        "samples": result["samples"],
                        "normalized_weights": result["normalized_weights"],
                    }
                ],
            }
        )

    save_jsonl(args.out, out_rows)
    print(
        f"[task4:PowerMCMC] wrote {len(out_rows)} prompts × {args.B} samples → {args.out}"
    )


if __name__ == "__main__":
    main()
