#!/usr/bin/env python3
"""Task 3: Twisted Sequential Monte Carlo (TSMC) — CLI.

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
            "method": "TSMC[topk=K; N=B; beta=β]",
            "samples": [                 # only fields needed by eval/plots
                {"text": str, "weight": float}, ...
            ],
            "normalized_weights": [float, ...]
        }]
    }
"""

import argparse

from generate_task3_tsmc import load_counts_and_reward, load_model, tsmc_for_prompt
from utils import ensure_dir, load_jsonl, save_jsonl, set_seed


def parse_args():
    """Parse CLI options for Task 3 (TSMC).

    Uses fixed model: meta-llama/Meta-Llama-3-8B-Instruct

    Returns:
        argparse.Namespace with:
            - hf_token, device
            - counts_dir, epsilon
            - beta (terminal reward scaling)
            - k (top-k sampling parameter)
            - test_file, A, B (particles), seed
            - out
    """
    p = argparse.ArgumentParser(
        description="Task 3: Twisted Sequential Monte Carlo (TSMC)"
    )

    # HF model
    p.add_argument("--hf-token", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")

    # counts + reward
    p.add_argument("--counts-dir", type=str, required=True)
    p.add_argument("--epsilon", type=float, default=1e-9)

    # terminal reward and resampling
    p.add_argument(
        "--beta", type=float, default=5.0, help="Terminal reward scale in exp(beta*R)"
    )

    # base proposal shaping - only top-k sampling for Task 3
    p.add_argument(
        "--k", type=int, default=10, help="Top-k parameter for base proposal q"
    )

    # dataset/run
    p.add_argument("--test-file", type=str, default="data/test_prompts.jsonl")
    p.add_argument("--A", type=int, required=True, help="Number of prompts")
    p.add_argument("--B", type=int, required=True, help="Particles per prompt (N=B)")
    p.add_argument("--seed", type=int, default=123)

    # output
    p.add_argument("--out", type=str, default="data/outputs_task3_TSMC.jsonl")
    return p.parse_args()


def _method_tag(args, N: int) -> str:
    """Generate method identification string for output tracking.

    Args:
        args: Parsed command line arguments
        N: Number of particles used

    Returns:
        str: Method identification string with key hyperparameters
    """
    return f"TSMC[topk={args.k}; N={N}; beta={args.beta}]"


def main():
    # Parse command line arguments and initialize environment
    args = parse_args()
    set_seed(args.seed)  # Ensure reproducible results
    ensure_dir("data")  # Create output directory if needed

    # Load test prompts and model components
    rows = load_jsonl(args.test_file)[: args.A]  # Load first A prompts
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok, model, eos_id = load_model(model_name, args.hf_token, args.device)
    reward_calc = load_counts_and_reward(args.counts_dir, epsilon=args.epsilon)

    out_rows = []
    for row in rows:
        prompt_id = row["prompt_id"]
        prefix = row["prefix"]
        instruction = row.get("instruction", "Continue the text.")
        full_prompt = f"{instruction} {prefix}"
        max_new = int(row.get("max_output_tokens", 50))

        result = tsmc_for_prompt(
            tok,
            model,
            reward_calc,
            prefix=full_prompt,
            N=args.B,
            max_new_tokens=max_new,
            eos_id=eos_id,
            beta=args.beta,
            k=args.k,
        )

        out_rows.append(
            {
                "prompt_id": prompt_id,
                "prefix": prefix,
                "continuations": [
                    {
                        "method": _method_tag(args, args.B),
                        "samples": result["samples"],
                        "normalized_weights": result["normalized_weights"],
                    }
                ],
            }
        )

    save_jsonl(args.out, out_rows)
    print(
        f"[task3:TSMC] wrote {len(out_rows)} prompts × {args.B} particles → {args.out}"
    )


if __name__ == "__main__":
    main()
