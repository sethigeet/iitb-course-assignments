#!/bin/bash

source .venv/bin/activate

# start="What is the answer to life, the universe, and everything?"
start="What is the capital of India?;What is the capital of France?;What is the capital of USA?"

python sample.py \
    --init_from=gpt2 \
    --start="${start}" \
    --num_samples=1 \
    --max_new_tokens=100 \
    --use_kv_cache=True \
    --device=cpu