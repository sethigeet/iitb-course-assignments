# AML Assignment 3: Text Generation with Sequential Sampling Methods

This repository contains implementations of various text generation algorithms including baseline methods, Sequential Importance Sampling (SIS), Sequential Monte Carlo (SMC), and Twisted Sequential Monte Carlo (TSMC).

## 📁 File Structure

```
AML/assignment_3_final/
├── README.md                   # This documentation file
├── PA3_Problem_Statement.pdf   # Assignment problem statement
├── .gitignore                  # Git ignore rules
├── .gitattributes             # Git LFS configuration
├── api.py                      # Optimized trigram reward calculator
├── utils.py                    # Shared utilities (I/O, seeding)
├── eval.py                     # Comprehensive evaluation suite
├── task0.py                    # Task 0: Baseline decoding methods CLI
├── task1.py                    # Task 1: Sequential Importance Sampling CLI
├── task2.py                    # Task 2: Sequential Monte Carlo CLI
├── task3.py                    # Task 3: Twisted Sequential Monte Carlo CLI
├── generate_task0.py           # Core baseline implementations
├── generate_task1_is.py        # SIS algorithm implementation
├── generate_task2_smc.py       # SMC algorithm implementation
├── generate_task3_tsmc.py      # TSMC algorithm implementation
├── plot_histogram.py           # Plotting utilities for analysis
├── tinystories_ngrams.tar.gz   # Pre-computed n-gram data archive (235MB)
├── data/
│   └── test_prompts.jsonl      # Test prompts for evaluation (20 prompts)
└── __pycache__/                # Python bytecode cache (auto-generated)
```

## 🗂️ Data Setup

### N-gram Data Setup

The pre-computed n-gram data is required for Tasks 1, 2, and 3. The data is included in this repository as `tinystories_ngrams.tar.gz`.

**Extract the included data:**

```bash
tar -xzf tinystories_ngrams.tar.gz
```

The extracted directory should contain:
- `trigram_probs.pkl` - Pre-computed trigram probabilities (561MB)
- `trigram_counts.json` - Raw trigram counts (392MB) 
- `bigram_counts.json` - Bigram counts (30MB)
- `unigram_counts.json` - Unigram counts (417KB)
- `vocab.json` - Vocabulary mappings (460KB)
- `totals.json` - Count statistics (369B)

### Alternative: Using Your Own N-gram Data

If you have n-gram data in a different location, update the `--counts-dir` parameter in the commands below to point to your directory containing `trigram_probs.pkl`.

## 🚀 Quick Start

### Setting up the codebase [VERY CRUCIAL]

#### 1. Getting Access to Llama Weights
You must first get access to the Llama model weights from HuggingFace:
1. Visit: https://huggingface.co/meta-llama/
2. Request access to `meta-llama/Meta-Llama-3-8B-Instruct`
3. Follow the approval process (may take some time)

#### 2. Creating HuggingFace Access Token
After getting model access, create an authentication token:
1. Visit: https://huggingface.co/docs/hub/en/security-tokens
2. Go to your HuggingFace settings → Access Tokens
3. Create a new token with "Read" permissions
4. Save this token securely - you'll need it for all commands

### Prerequisites

#### Option 1: Conda Environment (Recommended)
```bash
# Create a new conda environment
conda create -n cs791_a3 python=3.10

# Activate the environment
conda activate cs791_a3

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional dependencies
pip install transformers
```

### Environment Setup

```bash
# Set your HuggingFace token (required for all tasks)
export HF_TOKEN="your_huggingface_token_here"
```

**Note:** All tasks use the fixed model `meta-llama/Meta-Llama-3-8B-Instruct` - no need to specify it in commands.

## 📋 Task Instructions

### Task 0: Baseline Decoding Methods

Implements three fundamental text generation approaches: greedy, temperature, and top-k sampling.

#### Greedy Decoding
```bash
python task0.py \
    --hf-token $HF_TOKEN \
    --method greedy \
    --A 10 \
    --B 5 \
    --out data/outputs_task0_greedy.jsonl
```

#### Temperature Sampling
```bash
python task0.py \
    --hf-token $HF_TOKEN \
    --method temperature \
    --tau 1.0 \
    --A 10 \
    --B 5 \
    --out data/outputs_task0_temperature.jsonl
```

#### Top-k Sampling
```bash
python task0.py \
    --hf-token $HF_TOKEN \
    --method topk \
    --k 10 \
    --A 10 \
    --B 5 \
    --out data/outputs_task0_topk.jsonl
```

**Parameters:**
- `--method`: Choose from `greedy`, `temperature`, `topk`
- `--A`: Number of prompts to process
- `--B`: Number of samples per prompt
- `--tau`: Temperature for temperature sampling (default: 1.0)
- `--k`: Top-k parameter for top-k sampling (default: 10)

### Task 1: Sequential Importance Sampling (SIS)

```bash
python task1.py \
    --hf-token $HF_TOKEN \
    --counts-dir tinystories_ngrams \
    --A 10 \
    --B 32 \
    --beta 5.0 \
    --k 10 \
    --out data/outputs_task1_IS.jsonl
```

**Parameters:**
- `--counts-dir`: Path to n-gram data directory
- `--beta`: Reward scaling factor β in exp(β×R(x)) (default: 5.0)
- `--k`: Top-k parameter for proposal sampling (default: 10)
- `--epsilon`: Smoothing parameter for unseen trigrams (default: 1e-9)

### Task 2: Sequential Monte Carlo (SMC)

```bash
python task2.py \
    --hf-token $HF_TOKEN \
    --counts-dir tinystories_ngrams \
    --A 10 \
    --B 32 \
    --beta 5.0 \
    --k 10 \
    --out data/outputs_task2_SMC.jsonl
```

**Parameters:**
- `--counts-dir`: Path to n-gram data directory
- `--beta`: Terminal reward scaling factor (default: 5.0)
- `--k`: Top-k parameter for proposal distribution (default: 10)
- `--B`: Number of particles per prompt

### Task 3: Twisted Sequential Monte Carlo (TSMC)

```bash
python task3.py \
    --hf-token $HF_TOKEN \
    --counts-dir tinystories_ngrams \
    --A 10 \
    --B 32 \
    --beta 5.0 \
    --k 10 \
    --out data/outputs_task3_TSMC.jsonl
```

## 🔧 Common Parameters

### Model Configuration
- `--hf-token`: HuggingFace authentication token (required)
- `--device`: CUDA device (default: "cuda:0")

**Note:** All tasks use the fixed model `meta-llama/Meta-Llama-3-8B-Instruct`

### Dataset Configuration  
- `--test-file`: Path to test prompts (default: "data/test_prompts.jsonl")
- `--A`: Number of prompts to process (required)
- `--B`: Number of samples/particles per prompt (required)
- `--seed`: Random seed for reproducibility (default: 123)

### Reward Function
- `--counts-dir`: Directory with trigram_probs.pkl (required for tasks 1-3)
- `--epsilon`: Smoothing parameter (default: 1e-9)

## 📝 Output Format

All tasks generate JSONL files with the following structure:

```json
{
  "prompt_id": 1,
  "prefix": "The wind whispered through old ruins",
  "continuations": [{
    "method": "Greedy",
    "samples": [
      {"text": "and told ancient stories", "weight": 1.0}
    ],
    "normalized_weights": [1.0]
  }]
}
```

## 🎯 Example Workflow

```bash
# 1. Extract data
tar -xzf tinystories_ngrams.tar.gz

# 2. Run baseline
python task0.py --hf-token $HF_TOKEN --method greedy --A 5 --B 3

# 3. Run SIS  
python task1.py --hf-token $HF_TOKEN --counts-dir tinystories_ngrams --A 5 --B 16

# 4. Run SMC
python task2.py --hf-token $HF_TOKEN --counts-dir tinystories_ngrams --A 5 --B 16

# 5. Run TSMC
python task3.py --hf-token $HF_TOKEN --counts-dir tinystories_ngrams --A 5 --B 16

# 6. Evaluate results
python eval.py --inputs data/outputs_task*.jsonl --counts-dir tinystories_ngrams --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token $HF_TOKEN
```
