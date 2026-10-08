# SLS Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean, portable, public version of the SLS research code and paper at `https://github.com/Chenzheng402/SLS`.

**Architecture:** Preserve the active training and reward logic while replacing machine-specific configuration with a command-line entry point. Package only the canonical trainer and supporting utilities, validate the release statically, then publish the committed `main` branch through GitHub CLI.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Accelerate, DeepSpeed, vLLM, Hugging Face Datasets, SwanLab, Git, GitHub CLI, qrencode

---

### Task 1: Install GitHub tooling and authenticate

**Files:** None

- [ ] **Step 1: Install and verify tools**

```bash
brew install gh qrencode
gh --version
qrencode --version
```

Expected: both commands print version information.

- [ ] **Step 2: Authenticate through the browser**

```bash
gh auth login --hostname github.com --git-protocol https --web
```

Expected: GitHub CLI gives a one-time code and browser URL; the user authorizes the `Chenzheng402` account.

- [ ] **Step 3: Verify identity**

```bash
gh auth status
gh api user --jq .login
```

Expected: the second command prints exactly `Chenzheng402`. Stop if it prints another account.

### Task 2: Curate the canonical source tree

**Files:**
- Create: `paper/SLS.pdf`
- Create: `src/train.py`
- Create: `src/rewards.py`
- Create: `src/trainer/__init__.py`
- Create: `src/trainer/sls_trainer.py`
- Create: `src/trainer/data_utils.py`
- Create: `src/trainer/utils.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p paper src/trainer configs scripts
```

- [ ] **Step 2: Copy the selected artifacts**

Copy these mappings without modifying the originals:

```text
SLS_code/未命名/13801_Sample_Learning_State_Gu.pdf -> paper/SLS.pdf
SLS_code/未命名/src/SLS.py -> src/train.py
SLS_code/未命名/src/rewards.py -> src/rewards.py
SLS_code/未命名/src/trainer/SLS_trainer_new.py -> src/trainer/sls_trainer.py
SLS_code/未命名/src/trainer/data_utils.py -> src/trainer/data_utils.py
SLS_code/未命名/src/trainer/utils.py -> src/trainer/utils.py
```

Use filesystem copy for the binary PDF and `apply_patch` for text files. Create an empty `src/trainer/__init__.py`.

- [ ] **Step 3: Confirm excluded trainers are absent**

```bash
find src/trainer -maxdepth 1 -type f -print | sort
```

Expected: only `__init__.py`, `data_utils.py`, `sls_trainer.py`, and `utils.py`.

- [ ] **Step 4: Commit**

```bash
git add paper/SLS.pdf src
git commit -m "feat: add canonical SLS implementation and paper"
```

### Task 3: Make the entry point portable

**Files:**
- Modify: `src/train.py`
- Modify: `src/trainer/sls_trainer.py`

- [ ] **Step 1: Replace private class-level configuration**

Remove unused `datasets`, `DatasetDict`, `datetime`, and `time` imports. Add `argparse`. Replace `SLS_Config` with:

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Train SLS with dynamic SFT-RL weighting.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--system-prompt", default=(
        "You are a helpful AI assistant that provides detailed thought processes "
        "and correct answers. Your final answer must be placed within \\\\boxed{}."
    ))
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--learning-rate", dest="lr", type=float, default=5e-6)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-per-batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward-top", type=float, default=0.99)
    parser.add_argument("--sls-lambda", dest="l", type=float, default=2.0)
    parser.add_argument("--std-max", type=float, default=0.5)
    parser.add_argument("--rl-mode", choices=("dapo", "gspo"), default="gspo")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-rl-length", type=int, default=8192)
    parser.add_argument("--reward-funcs", nargs="+", default=["accuracy_soft"])
    parser.add_argument("--project-name", default="SLS")
    parser.add_argument("--experiment-name", default="sls-training")
    parser.add_argument("--log-mode", choices=("cloud", "local", "offline", "disabled"), default="offline")
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    args.reward_weights = [1.0] * len(args.reward_funcs)
    args.reward_num = len(args.reward_funcs)
    args.log_dir = args.log_dir or args.output_dir
    return args
```

Replace the final construction with `main(parse_args())`. Retain the existing dataset transformation, reward registry, model creation, trainer construction, training, and save calls.

- [ ] **Step 2: Update the canonical import and remove the debugger**

Change the import to:

```python
from trainer.sls_trainer import SLS_Trainer
```

Delete `import ipdb` from `src/trainer/sls_trainer.py`. Do not change trainer computations.

- [ ] **Step 3: Validate and scan**

```bash
python3 -m compileall -q src
rg -n '/mnt/|/mlx_devbox/|SLS_trainer_new|import ipdb' src
```

Expected: compilation succeeds and the scan has no matches.

- [ ] **Step 4: Commit**

```bash
git add src/train.py src/trainer/sls_trainer.py
git commit -m "refactor: make SLS training configuration portable"
```

### Task 4: Add runtime configuration

**Files:**
- Create: `configs/accelerate_zero2.yaml`
- Create: `scripts/train.sh`
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Add Accelerate configuration**

Create `configs/accelerate_zero2.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: "no"
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
rdzv_backend: static
same_network: true
use_cpu: false
```

- [ ] **Step 2: Add portable launcher**

Create executable `scripts/train.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MODEL_PATH DATASET_PATH OUTPUT_DIR [additional train.py arguments...]" >&2
  exit 2
fi

MODEL_PATH=$1
DATASET_PATH=$2
OUTPUT_DIR=$3
shift 3

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

accelerate launch \
  --config_file "$REPO_ROOT/configs/accelerate_zero2.yaml" \
  "$REPO_ROOT/src/train.py" \
  --model-name-or-path "$MODEL_PATH" \
  --dataset-name "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
```

- [ ] **Step 3: Record dependencies**

Create `requirements.txt`:

```text
accelerate==1.4.0
datasets==4.1.0
deepspeed==0.16.8
flash-attn==2.8.3
latex2sympy2-extended==1.0.6
math-verify==0.5.2
numpy==2.2.6
swanlab==0.6.10
torch==2.7.0
tqdm==4.67.1
transformers==4.52.3
vllm==0.9.2
```

- [ ] **Step 4: Add exclusions**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.env
.DS_Store
.idea/
.vscode/
*.log
wandb/
swanlog/
outputs/
checkpoints/
step_*/
*.pt
*.pth
*.bin
*.safetensors
data/
```

- [ ] **Step 5: Validate and commit**

```bash
bash -n scripts/train.sh
python3 -c 'import yaml; yaml.safe_load(open("configs/accelerate_zero2.yaml"))'
git add .gitignore requirements.txt configs scripts
git commit -m "build: add portable SLS launch configuration"
```

Expected: validation exits 0 before the commit.

### Task 5: Add documentation and citation

**Files:**
- Create: `README.md`
- Create: `CITATION.cff`

- [ ] **Step 1: Write README**

Create an English `README.md` with these sections and exact facts:

1. Official implementation statement and paper title.
2. Method: `M` is mean core reward, `D` is composite-reward standard deviation, and `alpha = min(1, lambda * (exp(M) - 1) * D / D_max)`.
3. Actual repository structure.
4. Installation: Python 3.10+, CUDA Linux, `pip install -r requirements.txt`, and CUDA/PyTorch compatibility warning for vLLM and FlashAttention.
5. Dataset contract: Hugging Face `DatasetDict` loaded from disk, with a `train` split and `problem`, `solution`, and `problem_type` fields.
6. Launch command: `bash scripts/train.sh Qwen/Qwen2.5-Math-1.5B /path/to/dataset /path/to/output --batch-size 32 --num-generations 4 --learning-rate 5e-6`.
7. Paper setup: 30k OpenR1-Math plus 2k GSM8K; Qwen2.5-Math 1.5B/7B; four rollouts; temperature 1.0; 8,192 output tokens; lambda 2; four A800-80GB GPUs.
8. Note that the preserved code snapshot had also been used with Qwen3-4B, while reported experiments use Qwen2.5-Math.
9. Link to `paper/SLS.pdf` and citation block.
10. License note: no license has been selected; reuse requires permission.

- [ ] **Step 2: Add citation metadata**

Create CFF 1.2.0 software metadata with repository URL, title, year 2026, and authors in this order: Zheng Yang, Yilin Liu, Zhuoran Li, Wenhao Gao, Bingwei Li, Haozhe Sun, Caixia Yuan, Fan Wan, Chen Wu, Xiaojie Wang.

- [ ] **Step 3: Verify and commit**

```bash
test -f paper/SLS.pdf
rg -n 'paper/SLS.pdf|Qwen2.5-Math|OpenR1-Math|GSM8K|A800|8192|CITATION.cff' README.md
git add README.md CITATION.cff
git commit -m "docs: document SLS reproduction and citation"
```

### Task 6: Validate the complete release

**Files:** All tracked release files

- [ ] **Step 1: Run static checks**

```bash
python3 -m compileall -q src
bash -n scripts/train.sh
python3 -c 'import yaml; yaml.safe_load(open("configs/accelerate_zero2.yaml"))'
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Scan for private or sensitive content**

```bash
rg -n '/mnt/|/mlx_devbox/|api[_-]?key|access[_-]?token|secret|password|passwd|BEGIN .*PRIVATE KEY' . --glob '!docs/superpowers/**'
find . -type f \( -name 'SLS_trainer_old.py' -o -name 'SLS_trainer1.py' -o -name 'sls_trainer_n.py' \)
```

Expected: no matches.

- [ ] **Step 3: Review manifest and status**

```bash
git ls-files
du -sh .
git status --short
```

Expected: only intended release files are tracked and status is clean.

### Task 7: Publish and generate QR codes

**Files:**
- Create outside repository: `/Users/bytedance/Desktop/middle/may/SLS-GitHub-QR.png`
- Create outside repository: `/Users/bytedance/Desktop/middle/may/SLS-GitHub-QR.svg`

- [ ] **Step 1: Confirm the target is absent**

```bash
gh repo view Chenzheng402/SLS
```

Expected before creation: repository not found. If it exists, stop rather than overwrite.

- [ ] **Step 2: Create and push**

```bash
gh repo create Chenzheng402/SLS --public --source=. --remote=origin --push --description "Official implementation of Sample Learning State guided dynamic SFT-RL integration"
```

- [ ] **Step 3: Verify publication**

```bash
gh repo view Chenzheng402/SLS --json nameWithOwner,visibility,url,defaultBranchRef
git remote -v
git status --short --branch
```

Expected: `Chenzheng402/SLS`, `PUBLIC`, the intended URL, default branch `main`, and clean tracking state.

- [ ] **Step 4: Generate QR codes**

```bash
qrencode -o /Users/bytedance/Desktop/middle/may/SLS-GitHub-QR.png -s 10 -m 2 'https://github.com/Chenzheng402/SLS'
qrencode -t SVG -o /Users/bytedance/Desktop/middle/may/SLS-GitHub-QR.svg -m 2 'https://github.com/Chenzheng402/SLS'
```

Expected: both files exist and are non-empty. If `zbarimg` is available, decode the PNG and verify the payload is exactly the repository URL.

