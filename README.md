# Sample Learning State (SLS)

Official implementation of **Sample Learning State Guided Dynamic SFT--RL Integration Post-training**.

SLS performs single-stage post-training by assigning each sample a dynamic mixture of supervised fine-tuning (SFT) and reinforcement learning (RL). The mixture follows the model's current learning state instead of using a fixed global loss weight.

## Method

For each training prompt, the current policy generates a group of rollout trajectories. SLS measures:

- **Mastery** `M`: the mean core-task reward across the rollout group.
- **Dispersion** `D`: the standard deviation of the composite rewards across the same group.

The sample-specific RL weight is

```text
alpha = min(1, lambda * (exp(M) - 1) * D / D_max).
```

The training objective is `(1 - alpha) * L_SFT + alpha * L_RL`. Fully saturated samples whose rollouts all achieve the maximum reward are filtered because they provide no group-relative contrast.

See [the paper](paper/SLS.pdf) for the derivation, analysis, and experimental results.

## Repository Structure

```text
.
├── paper/SLS.pdf                  # Paper PDF
├── configs/accelerate_zero2.yaml # Accelerate + DeepSpeed ZeRO-2 config
├── scripts/train.sh              # Portable multi-GPU launcher
├── src/train.py                  # Training entry point and arguments
├── src/rewards.py                # Math, format, length, and type rewards
└── src/trainer/
    ├── sls_trainer.py            # SLS rollout, selection, and optimization
    ├── data_utils.py             # Chat formatting and batch utilities
    └── utils.py                  # Distributed/model helper functions
```

## Installation

The training stack requires Linux, NVIDIA CUDA GPUs, and Python 3.10 or newer. The preserved experiment environment used PyTorch 2.7.0, Transformers 4.52.3, vLLM 0.9.2, and DeepSpeed 0.16.8.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Match the PyTorch, vLLM, FlashAttention, and CUDA builds to the CUDA version installed on the training machine. `flash-attn` may need to be installed after PyTorch with `--no-build-isolation` on some systems.

## Dataset Format

`src/train.py` loads a Hugging Face `DatasetDict` produced by `save_to_disk`. It expects a `train` split. Each example must contain:

- `messages`: the supervised conversation as a list of `{role, content}` objects;
- `problem`: the user-side math problem used for online rollout generation;
- `solution`: the reference solution consumed by the correctness reward;
- `problem_type`: the problem category consumed when the optional type reward is enabled.

The tokenizer chat template converts `messages` into the supervised `text`, token IDs, attention mask, and labels at load time.

## Training

The launcher uses four processes by default. Override that setting through `NUM_PROCESSES`.

```bash
bash scripts/train.sh \
  Qwen/Qwen2.5-Math-1.5B \
  /path/to/dataset \
  /path/to/output \
  --batch-size 32 \
  --num-generations 4 \
  --max-rl-length 8192 \
  --learning-rate 5e-6
```

For a different GPU count:

```bash
NUM_PROCESSES=2 bash scripts/train.sh MODEL_PATH DATASET_PATH OUTPUT_DIR
```

Run `python src/train.py --help` to see all model, reward, logging, rollout, and optimization arguments.

## Paper Configuration

The reported experiments use:

- 30,000 filtered OpenR1-Math samples and 2,000 GSM8K samples;
- Qwen2.5-Math-1.5B and Qwen2.5-Math-7B base models;
- four rollout trajectories per sample at temperature 1.0;
- an 8,192-token output limit and 16,384-token extended context;
- `lambda = 2`, with `D_max` determined by the active reward configuration;
- four NVIDIA A800-80GB GPUs.

The preserved implementation snapshot was subsequently exercised with a Qwen3-4B configuration. The public entry point removes that machine-specific default and requires the model path explicitly. The results reported in the paper are based on the Qwen2.5-Math configurations above.

## Citation

Citation metadata is also available in [CITATION.cff](CITATION.cff).

```bibtex
@article{yang2026sls,
  title   = {Sample Learning State Guided Dynamic SFT--RL Integration Post-training},
  author  = {Yang, Zheng and Liu, Yilin and Li, Zhuoran and Gao, Wenhao and Li, Bingwei and Sun, Haozhe and Yuan, Caixia and Wan, Fan and Wu, Chen and Wang, Xiaojie},
  year    = {2026}
}
```

## License

No open-source license has been selected for this release. Copyright remains with the authors, and reuse requires permission from the copyright holders.
