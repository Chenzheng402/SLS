# SLS Public Release Design

## Goal

Create a clean public research-code repository for *Sample Learning State Guided Dynamic SFT--RL Integration Post-training*. The repository will preserve the current SLS algorithm, expose portable training configuration, include the paper PDF, and be published as `https://github.com/Chenzheng402/SLS`.

## Source of Truth

- Code source: `/Users/bytedance/Desktop/middle/may/SLS_code/未命名/src`
- Canonical trainer: `trainer/SLS_trainer_new.py`, because it is imported by the current `SLS.py` entry point
- Paper: `/Users/bytedance/Desktop/middle/may/SLS_code/未命名/13801_Sample_Learning_State_Gu.pdf`
- Paper configuration reference: `/Users/bytedance/Desktop/middle/may/SLS/duc/3.method.tex` and `/Users/bytedance/Desktop/middle/may/SLS/duc/4.experimental setup.tex`

The source directories remain unchanged. All release work happens in `/Users/bytedance/Desktop/middle/may/SLS-public`.

## Repository Layout

```text
SLS-public/
├── README.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── paper/
│   └── SLS.pdf
├── configs/
│   └── accelerate_zero2.yaml
├── scripts/
│   └── train.sh
├── src/
│   ├── train.py
│   ├── rewards.py
│   └── trainer/
│       ├── __init__.py
│       ├── sls_trainer.py
│       ├── data_utils.py
│       └── utils.py
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

## Code Curation

The release will include only the trainer used by the active entry point. Historical files `SLS_trainer.py`, `SLS_trainer1.py`, `SLS_trainer_old.py`, and `sls_trainer_n.py` will remain in the original folder but will not be copied into the public repository.

`SLS.py` becomes `src/train.py`, and `SLS_trainer_new.py` becomes `src/trainer/sls_trainer.py`. Imports will be updated to match those names. The mathematical SLS rule, rollout generation, SFT loss, RL loss, filtering, checkpointing, and distributed training behavior will not be redesigned.

Personal absolute paths will be removed. Model, dataset, output, logging mode, GPU count, and main hyperparameters will be configurable through command-line arguments with paper-aligned defaults where a safe default exists. Paths will be required arguments or portable relative paths.

## Documentation and Reproducibility

The English README will contain:

- a concise explanation of mastery, dispersion, and the SLS coefficient;
- installation requirements and tested dependency versions inferred from the preserved environment record;
- expected Hugging Face `DatasetDict` fields;
- a portable multi-GPU launch example;
- the paper settings for Qwen2.5-Math, 32k training samples, four rollouts, 8,192-token generation limit, and four A800-80GB GPUs;
- an explicit note where the currently preserved implementation differs from the paper setup;
- repository structure, paper link, and citation instructions.

`CITATION.cff` will use the author list from the final LaTeX manuscript. No open-source license will be added because the repository owner has not selected one; GitHub will therefore apply default copyright rules.

## Validation

Before publication, the release will be checked with:

1. Python syntax compilation for every Python file.
2. Import and path scans to ensure removed historical trainer names and private `/mnt/...` or `/mlx_devbox/...` paths are absent.
3. Shell syntax validation for `scripts/train.sh`.
4. YAML parsing for the Accelerate configuration when PyYAML is available, with a text-level fallback otherwise.
5. A secret scan for common token, password, private-key, and credential patterns.
6. A clean Git status and review of the exact committed file list.

Full GPU training is outside local validation because the required CUDA hardware, model weights, and 32k dataset are not present on this Mac.

## Publication Flow

Install GitHub CLI with Homebrew, authenticate interactively as `Chenzheng402`, verify the authenticated account, create the public repository `Chenzheng402/SLS`, and push the local `main` branch. After confirming the public URL is reachable, generate PNG and SVG QR codes that encode `https://github.com/Chenzheng402/SLS` and store them outside the repository in the release folder's parent directory.

## Non-goals

- Publishing manuscript LaTeX, reviewer correspondence, posters, archived experiments, datasets, model checkpoints, or training logs.
- Reproducing the full training run locally.
- Rewriting the SLS algorithm or merging historical trainer experiments.
- Choosing a software license on the authors' behalf.
