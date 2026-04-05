# Wolo
A small language model (SLM) trained solely on Wikipedia.

In this project directory is a gitignored data folder. Inside it contains the zim file in which this SLM was trained on. The 115GB file can be found here: https://library.kiwix.org/ 

## Training Setup

This repository includes a CPU-friendly PyTorch trainer for a byte-level language model that samples text directly from the Wikipedia ZIM archive.

### Quick start

Simply run:
```bash
./scripts/train.sh
```

That's it! The script handles:
- Creating a virtual environment at `~/.venvs/wolo`
- Installing CPU-optimized PyTorch (avoids native library conflicts)
- Installing dependencies
- Auto-resuming from the latest checkpoint if one exists
- Saving checkpoints every 100 steps to `checkpoints/`
- Logging metrics to `runs/metrics.jsonl`

### Manual setup (if needed)

1. Create a virtual environment with `python3 -m venv ~/.venvs/wolo`.
2. Install CPU PyTorch with `~/.venvs/wolo/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch`.
3. Install project dependencies with `~/.venvs/wolo/bin/pip install -r requirements.txt`.
4. Run `~/.venvs/wolo/bin/python src/main.py` from the project root.

### What it does

- Samples random entries from the full Wikipedia ZIM archive in `data/wiki_en_all_maxi_2026-02.zim`
- Strips HTML and trains a byte-level next-token language model
- Writes checkpoints to `checkpoints/`
- Writes structured metrics to `runs/metrics.jsonl`

### Training options

All arguments are optional:

```bash
# Start fresh (ignore existing checkpoints)
./scripts/train.sh --fresh

# Customize training parameters
./scripts/train.sh --steps 1000 --batch-size 2 --sequence-length 128 --embedding-dim 96

# Resume from a specific checkpoint
./scripts/train.sh --resume-from checkpoints/step_000050.pt

# Combine options
./scripts/train.sh --steps 2000 --checkpoint-every 50
```

By default, training auto-resumes from the latest checkpoint when parameters match the current architecture.