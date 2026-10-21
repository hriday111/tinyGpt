# tinyGpt

A from-scratch, character-level GPT trained on Shakespeare (`input.txt`), following [Andrej Karpathy's nanoGPT / Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).

The repo covers both sides of a real ML workflow:

- **Model:** bigram baseline → scaled dot-product self-attention Transformer in PyTorch
- **Cloud:** cost-capped training on **Azure Machine Learning** with **Azure CLI** (`az` / `az ml`), using a single **NVIDIA T4** node that scales to zero when idle

---

## LLM use disclosure

Agentic workflows have been strictly used to generate doxygen compatible documentation along with `README.md`and `Doxyfile` files.
However, for the sake of learning, use of AI has been avoided for all automations and python programs.

## Model progression

### 1. Bigram baseline — `bigram.py`

Next-token prediction from the current character only (no long context). Used as a loss/quality baseline before attention.

### 2. Self-attention Transformer — `sdp_attention.py`

Scaled dot-product attention (heads → multi-head), feed-forward blocks, and stacked Transformer layers. Training reduces train/val loss relative to the bigram baseline.

Target hyperparameters used for the cloud run:

| Hyperparameter | Value |
| -------------- | ----- |
| `batch_size`   | 64    |
| `block_size`   | 256   |
| `n_embd`       | 384   |
| `n_head`       | 6     |
| `n_layer`      | 6     |
| `max_iters`    | 5000  |
| Optimizer      | AdamW |

These settings exhausted local VRAM (~4 GB), which is why training runs on Azure ML with a T4.

---

## Azure ML architecture

```text
Laptop (az CLI)
    │
    ▼
Azure ML workspace  mlw-tinygpt
    │
    ├── Compute cluster  gpu-t4-lp
    │     size: Standard_NC4as_T4_v3  (1× T4 16 GB)
    │     min_instances: 0
    │     max_instances: 1
    │     tier: low_priority (spot) when available
    │
    └── Job  (az ml job create -f azure/job.yml)
          trains sdp_attention.py on input.txt
          cluster returns to 0 nodes after the job
```

### Cost controls

- Subscription budget / spend alert (~$1–3 for learning runs)
- One GPU at a time (`max_instances: 1`); idle scale-down (`min_instances: 0`)
- Prefer **low_priority** T4 to keep a short job well under **$1**
- Delete compute (and optionally the resource group) after the run — no always-on VMs

### Resources

| Resource        | Name                   | Region      |
| --------------- | ---------------------- | ----------- |
| Resource group  | `rg-tinygpt`           | East US     |
| AML workspace   | `mlw-tinygpt`          | East US     |
| Compute cluster | `gpu-t4-lp`            | East US     |
| VM size         | `Standard_NC4as_T4_v3` | 1× T4 16 GB |

---

## Setup completed (Azure CLI)

| Step | Action                                       | Status |
| ---- | -------------------------------------------- | ------ |
| 1    | Install Azure CLI + `ml` extension           | Done   |
| 2    | `az login` / select subscription             | Done   |
| 3    | Budget alert in Cost Management              | Done   |
| 4    | Create resource group `rg-tinygpt`           | Done   |
| 5    | Create workspace `mlw-tinygpt`               | Done   |
| 6    | Create T4 cluster `gpu-t4-lp` (min=0, max=1) | Done   |
| 7    | Submit one training job (`az ml job create`) | Done   |
| 8    | Scale to zero / delete compute after the job | Done   |

### Reproduce the foundation

```bash
az version
az extension add -n ml

az login
az account set --subscription "<your-subscription>"

az group create -n rg-tinygpt -l eastus
az ml workspace create -n mlw-tinygpt -g rg-tinygpt

# Compute: 1× T4, scale-to-zero (see azure/compute.yml)
az ml compute create --file azure/compute.yml \
  --resource-group rg-tinygpt \
  --workspace-name mlw-tinygpt

az ml compute list -g rg-tinygpt -w mlw-tinygpt -o table
```

Configs live under `azure/` (`compute.yml` for T4, `compute-cpu.yml` fallback, `job.yml`).

### Submit one training job

```bash
# create the T4 cluster once (or use -f azure/compute-cpu.yml if no GPU quota)
az ml compute create -f azure/compute.yml -g rg-tinygpt -w mlw-tinygpt

az ml job create -f azure/job.yml \
  -g rg-tinygpt \
  -w mlw-tinygpt
```

The job allocates the T4 node, runs `sdp_attention.py`, writes metrics/checkpoints to the workspace datastore, then the cluster scales back to **0 nodes**.

### Teardown

```bash
az ml compute delete -n gpu-t4-lp -g rg-tinygpt -w mlw-tinygpt -y

# optional full cleanup when the project is finished:
# az group delete -n rg-tinygpt -y --no-wait
```

---

## Local run

Useful for debugging smaller hyperparameter settings before a cloud job:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch

python bigram.py
python sdp_attention.py
```

If you hit CUDA OOM locally, lower `batch_size` / `block_size`, or rely on the Azure T4 job for the full config.

---

## API docs (Doxygen)

Python modules use Doxygen-style `##` / `@brief` / `@param` comments. HTML docs are built by GitHub Actions and published to GitHub Pages on every push to `main`.

```bash
# local preview
sudo dnf install doxygen graphviz   # or: sudo apt install doxygen graphviz
doxygen Doxyfile
xdg-open docs/doxygen/html/index.html
```

Enable Pages once: **Repo → Settings → Pages → Source: GitHub Actions**.

After the first successful `Docs` workflow run, the site is at:

`https://hriday111.github.io/tinyGpt/`

---

## Project layout

```text
tinyGpt/
  bigram.py                 # bigram baseline (+ Doxygen comments)
  sdp_attention.py          # multi-head Transformer (+ Doxygen comments)
  input.txt                 # Shakespeare corpus
  Doxyfile                  # Doxygen config
  docs/mainpage.md          # docs landing page
  .github/workflows/docs.yml
  azure/
    compute.yml             # T4 cluster (min=0, max=1, low_priority)
    compute-cpu.yml         # F4s_v2 fallback if no GPU quota
    environment.yml         # optional conda env reference
    job.yml                 # single AML training job
  README.md
```

## Acknowledgments

- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/) and [Azure CLI (`az ml`)](https://learn.microsoft.com/en-us/cli/azure/ml) documentation
