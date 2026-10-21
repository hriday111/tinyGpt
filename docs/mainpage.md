# tinyGpt API documentation {#mainpage}

Character-level GPT experiments in PyTorch (Karpathy nanoGPT style), with
Azure ML training and this Doxygen-generated API reference.

## Modules

| File | Role |
|------|------|
| `bigram.py` | Bigram next-token baseline |
| `sdp_attention.py` | Multi-head self-attention Transformer LM |

## Build locally

```bash
sudo dnf install doxygen graphviz   # or apt install doxygen graphviz
doxygen Doxyfile
# open docs/doxygen/html/index.html
```

Docs are also published automatically to GitHub Pages on every push to `main`.
