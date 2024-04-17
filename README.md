# graphormer-pyg
[Microsoft Graphormer](https://github.com/microsoft/Graphormer) rewritten in PyTorch-Geometric

![image](https://github.com/leffff/graphormer-pyg/assets/57654885/34c1626e-aa71-4f2a-a12c-0d5900d32cbf)

Reimplemented to support the Ames-graphormer project for regulatory mutagenicity detection.

# Implemented Layers
1. Centrality Encoding
2. Spatial Encoding
3. Edge Encoding
4. Multi-Head Self-Attention

# Warning
This implementation differs from the original implementation in the paper in following ways:
1. No [VNode] ([CLS] token analogue in BERT)

# Installation
## Requirements
This repository includes some tools which are built using [Rust](https://www.rust-lang.org/) and create python bindings with [Maturin](https://github.com/PyO3/maturin).  These must both be installed in order to build from source.

## With Poetry
Installation is simplest with [Poetry](https://python-poetry.org/docs/). Run:
- `poetry lock --no-update` to gather the required information and populate caches
- `poetry install` to furnish a virtual environment.  
- `poetry run train` to begin training the model.  See `poetry run train --help` for options.

# Visualization
## Requirements
Visualization is provided with `tensorboard`.  To see local readouts during training, first install `tensorboard` on your system.  We recommend via [pipx](https://github.com/pypa/pipx):
- `pipx install tensorboard`

## Local Server
You can start a local tensorboard server via the following command:
- `tensorboard --logdir=<logdir>`, where `<logdir>` is the path to the tensorboard logs.  By default, these are created in the `runs` folder.
