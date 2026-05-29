# Particle Tomography Results

This repository contains the scripts used to generate the reconstructions and plots for *Particle Tomography: A Grid-Free Approach to Tomographic Reconstruction*.

The repository is not a one-command reproduction of every paper figure. Only the simulated protein dataset is bundled here. The vesicle, thin-film, and platinum datasets are third-party datasets and are not redistributed because of licensing/size constraints. RESIRE reconstructions were generated externally with MATLAB scripts and are treated as optional user-provided outputs.

## Reproducibility Scope

### Included Protein Workflow

The bundled protein dataset supports a reproducible repository-local workflow. It can rerun the available Python methods on the included data and regenerate protein plots from generated outputs.

```bash
python -m scripts.check_data --require-astra
python -m scripts.main
```

By default, `scripts.main` runs the bundled dataset only and benchmarks `particle_tomography`, `SIRT`, and `FBP`. SIRT/FBP require ASTRA Toolbox. To run only particle tomography, use:

```bash
python -m scripts.main --methods particle_tomography
```

The particle-tomography benchmark uses `PARTICLE_TOMOGRAPHY_DEVICE=auto`, which selects CUDA when available and otherwise uses CPU. To force a device:

```bash
PARTICLE_TOMOGRAPHY_DEVICE=cuda python -m scripts.main --methods particle_tomography
```

### External Full-Paper Inputs

To reproduce the full paper workflow, place third-party datasets under the paths checked by:

```bash
python -m scripts.check_data --include-external
```

Dataset sources:

- Vesicle and thin-film data: https://zenodo.org/records/7819857
- Platinum particle data: https://springernature.figshare.com/collections/Nanomaterial_datasets_to_advance_tomography_in_scanning_transmission_electron_microscopy/2185342

Expected paths are encoded in `scripts/datasets.py` and printed by `scripts.check_data`.

### RESIRE Outputs

RESIRE outputs are not included. The MATLAB scripts used for those reconstructions are in `scripts/reconstruct/resire_matlab_scripts`, but reproducing them requires MATLAB and the original RESIRE workflow. To include RESIRE in metrics/plots, place each output at:

```text
out/RESIRE/<dataset>/RESIRE_volume_reconstruction.mat
```

Then verify the files and run with RESIRE enabled:

```bash
python -m scripts.check_data --include-external --include-resire
python -m scripts.main --include-external --with-resire
```

## Installation

Create the conda environment for ASTRA and the scientific Python stack:

```bash
conda env create -f environment.yml
conda activate particle-tomography-results
```

Install PyTorch separately so the CPU/CUDA build matches your system. Follow the official PyTorch instructions: https://pytorch.org/get-started/locally/

Install `particle-tomography` and its `differentiable-rasterizer` dependency using the instructions in the main package repository. The optional custom CUDA rasterizer backend requires the CUDA Toolkit and `python build_cuda.py` in a local `differentiable-rasterizer` clone; otherwise the PyTorch rasterizer fallback is used.

## Useful Commands

Check bundled data and core imports:

```bash
python -m scripts.check_data
```

Check ASTRA-dependent methods:

```bash
python -m scripts.check_data --require-astra
```

Run protein benchmarks without generating plots:

```bash
python -m scripts.main --skip-plots
```

Run one method on the bundled protein dataset:

```bash
python -m scripts.main --methods particle_tomography --skip-plots
```

Check all registered external inputs and RESIRE outputs:

```bash
python -m scripts.check_data --include-external --include-resire --require-astra
```
