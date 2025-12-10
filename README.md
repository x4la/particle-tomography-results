# Particle Tomography Results
This is the official repository for the paper *Particle Tomography: A Grid-Free Approach to Tomographic Reconstruction*. It contains all scripts that were used to create the reconstructions and plots shown in the paper. The actual Particle Tomography algorithm is implemented [here](https://github.com/microscopic-image-analysis/particle-tomography).

To reproduce the results shown in the paper, follow these steps:

## 1. Clone the repository

```bash
git clone https://github.com/microscopic-image-analysis/particle_tomography_results.git
cd particle_tomography_results
```

## 2. Set up the environment
We provide a conda environment specification to ensure all dependencies are installed correctly.
```bash
conda env create -f environment.yml
conda activate particle-tomography-results
```
## 3. Install Particle Tomography
Follow the instructions given [here](https://github.com/microscopic-image-analysis/particle-tomography) to install the Particle Tomography package.

## 4. Run the scripts
From the repository root, run:
```bash
python scripts/main.py
```
This will run all benchmarks on the protein dataset and generate plots. Internally, it executes
```python
benchmark.run_all_benchmarks()
plot.plot_protein(show_3d_volumes=False)
```
## 5. Notes
 * We only provide the protein dataset in the data folder. The vesicle and thinfilm dataset from [“Accurate real space iterative reconstruction (RESIRE) algorithm for tomography”](https://www.nature.com/articles/s41598-023-31124-7) can be downloaded from [here](https://zenodo.org/records/7819857), and the platinum dataset from ["Nanomaterial datasets to advance tomography in scanning transmission electron microscopy"](https://www.nature.com/articles/sdata201641) from [here](https://springernature.figshare.com/collections/Nanomaterial_datasets_to_advance_tomography_in_scanning_transmission_electron_microscopy/2185342). In benchmark.py the parts corresponding to the other datasets have been commented out.
 * Resire reconstructions were obtained with the scripts in scripts/reconstruct/resire_matlab_scripts. To obtain plots for these, the reconstructed volumes must be placed in out/RESIRE/name_of_dataset.
