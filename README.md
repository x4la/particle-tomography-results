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
