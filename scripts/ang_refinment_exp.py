from pathlib import Path
import time
import torch
import sys
import pathlib
import numpy as np

from particle_tomography import reconstruct
from particle_tomography.utils import quat_to_matrix
from scipy.stats import vonmises_fisher

from scripts import utils
from scripts.plot import plot_rotation_refinement_pdf, save_and_plot_fsc_multiple
from data.loader import load_vesicle_noisy_rotations, load_vesicle_data
from scipy.spatial.transform import Rotation as R

from scripts.utils import show_images

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATHS_VESICLE = {
    "projection_file": ROOT / "data" / "vesicle" / "projections_vesicle.mat",
    "rotations_file": ROOT / "data" / "vesicle" / "projections_vesicle_euler_angles.mat",
    "true_volume_file": ROOT / "data" / "vesicle" / "vesicle.mrc"
}
NOISY_ROTATIONS_PATH = ROOT / "data" / "vesicle" / "noisy_rotations.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################################################################
######################## create noisy rotations #######################
#######################################################################


def add_rotation_noise(rotation_matrices, kappa=10000):
    """
    Add noise to a stack of 3x3 rotation matrices using von Mises-Fisher distribution.
    """
    np.random.seed(0)
    # Convert rotation matrices to quaternions
    rotations = R.from_matrix(rotation_matrices)
    quaternions = rotations.as_quat()  # Returns (x, y, z, w) format

    noisy_quaternions = np.zeros_like(quaternions)

    # Sample from von Mises-Fisher around each quaternion mean
    for i, q in enumerate(quaternions):
        # Sample from von Mises-Fisher with mean direction q and concentration kappa
        vmf = vonmises_fisher(mu=q, kappa=kappa)
        noisy_q = vmf.rvs(1)

        # Ensure unit norm (should already be, but doesnt hurt)
        noisy_quaternions[i] = noisy_q / np.linalg.norm(noisy_q)

    # Convert back to rotation matrices
    noisy_rotations = R.from_quat(noisy_quaternions)
    noisy_matrices = noisy_rotations.as_matrix()

    return noisy_matrices


#######################################################################
####################### run refinement experiment #####################
#######################################################################


def run_refinement_exp(num_repetitions=10):
    images, true_rotations, shifts, true_volume = load_vesicle_data(INPUT_PATHS_VESICLE)
    noisy_rotations = np.load(NOISY_ROTATIONS_PATH)["arr_0"]

    # # no tilt series but random orientations exp:
    # data = np.load("../data/vesicle/rand_anchored_40_64x64.npz")
    # images = data["projections"]
    # shifts = None
    # quats = torch.from_numpy(data["quats"])
    # true_rotations = quat_to_matrix(quats)
    # noisy_rotations = add_rotation_noise(true_rotations, kappa=1000)
    # true_volume = np.load("../data/vesicle/volume_test.npz")["arr_0"]

    all_corrs = []
    all_refined_rots = []
    for i in range(num_repetitions):
        model = reconstruct(images,
                            noisy_rotations,
                            shifts=shifts,
                            num_points=5000,
                            total_iterations=2000,
                            num_rejuvenates=1,
                            kernel_size=3,
                            geom_start_fraction=0.8,
                            device=DEVICE,
                            )
        vol_with_refinement = model.get_volume()
        refined_rots = model.rotations.cpu().detach().numpy()
        all_refined_rots.append(refined_rots)
        model = reconstruct(images,
                            noisy_rotations,
                            shifts=shifts,
                            num_points=5000,
                            total_iterations=2000,
                            num_rejuvenates=1,
                            kernel_size=3,
                            geom_start_fraction=1.0,
                            device=DEVICE,
                            )
        vol_without_refinement = model.get_volume()
        model = reconstruct(images,
                            true_rotations,
                            shifts=shifts,
                            num_points=5000,
                            total_iterations=2000,
                            num_rejuvenates=1,
                            kernel_size=3,
                            geom_start_fraction=1.0,
                            device=DEVICE,
                            )
        vol_with_correct_rotations = model.get_volume()
        volumes = [vol_with_correct_rotations, vol_with_refinement, vol_without_refinement]
        freqs, corrs = utils.fsc(volumes, true_volume, n_bins=20)
        all_corrs.append(corrs)
    all_corrs = np.array(all_corrs)
    mean_corrs = np.mean(all_corrs, axis=0)
    return freqs, all_corrs, mean_corrs, true_rotations, noisy_rotations, all_refined_rots


def main():
    # ### create and save noisy rotaitons
    # _, true_rotations, _, _ = load_vesicle_data(INPUT_PATHS_VESICLE)
    # noisy_rotations = add_rotation_noise(true_rotations, kappa=10000)
    # np.savez(NOISY_ROTATIONS_PATH, noisy_rotations)
    # #
    # ### run experiment
    # freqs, all_corrs, mean_corrs, true_rotations, noisy_rotations, all_refined_rots = run_refinement_exp()
    #
    # ### save results
    # save_path = ROOT / "out" / "refinement_exp.npz"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # np.savez_compressed(
    #     save_path,
    #     freqs=freqs,
    #     all_corrs=np.array(all_corrs),
    #     mean_corrs=mean_corrs,
    #     all_refined_rots=np.array(all_refined_rots, dtype=object),  # list of arrays
    #     true_rotations=true_rotations,
    #     noisy_rotations=noisy_rotations,
    # )
    #
    # print(f"Saved results to {save_path}")

    ### load saved results
    load_path = ROOT / "out" / "refinement_exp.npz"
    data = np.load(load_path, allow_pickle=True)
    freqs = data["freqs"]
    all_corrs = data["all_corrs"]
    mean_corrs = data["mean_corrs"]
    all_refined_rots = list(data["all_refined_rots"])  # convert back to list
    true_rotations = data["true_rotations"]
    noisy_rotations = np.load(NOISY_ROTATIONS_PATH)["arr_0"]


    ### plot results
    algorithm_names = ["Baseline", "Reconstruction with angular refinement", "Reconstruction without angular refinement"]
    colors = ["tab:green", "tab:red", "tab:blue"]
    plot_rotation_refinement_pdf(true_rotations, noisy_rotations, all_refined_rots, path=ROOT / "plots" / "histogram.pdf")
    save_and_plot_fsc_multiple(freqs, mean_corrs, algorithm_names, filename=ROOT / "plots", y_lim=(0.6, 1.0), colors=colors)


if __name__ == "__main__":
    main()
