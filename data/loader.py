import h5py
import numpy as np
import scipy
import mrcfile
import pandas as pd
import tifffile
from scipy.spatial.transform import Rotation as R
import scipy.io as sio

def load_vesicle_data(paths: dict, return_angles: bool = False):
    """Load vesicle projection images, rotations, shifts, and true volume.
    """
    # Load projection images
    images = scipy.io.loadmat(paths["projection_file"])["projections_FST"].astype(np.float32)
    images = np.transpose(images, (2, 1, 0)).copy()  # (N, H, W)
    print(f"File contains {images.shape[0]} projection images of size {images.shape[1]}x{images.shape[2]} pixels")

    # Load true volume
    with mrcfile.open(paths["true_volume_file"], permissive=True) as mrc:
        true_vol = mrc.data.astype(np.float32).transpose(2, 1, 0)  # (Z, Y, X)

    # Load rotation matrices
    angles = scipy.io.loadmat(paths["rotations_file"])["angles"]
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix().astype(np.float32)  # (N, 3, 3)

    # Compute pixel shifts
    shifts = compute_halfpixel_shifts(rotations, grid_size=images.shape[-1]).astype(np.float32)  # (N, 2)

    if return_angles:
        return images, rotations, shifts, true_vol, angles
    else:
        return images, rotations, shifts, true_vol


def load_vesicle_noisy_rotations(path, return_angles=False):
    with h5py.File(path, 'r') as f:
        noisy_angles = np.array(f['angles'], dtype=np.float32).transpose(1,0)
        noisy_rotations = R.from_euler("zyx", noisy_angles, degrees=True).inv().as_matrix() # (N, 3, 3)

    if return_angles:
        return  noisy_rotations, noisy_angles
    else:
        return noisy_rotations


def load_protein_data(paths: dict, return_angles: bool = False):
    """Load protein tomography data."""
    images = np.load(paths["projection_file"]).transpose(2, 1, 0).copy()
    angles = np.loadtxt(paths["rotations_file"], dtype=np.float32)
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix()
    true_volume = np.load(paths["true_volume_file"]).transpose(2, 1, 0).copy()

    if return_angles:
        return images, rotations, true_volume, angles
    else:
        return images, rotations, true_volume


def load_thinfilm_data(paths: dict, return_angles: bool = False):
    """Load thin-film tomography data."""
    mat = scipy.io.loadmat(paths["projection_file"])
    images = mat['proj_crop'].astype(np.float32).transpose(2, 1, 0).copy()  # (N, H, W)

    mat = scipy.io.loadmat(paths["rotations_file"])
    angles = np.array(mat['final_ang'], dtype=np.float32)
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix().astype(np.float32)

    if return_angles:
        return images, rotations, angles
    else:
        return images, rotations


def load_platinum_data(paths: dict, max_image=140, spacing=1, return_angles=False):
    images = tifffile.imread(paths["projection_file"]).transpose(0,2,1).copy() # so that rotation is around the y-axis
    angles = np.loadtxt(paths["rotations_file"], dtype=np.float32)
    rotations = R.from_euler("xyz", angles, degrees=True).inv().as_matrix()

    idx = slice(0, max_image, spacing)

    if return_angles:
        return images[idx], rotations[idx], angles[idx]
    else:
        return images[idx], rotations[idx]

def save_for_matlab(paths: dict, max_image=140, spacing=1):
    # load projections
    images = tifffile.imread(paths["projection_file"])
    angles = np.loadtxt(paths["rotations_file"], dtype=np.float32)

    # build rotation matrices like in your MATLAB script
    rotations = R.from_euler("xyz", angles, degrees=True).inv().as_matrix()

    idx = slice(0, max_image, spacing)
    images_sel = images[idx]
    angles_sel = angles[idx]
    rotations_sel = rotations[idx]

    # save in MATLAB-friendly format
    sio.savemat("platinum_projections.mat", {"projections": images_sel.astype(np.float32)})
    sio.savemat("platinum_angles.mat", {"angles": angles_sel.astype(np.float32)})

    print("Saved proj_crop3.mat, final_ang.mat, rot_mats.mat")


def load_Pd2_data(paths: dict):
    """Load Pd2 tomography data."""
    images = np.array(scipy.io.loadmat(paths["projection_file"])["proj"]).transpose(2, 1, 0).copy()
    angles = np.squeeze(scipy.io.loadmat(paths["rotations_file"])["ang"])
    vol = np.array(scipy.io.loadmat(paths["true_volume_file"])["final_Rec"])
    rotations = R.from_euler("zyx", angles, degrees=True).inv().as_matrix()
    return images, rotations, vol

def compute_halfpixel_shifts(rotations: np.ndarray, grid_size: int = 64) -> np.ndarray:
    pixel_size = 2.0 / grid_size
    t = np.array([-0.5 * pixel_size, -0.5 * pixel_size, -0.5 * pixel_size])  # shape [3]
    t_expanded = np.tile(t, (rotations.shape[0], 1))[:, :, None]  # [N, 3, 1]

    rotated_t = np.matmul(rotations, t_expanded).squeeze(-1)  # [N, 3]
    delta = rotated_t - t  # [N, 3]
    return delta[:, :2]  # [N, 2]
