import math
from pathlib import Path

import numpy as np
import astra
from scipy.ndimage import shift as scipy_shift
from data.loader import load_vesicle_data, load_protein_data, load_thinfilm_data, load_Pd2_data


def reconstruct_sirt(images, angles, shift_half_pixel=False, algorithm="SIRT3D_CUDA", iterations=200,  **kwargs):
    """
    Reconstruct a 3D volume from a set of 2D projection images using ASTRA's 3D SIRT.

    Args:
        images: (N, Y, X) - projection images where N is number of projections
        angles: rotation angles in degrees around y axis (length N)
        algorithm: reconstruction algorithm ("SIRT3D_CUDA")
        iterations: number of iterations for iterative algorithms
        **kwargs: additional options for the algorithm

    Returns:
        3D volume (Z, Y, X) where Z corresponds to the reconstructed dimension
    """
    if shift_half_pixel:
        print("Shifting by half a pixel!")
    N, Y, X = images.shape

    # Convert angles to radians
    angles_rad = np.deg2rad(angles)

    print(f"Input data shape: {images.shape}")
    print(f"Number of angles: {len(angles)}")

    proj_geom = astra.create_proj_geom('parallel3d',
                                        1.0, 1.0,       # det_spacing_x, det_spacing_y,
                                       Y, X,                # detector rows, detector columns
                                       angles_rad)

    # Create 3D volume geometry
    # Volume dimensions: (Z, Y, X) where Z is the reconstructed dimension
    if shift_half_pixel:
        vol_geom = astra.create_vol_geom(X, X, Y,
                                         -X / 2 + 0.5, X / 2 + 0.5,  # X bounds (shifted by 0.5)
                                         -X / 2 + 0.5, X / 2 + 0.5,  # Z bounds (shifted by 0.5)
                                         -Y / 2, Y / 2)  # Y bounds (no shift needed)
    else:
        vol_geom = astra.create_vol_geom(X, X, Y)  # represents (Y, Z, X) since we transpose at the end


    # ASTRA 3D expects projection data in shape: (det_row_count, angles, det_col_count)
    # Our input is (N, Y, X) = (angles, det_rows, det_cols)
    # We need to transpose to (Y, N, X) = (det_rows, angles, det_cols)
    projections_3d = np.transpose(images, (1, 0, 2)).astype(np.float32)

    # Create ASTRA data objects
    proj_id = astra.data3d.create('-sino', proj_geom, projections_3d)
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up algorithm configuration
    cfg = astra.astra_dict(algorithm)
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = rec_id

    # Set up options
    options = {
        'MinConstraint': 0.0,  # Non-negativity constraint
        'GPUindex': 0,
    }
    options.update(kwargs.get('option', {}))
    cfg['option'] = options

    print(f"Starting {algorithm} reconstruction with {iterations} iterations...")

    # Create and run algorithm
    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, iterations)

    # Retrieve final reconstruction
    reconstruction = astra.data3d.get(rec_id)

    # Clean up ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete([proj_id, rec_id])

    print("SIRT 3D reconstruction completed!")

    # transpose to desired format
    vol = reconstruction.astype(np.float32)
    vol = vol.transpose(1,0,2).copy()
    vol = np.flip(vol, 0).copy()
    print(f"Output volume shape: {vol.shape}")
    return vol

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import utils
    import plot

    ROOT = Path(__file__).resolve().parent.parent.parent
    INPUT_PATHS_VESICLE = {
        "projection_file": ROOT / "data" / "vesicle" / "projections_vesicle.mat",
        "rotations_file": ROOT / "data" / "vesicle" / "projections_vesicle_euler_angles.mat",
        "true_volume_file": ROOT / "data" / "vesicle" / "vesicle.mrc"
    }
    # vesicle:
    images, rotations, shifts, true_volume, angles_deg= load_vesicle_data(INPUT_PATHS_VESICLE, return_angles=True)
    reconstructed = reconstruct_sirt(images, angles_deg[:,1], shift_half_pixel=True)
    freq, corr = utils.fsc(reconstructed, true_volume, 20)
    utils.plot_fsc(freq, corr)
    utils.plot_3dvoxel(reconstructed)


    # protein:
    # config = build_protein_config()
    # images, rotations, _ = load_protein_data(config.input_paths)
    # voxel_sirt = astra_parallel_reconstruction(images, rotations, None, 'SIRT3D_CUDA', 500)
    # save_output(None, None, None, voxel_sirt, true_volume=None, logging_prefix="SIRT",
    #             outdir=Path("Results/SIRT" + "/protein"), slice_thickness=10)

    # # thin film:
    # config = build_thinfilm_config()
    # images, rotations = load_thinfilm_data(config.input_paths)
    # voxel_sirt = astra_parallel_reconstruction(images, rotations, None, 'SIRT3D_CUDA', 500)
    # save_output(None, None, None, voxel_sirt, true_volume=None, logging_prefix="SIRT",
    #             outdir=Path("Results/SIRT" + "/thinfilm"), slice_thickness=5)

    # Pd2:
    # config = build_Pd2_config()
    # images, rotations, _ = load_Pd2_data(config.input_paths)
    # voxel_sirt = astra_parallel_reconstruction(images, rotations, None, 'SIRT3D_CUDA', 500)
    # save_output(None, None, None, voxel_sirt, true_volume=None, logging_prefix="SIRT",
    #             outdir=Path("Results/SIRT" + "/Pd2"), slice_thickness=5)