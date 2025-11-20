import numpy as np
import astra

from pathlib import Path
from data.loader import load_vesicle_data
from scripts import plot

def reconstruct_fbp(images, angles, shift_half_pixel=False, algorithm="FBP_CUDA"):
    """
    Reconstruct a 3D volume from a set of 2D projection images using ASTRA's FBP.
    Projections were performed around the y-axis. ASTRA implements FBP only for 2D
    reconstructions, so we need to apply it for every slice along the y-axis.

    Args:
        images: (N, Y, X) - projection images where N is number of projections
        angles: rotation angles in degrees around y axis (length N)
        algorithm: reconstruction algorithm ("FBP_CUDA", "FBP", "SIRT_CUDA", etc.)

    Returns:
        3D volume (Z, Y, X) where Z corresponds to the reconstructed dimension
    """
    if shift_half_pixel:
        print("shifting by half a pixel!")
    N, Y, X = images.shape

    # Convert angles to radians
    angles_rad = np.deg2rad(angles)

    # Initialize output volume - Z dimension will be same as X (reconstruction in XZ plane)
    volume = np.zeros((X, Y, X), dtype=np.float32)

    print(f"Reconstructing {Y} slices using {algorithm}...")

    # Process each Y slice
    for y_idx in range(Y):

        # Extract current slice across all projections: (N, X)
        slice_projections = images[:, y_idx, :].astype(np.float32)

        # Create ASTRA geometries
        # Projection geometry - parallel beam with rotation around y-axis
        proj_geom = astra.create_proj_geom('parallel', 1.0, X, angles_rad)

        # Volume geometry - 2D reconstruction in XZ plane
        if shift_half_pixel:
            vol_geom = astra.create_vol_geom(X, X, -X/2 + 0.5, X/2 + 0.5, -X/2 + 0.5, X/2 + 0.5)
        else:
            vol_geom = astra.create_vol_geom(X, X)

        # Create ASTRA data objects
        proj_id = astra.data2d.create('-sino', proj_geom, slice_projections)
        rec_id = astra.data2d.create('-vol', vol_geom)

        # Configure reconstruction algorithm
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id

        # Additional parameters for iterative algorithms
        if 'SIRT' in algorithm or 'CGLS' in algorithm:
            cfg['option'] = {'MinConstraint': 0}  # Non-negativity constraint

        # Create and run algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Retrieve reconstruction
        reconstruction = astra.data2d.get(rec_id)
        volume[:, y_idx, :] = reconstruction

        # Clean up ASTRA objects
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([proj_id, rec_id])

    print("FBP reconstruction completed! Final shape:", volume.shape)

    # Return with proper axis ordering (Z, Y, X)
    return volume


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
    reconstructed = reconstruct_fbp(images, angles_deg[:,1], shift_halpf_pixel=True)
    freq, corr = utils.fsc(reconstructed, true_volume, 20)
    utils.plot_fsc(freq, corr)
    utils.plot_3dvoxel(reconstructed)


    # plot.save_output(None, None, None, reconstructed, true_volume=None, logging_prefix="FBP",
    #                  outdir=ROOT / "test" / "FBP" / "vesicle", slice_thickness=10)