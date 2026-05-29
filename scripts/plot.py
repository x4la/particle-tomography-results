import numpy as np
import scipy
import h5py
from data.loader import load_vesicle_data, load_protein_data
from pathlib import Path
import os
import pyvista as pv
import torch

import matplotlib.pyplot as plt
from particle_tomography import ParticleTomographyModel

from scripts import utils
from scripts.utils import angular_errors_deg

ROOT = Path(__file__).resolve().parent.parent

STANDARD_ALGORITHMS = ("particle_tomography", "RESIRE", "SIRT", "FBP")
ALGORITHM_LABELS = {
    "particle_tomography": "Particle Tomography",
    "RESIRE": "RESIRE",
    "SIRT": "SIRT",
    "FBP": "FBP",
}


def load_standard_reconstructions(dataset_name, transpose_mat=True):
    volumes = []
    labels = []
    for algo in STANDARD_ALGORITHMS:
        for ext in (".npz", ".mat"):
            path = ROOT / "out" / algo / dataset_name / f"{algo}_volume_reconstruction{ext}"
            if not path.exists():
                continue
            if ext == ".npz":
                data = np.load(path)
                volume = data[data.files[0]]
            else:
                data = scipy.io.loadmat(path)
                keys = [k for k in data if not k.startswith("__")]
                volume = data[keys[0]]
                if transpose_mat:
                    volume = volume.transpose(1, 0, 2).copy()
            volumes.append(volume)
            labels.append(ALGORITHM_LABELS[algo])
            break
    if not volumes:
        raise FileNotFoundError(f"No reconstruction outputs found for {dataset_name!r} under {ROOT / 'out'}")
    return volumes, labels


def plot_rotation_refinement_pdf(true_rots,
                                 noisy_rots,
                                 refined_rots_list,
                                 degrees=True,
                                 bins=25,
                                 xlim=(0, 5),
                                 ax=None,
                                 legend_fontsize=12,
                                 labels=("Initial noisy rotations", "Refined rotations"),
                                 colors=("tab:red", "tab:blue"),
                                 path=None):
    """
    Compare histograms (PDFs) of angular errors for initial noisy and refined rotations.
    Averages histograms across multiple runs.
    """

    # compute noisy rotation errors once
    err_noisy = angular_errors_deg(true_rots, noisy_rots, degrees=degrees)

    # compute bin edges and centers
    bins_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

    # compute histograms for each run
    hist_refined_runs = []
    for refined_rots in refined_rots_list:
        err_refined = angular_errors_deg(true_rots, refined_rots, degrees=degrees)
        hist_refined, _ = np.histogram(err_refined, bins=bins_edges, density=False)
        hist_refined_runs.append(hist_refined)

    # average across runs
    hist_refined_mean = np.mean(hist_refined_runs, axis=0)

    # histogram for noisy (single)
    hist_noisy, _ = np.histogram(err_noisy, bins=bins_edges, density=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))

    width = (xlim[1] - xlim[0]) / bins * 0.8

    # --- plot noisy histogram
    ax.bar(bin_centers, hist_noisy, width=width, color=colors[0], alpha=0.4, label=labels[0])

    # --- plot refined mean + std
    ax.bar(bin_centers, hist_refined_mean, width=width, color=colors[1], alpha=0.4, label=labels[1])

    ax.set_xlabel("Rotational error (°)" if degrees else "Angular deviation (rad)")
    ax.set_xlim(*xlim)
    ax.legend(frameon=False, fontsize=legend_fontsize)
    ax.set_ylabel(" ")  # empty y-label
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_vesicle(show_3d_volumes=False):
    # Input data paths
    input_paths = {
        "projection_file": ROOT / "data" / "vesicle" / "projections_vesicle.mat",
        "rotations_file": ROOT / "data" / "vesicle" / "projections_vesicle_euler_angles.mat",
        "true_volume_file": ROOT / "data" / "vesicle" / "vesicle.mrc"
    }

    _, _, _, true_volume = load_vesicle_data(input_paths, return_angles=False)

    volumes, algorithms = load_standard_reconstructions("vesicle")
    make_and_save_plots("vesicle", volumes, algorithms, true_volume, ROOT / "plots" / "vesicle", 10)

    if show_3d_volumes:
        for n, vol in enumerate(volumes):
            plot_3dvoxel(vol)


def plot_protein(show_3d_volumes=False):
    # Input data paths
    input_paths = {
        "projection_file": ROOT / "data" / "protein" / "projections_6b3r_wedge40_step-2.0_pixel-132_blur-0.6299118258855213.npz",
        "rotations_file": ROOT / "data" / "protein" / "projections_6b3r_wedge40_step-2.0_euler_angles.txt",
        "true_volume_file": ROOT / "data" / "protein" / "6b3r_pixel-132_blur-0.6299118258855213.npz"
    }

    _, _, true_volume = load_protein_data(input_paths, return_angles=False)

    volumes, algorithms = load_standard_reconstructions("protein")
    make_and_save_plots("protein", volumes, algorithms, true_volume, ROOT / "plots" / "protein",
                        slice_thickness=10)

    if show_3d_volumes:
        for n, vol in enumerate(volumes):
            plot_3dvoxel(vol)


def plot_thinfilm(show_3d_volumes=False):
    volumes, algorithms = load_standard_reconstructions("thinfilm", transpose_mat=False)
    make_and_save_plots("thinfilm", volumes, algorithms, true_volume=None, outdir=ROOT / "plots" / "thinfilm", slice_thickness=5)

    if show_3d_volumes:
        for n, vol in enumerate(volumes):
            plot_3dvoxel(vol)


def plot_platinum(show_3d_volumes=False):
    volumes = {}

    data = np.load(ROOT / "out" / "particle_tomography" / "platinum" / "particle_tomography_volume_reconstruction.npz")
    volumes["particle_tomography"] = data[data.files[0]]

    mat_path = Path(ROOT) / "out" / "RESIRE" / "platinum" / "RESIRE_volume_reconstruction.mat"
    with h5py.File(mat_path, "r") as f:
        vol = np.array(f["rec"])  # or whatever the dataset is called
    print("Volume shape:", vol.shape)
    volumes["RESIRE"] = vol

    path = ROOT / "out" / "FBP" / "platinum" / "FBP_volume_reconstruction.npz"
    data = np.load(path)
    volumes["FBP"] = data[data.files[0]]


    volumes = list(volumes.values())
    algorithms = ["Particle Tomography", "RESIRE", "FBP"]
    make_and_save_plots("platinum", volumes, algorithms, true_volume=None, outdir=ROOT / "plots" / "platinum",
                        clip_color_map=True, slice_thickness=30)

    if show_3d_volumes:
        for n, vol in enumerate(volumes):
            vol_section = vol[200:800,300:700,200:800]
            vol_section[vol_section < 0] = 0
            plot_3dvoxel(vol_section)


def save_and_plot_fsc_multiple(frequencies_list, correlations_list, algo_names,
                               legend_fontsize=14,
                               y_lim=(0.4, 1),
                               figsize=(8, 4), filename=None, colors=None):
    """
    Plot multiple FSC curves with optional saving.
    """
    if colors is None:
        # Default to Matplotlib’s color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=figsize)

    # Plot each FSC curve
    for freqs, cors, name, color in zip(frequencies_list, correlations_list, algo_names, colors):
        plt.plot(freqs * 100, cors, linewidth=2, label=name, color=color)

    plt.xlabel('Spatial Frequency [% of Nyquist]', fontsize=14)
    plt.ylabel('Fourier Shell Correlation', fontsize=14)
    # plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=legend_fontsize)
    plt.xlim(0, 100)
    plt.ylim(y_lim[0], y_lim[1])

    plt.tight_layout()
    # Print last FSC values for each curve
    for name, cors in zip(algo_names, correlations_list):
        print(f"FSC at Nyquist for {name}: {cors[-1]:.3f}")

    if filename:
        plt.savefig(filename / "fsc_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()


def make_and_save_plots(experiment_name, volumes, algorithm_names, true_volume, outdir, slice_thickness, clip_color_map=False):
    if len(volumes) != len(algorithm_names):
        raise ValueError("volumes and algorithm_names must have the same length")

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Make and save Fourier Shell Correlation plot (only if true_volume exists)
    if true_volume is not None:
        freqs, corrs = utils.fsc(volumes, true_volume, n_bins=20)
        save_and_plot_fsc_multiple(freqs, corrs, algorithm_names, filename=outdir)

    # Make projections for all volumes and ground truth (if available)
    if true_volume is not None:
        all_volumes = [true_volume] + volumes
        all_names = ["Ground Truth"] + algorithm_names
    else:
        all_volumes = volumes
        all_names = algorithm_names

    print("Generating projections...")
    all_projections = []
    all_slices = []

    for volume in all_volumes:
        proj_data = make_projections(volume, slice_thickness=slice_thickness)
        projections = proj_data["projection"]["reconstruction"]
        slices = proj_data["slice"]["reconstruction"]
        all_projections.append(projections)
        all_slices.append(slices)



    # Save individual projection and slice images
    print("Saving individual images...")
    for i, name in enumerate(all_names):
        save_name = name.lower().replace(" ", "_")

        # Save projections
        for plane in ["xy", "xz", "yz"]:
            save_image(outdir / f"{experiment_name}_projection_{save_name}_{plane}", all_projections[i][plane])
            save_image(outdir / f"{experiment_name}_slice_{save_name}_{plane}", all_slices[i][plane])

    # Create comparison plots
    print("Creating comparison plots...")
    title_suffix = f" - {experiment_name.capitalize()}"
    create_comparison_plot(all_projections, all_names,filename=outdir / f"{experiment_name}_projections_comparison.pdf",
                           clip_color_map=clip_color_map)
    create_comparison_plot(all_slices, all_names,filename=outdir / f"{experiment_name}_slices_comparison.pdf",
                           clip_color_map=clip_color_map)

    print(f"All plots saved in {outdir}")


def create_comparison_plot(data_list, col_names, filename, cmap="viridis", clip_color_map=False, dpi=300):
    """
    Create a comparison plot with rows for xy, xz, yz planes
    and columns for each algorithm/ground truth.
    """
    planes = ["xy", "xz", "yz"]
    n_rows = len(planes)
    n_cols = len(data_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 0.5, 10), constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, plane in enumerate(planes):
        for j, (data, name) in enumerate(zip(data_list, col_names)):
            ax = axes[i, j]

            # --- PLOT WITH CONTRAST CONTROL ---
            if clip_color_map:
                vmax = np.percentile(data[plane], 99.5)
            else:
                vmax = 1.0

            img = ax.imshow(data[plane], cmap=cmap, vmin=0, vmax=vmax)

            if i == 0:
                ax.set_title(name, fontsize=18, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"{plane.upper()}-plane", fontsize=18, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()


def make_projections(reconstruction, slice_thickness=10, true_volume=None):
    proj_data = {
        "projection": {"reconstruction": project(reconstruction)},
        "slice": {"reconstruction": project(reconstruction, slice_thickness)}
    }
    if true_volume is not None:
        proj_data["projection"]["ground_truth"] = project(true_volume)
        proj_data["slice"]["ground_truth"] = project(true_volume, slice_thickness)
    return proj_data


def project(img3d, slice_thickness=None):
    projections = {"xy": 0, "xz": 1, "yz": 2}  # axis indices for numpy

    def do_project(axis):
        if slice_thickness is None:
            return np.sum(img3d, axis=axis)
        else:
            center = img3d.shape[axis] // 2
            lo = max(0, center - slice_thickness // 2)
            hi = min(img3d.shape[axis], center + slice_thickness // 2)
            slices = [slice(None)] * img3d.ndim
            slices[axis] = slice(lo, hi)
            return np.sum(img3d[tuple(slices)], axis=axis)

    result = {}
    for name, axis in projections.items():
        proj = do_project(axis)
        # Scale to 0-1
        proj = (proj - proj.min()) / (proj.max() - proj.min())
        proj = np.rot90(proj, k=1)
        result[name] = np.clip(proj, 0, 1)
    return result


def save_image(name, value, cmap='viridis', dpi=300):
    if isinstance(value, dict):
        for k, v in value.items():
            save_image(f"{name}_{k}", v, cmap=cmap)
    else:
        # Normalize
        value = (value - value.min()) / (value.max() - value.min())

        # Save using Matplotlib with a colormap
        plt.imsave(f"{name}.pdf", value, cmap=cmap, dpi=dpi)



def plot_3dvoxel(vol, cmap="viridis"):
    """
    Visualize a 3D volume (Z, Y, X) using PyVista.
    """
    if pv is None:
        raise RuntimeError("pyvista not installed. Use `pip install particle-tomography[plotting]`")
    # Convert to NumPy array if input is a torch tensor
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()


    # Create a PyVista uniform grid with voxel dimensions
    nz, ny, nx = vol.shape
    grid = pv.ImageData(dimensions=(nx+1, ny+1, nz+1))
    grid["values"] = vol.flatten(order="C")

    # Plot the volume
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="values", cmap=cmap, opacity="linear")
    plotter.show_axes()
    plotter.show_grid()
    plotter.show()


def plot_multiple_K(K_values, fsc_values):
    # K_values = [1000, 2000, 3000, 4000, 5000, 10000, 25000, 100000]
    # fsc values obtained from averaging 10 times
    # fsc_values = [0.4091416746377945, 0.6570131540298462, 0.7742855310440063, 0.8343972563743591, 0.8595863342285156,
      #           0.864678293466568, 0.8675709426403045, 0.8715670645236969]
    ROOT = Path(__file__).resolve().parent.parent
    path = ROOT / "plots" / "vesicle" / "FSC_vs_K.pdf"
    plt.figure(figsize=(6, 4))
    plt.plot(K_values, fsc_values, marker='o', lw=2)

    plt.xscale('log')
    plt.xlabel("Number of Gaussian Particles (K)")
    plt.ylabel("FSC at Nyquist")
    plt.grid(True, which='both', ls='--', alpha=0.5)

    # highlight K=5000
    plt.axvline(5000, color='gray', ls='--', lw=1.5)
    plt.text(5000 * 1.1, fsc_values[1] - 0.01, "K=5000", color='gray')

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

def plot_points_pyvista_colored(X, side=(1.0, 1.0, 1.0), weights=None, radius=0.01, scale=1.0, do_shift=False, cmap="viridis",
                                camera_position=None, save_path=None, window_size=[2000, 2000]):
    """
    Plot 3D points in PyVista as spheres, colored by weights with a colormap.
    Adds "fake shadows" to simulate depth without requiring OpenGL shadows.
    Args:
        X (torch.Tensor or np.ndarray): (N, 3) point coordinates.
        side (tuple): (x_range, y_range, z_range) - half-ranges for axes (e.g., (1.0, 1.0, 1.0) gives -1 to 1).
        weights (torch.Tensor or np.ndarray, optional): (N,) scalar weights for color mapping.
        radius (float): Sphere radius in data units.
        cmap (str): Colormap name (e.g., 'viridis', 'plasma', 'coolwarm').
        camera_position (tuple or str): Optional PyVista camera setting.
        save_path (str): If given, saves a screenshot instead of showing interactively.
    """
    # optional shifting and scaling
    if do_shift:
        shift = side
    else:
        shift = (0, 0, 0)
    X = (X + np.array(shift))*scale
    side = ((side[0] + shift[0])*scale, (side[1] + shift[1])*scale, (side[2] + shift[2])*scale)
    radius = radius * scale
    (pos, focal_point, view_up) = camera_position
    pos = ((pos[0]+shift[0])*scale, (pos[1]+shift[1])*scale, (pos[2]+shift[2])*scale)
    focal_point = (shift[0]*scale, shift[1]*scale, shift[2]*scale)
    camera_position = (pos, focal_point, view_up)


    # Convert inputs
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if weights is not None and isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    # Create a PolyData point cloud
    cloud = pv.PolyData(X)
    if weights is not None:
        cloud["weights"] = weights

    # Sphere glyphs for points
    sphere = pv.Sphere(radius=radius)
    glyphs = cloud.glyph(scale=False, geom=sphere)

    # ------------------
    # Create fake shadows
    # ------------------
    shadow_points = X.copy()
    # Flatten Z to place shadows on a "floor" slightly below min Z
    min_z = np.min(X[:, 2]) - radius * 0.5
    shadow_points[:, 2] = min_z
    shadow_cloud = pv.PolyData(shadow_points)
    shadow_glyphs = shadow_cloud.glyph(scale=False, geom=pv.Sphere(radius=radius))
    shadow_color = (0.0, 0.0, 0.0)
    shadow_opacity = 0.1

    # Create plotter
    if save_path is not None:
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
    else:
        plotter = pv.Plotter()
    plotter.show_axes()
    plotter.set_background("white")

    # Plot shadows
    plotter.add_mesh(shadow_glyphs, color=shadow_color, opacity=shadow_opacity)

    # Plot actual points
    if weights is not None:
        vmax = np.percentile(weights, 99.9)  # clip top 99.95% for color mapping
        plotter.add_mesh(
            glyphs,
            cmap="viridis",
            show_scalar_bar=False,
            clim=[0, vmax],  # sets color limits
            lighting=True,
            smooth_shading=True,
            specular=0.8,
            specular_power=15,
            ambient=0.2,
            diffuse=0.7,
        )
    else:
        plotter.add_mesh(glyphs, color="darkblue", lighting=True, smooth_shading=True,
                         specular=0.8, specular_power=15, diffuse=0.7)

    # Set bounds
    if do_shift:
        plotter.show_bounds(
            bounds=(0, side[0], 0, side[1], 0, side[2]),
            grid='back',
            location='outer'
        )
    else:
        plotter.show_bounds(
            bounds=(-side[0], side[0], -side[1], side[1], -side[2], side[2]),
            grid='back',
            location='outer'
        )

    # Camera
    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        plotter.view_isometric()

    # Show or save
    if save_path is not None:
        plotter.show()  # This renders the scene off-screen
        plotter.screenshot(save_path, transparent_background=False)
        print(f"Saved colored point cloud to {save_path}")
    else:
        plotter.show()



def plot_protein_3D():
    path = ROOT / "out" / "particle_tomography" / "protein" / "model.pt"
    model = ParticleTomographyModel.from_saved_state(path)
    points, weights, bandwidth = model.get_volume_sparse()
    bounding_box=(1.0, 1.0, 1.0)
    mask = (
            (points[:, 0] >= -bounding_box[0]) & (points[:, 0] <= bounding_box[0]) &
            (points[:, 1] >= -bounding_box[1]) & (points[:, 1] <= bounding_box[1]) &
            (points[:, 2] >= -bounding_box[2]) & (points[:, 2] <= bounding_box[2])
    )
    points = points[mask]

    if weights is not None:
        weights = weights[mask]
        # Mask out the lowest 10% of weights
        threshold = np.percentile(weights, 15)  # percentile
        high_mask = weights > threshold
        points = points[high_mask]
        weights = weights[high_mask]

    coords = np.array([0.55, 0.5, 1.2])*0.85
    pos = tuple(coords*3) # camera position in 3D space
    focal_point = (0, 0, 0)  # what the camera looks at
    view_up = (0, 0, 1)  # which direction is 'up'
    camera = (pos, focal_point, view_up)

    plot_points_pyvista_colored(points, side=bounding_box, save_path=ROOT / "plots" / "protein_points.png",
                      camera_position=camera, weights=weights)
    model.plot_volume(camera=camera, path=ROOT / "plots" / "protein_volume.png")
    # model.plot_weights()


def plot_thinfilm_3D():
    path = ROOT / "out" / "particle_tomography" / "thinfilm" / "model.pt"
    model = ParticleTomographyModel.from_saved_state(path)
    points, weights, bandwidth = model.get_volume_sparse()

    coords = np.array([0.55, 0.2, 1.1]) * 2.8
    pos = tuple(coords)  # camera position in 3D space
    focal_point = (0, 0, 0)  # what the camera looks at
    view_up = (0, 0, 1)  # which direction is 'up'
    camera = (pos, focal_point, view_up)
    bounding_box = (0.6, 0.85, 0.25)

    mask = (
            (points[:, 0] >= -bounding_box[0]) & (points[:, 0] <= bounding_box[0]) &
            (points[:, 1] >= -bounding_box[1]) & (points[:, 1] <= bounding_box[1]) &
            (points[:, 2] >= -bounding_box[2]) & (points[:, 2] <= bounding_box[2])
    )
    points = points[mask]

    if weights is not None:
        weights = weights[mask]
        # Mask out the lowest 10% of weights
        threshold = np.percentile(weights, 15)  # percentile
        high_mask = weights > threshold
        points = points[high_mask]
        weights = weights[high_mask]

    plot_points_pyvista_colored(
        points,
        side=bounding_box,
        weights=weights,
        radius=0.011,
        scale=68.425,
        shift=True,
        camera_position=camera,
        save_path=ROOT / "plots" / "thinfilm_points.png",
        window_size=[2000, 1500]
    )
if __name__ == "__main__":
    # plot_protein_3D()
    # plot_thinfilm_3D()
    plot_platinum()
