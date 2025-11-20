import os
import pyvista as pv

import matplotlib.pyplot as plt


def plot_fsc(frequencies, correlations, title="Fourier Shell Correlation",
             threshold_lines=[0.5, 0.143], figsize=(8, 6), filename=None):
    """
    Plot FSC curve with optional saving.
    """
    plt.figure(figsize=figsize)

    # Plot FSC curve
    plt.plot(frequencies * 100, correlations, 'b-', linewidth=2, label='FSC')

    # Add threshold lines
    colors = ['red', 'orange', 'green']
    labels = ['0.5 threshold', '0.143 threshold (gold standard)', 'Custom threshold']

    for i, threshold in enumerate(threshold_lines):
        color = colors[i] if i < len(colors) else 'gray'
        label = labels[i] if i < len(labels) else f'{threshold} threshold'
        plt.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=label)

    plt.xlabel('Spatial Frequency (% of Nyquist)', fontsize=12)
    plt.ylabel('Fourier Shell Correlation', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(-0.2, 1.0)

    # Add resolution estimates at threshold crossings
    for threshold in threshold_lines:
        crossing_indices = np.where((correlations[:-1] >= threshold) & (correlations[1:] < threshold))[0]
        if len(crossing_indices) > 0:
            idx = crossing_indices[0]
            y1, y2 = correlations[idx], correlations[idx + 1]
            x1, x2 = frequencies[idx], frequencies[idx + 1]
            crossing_freq = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            plt.annotate(f'FSC={threshold:.3f}\n{crossing_freq * 100:.1f}% Nyquist',
                         xy=(crossing_freq * 100, threshold),
                         xytext=(crossing_freq * 100 + 10, threshold + 0.15),
                         arrowprops=dict(arrowstyle='->', color='black', alpha=0.6),
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    print(f"FSC at Nyquist: {correlations[len(correlations) - 1]:.3f}")
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_3dvoxel(vol, cmap="viridis"):
    """
    Visualize a 3D volume (Z, Y, X) using PyVista.

    Parameters:
        vol (torch.Tensor or np.ndarray): 3D volume data in (Z, Y, X) order.
        cmap (str): Colormap name for visualization.
    """
    # Convert to NumPy array if input is a torch tensor
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()


    # Create a PyVista uniform grid with voxel dimensions
    grid = pv.ImageData(dimensions=vol.shape)
    grid["values"] = vol.flatten(order="C")  # Fortran order to match PyVista's expectations

    # Plot the volume
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="values", cmap=cmap, opacity="linear")
    plotter.show_axes()
    plotter.show_grid()
    plotter.show()


def show_images(images, indices=None):
    """
    Display images from a batch or a single image.

    Args:
        images: Tensor/array of shape [B, H, W] or [H, W]
        indices: int, range, list, or None (default: first 2 for batch, 0 for single)
    """
    # Convert to numpy and normalize dimensions
    if hasattr(images, 'cpu'):  # torch tensor
        img_array = images.cpu().numpy()
    else:
        img_array = np.asarray(images)

    if img_array.ndim == 2:
        img_array = img_array[None, ...]  # [H, W] -> [1, H, W]
        single_image = True
    else:
        single_image = False

    B, Y, X = img_array.shape
    img_array = img_array.transpose(0, 2, 1)  # Transpose to match matlab format

    # Handle indices
    if indices is None:
        indices = [0] if single_image else list(range(min(2, B)))
    elif isinstance(indices, int):
        indices = [indices]
    else:
        indices = list(indices)

    # Filter valid indices
    valid_indices = [i for i in indices if 0 <= i < B]
    if not valid_indices:
        print(f"No valid indices. Batch size: {B}")
        return

    # Display images
    for i in valid_indices:
        plt.figure(figsize=(6, 6))
        plt.imshow(img_array[i], extent=[0, Y, X, 0], interpolation='nearest', cmap='viridis')
        plt.xlabel("Y")
        plt.ylabel("X")
        plt.title("Raster output" if single_image else f"Raster output for image {i}")
        plt.colorbar()
        plt.show()

import numpy as np


def fsc(vol1, vol2, n_bins=20):
    """
    Calculate Fourier Shell Correlation between two 3D volumes.

    Parameters:
    vol1, vol2: numpy arrays of shape (N, N, N)
        Two 3D volumes to compare
    n_bins: int, optional
        Number of frequency shell bins. If None, uses N//2 (default)

    Returns:
    frequencies: numpy array
        Spatial frequencies as fraction of Nyquist frequency
    correlations: numpy array
        FSC values at each frequency shell
    """
    # handle multiple volumes recursively
    if isinstance(vol1, list):
        freqs = []
        fsc_vals = []
        for vol in vol1:
            freq, fsc_val = fsc(vol, vol2, n_bins=20)
            freqs.append(freq)
            fsc_vals.append(fsc_val)
        return freqs, fsc_vals

    assert vol1.shape == vol2.shape, "Volumes must have the same shape"
    assert len(vol1.shape) == 3, "Volumes must be 3D"
    n = vol1.shape[0]

    # Compute 3D FFTs
    fft1 = np.fft.fftshift(np.fft.fftn(vol1))
    fft2 = np.fft.fftshift(np.fft.fftn(vol2))

    # Create coordinate grids
    center = n // 2
    x, y, z = np.meshgrid(np.arange(n) - center,
                          np.arange(n) - center,
                          np.arange(n) - center, indexing='ij')

    # Calculate radial distances from center
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Define frequency shells (up to Nyquist)
    max_radius = center
    n_shells = n_bins if n_bins is not None else max_radius

    frequencies = []
    correlations = []

    for i in range(1, n_shells + 1):
        # Define shell boundaries
        r_inner = (i - 1) * max_radius / n_shells
        r_outer = i * max_radius / n_shells

        # Create mask for current shell
        mask = (r >= r_inner) & (r < r_outer)

        if not np.any(mask):
            continue

        # Extract values in this shell
        f1_shell = fft1[mask]
        f2_shell = fft2[mask]

        # Calculate correlation coefficient
        numerator = np.sum(f1_shell * np.conj(f2_shell)).real
        denom1 = np.sum(np.abs(f1_shell) ** 2).real
        denom2 = np.sum(np.abs(f2_shell) ** 2).real
        if denom1 > 0 and denom2 > 0:
            correlation = numerator / np.sqrt(denom1 * denom2)
            frequencies.append(i / n_shells)  # Fraction of Nyquist
            correlations.append(correlation)
        else:
            raise ValueError

    freqs, fsc_vals = np.array(frequencies), np.array(correlations)
    return freqs, fsc_vals

import numpy as np
import torch
import torch.nn.functional as F

def calc_r_factor(volume, images, angles, device="cuda", shift_half_oixel=False):
    """
    GPU-accelerated R-factor calculation using PyTorch.

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (Z, Y, X).
    images : np.ndarray
        Ground truth projection images of shape (N, Y, X).
    angles : np.ndarray
        Rotation angles around the y-axis in degrees, shape (N,).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    float
        The R-factor value.
    """
    # Convert inputs to torch tensors
    volume_t = torch.tensor(volume, dtype=torch.float32, device=device)
    images_t = torch.tensor(images, dtype=torch.float32, device=device)
    angles_t = torch.tensor(angles, dtype=torch.float32, device=device)

    N, Y, X = images.shape
    Z = volume.shape[0]

    # Normalize coordinates for grid_sample: range [-1, 1]
    z_lin = torch.linspace(-1, 1, Z, device=device)
    y_lin = torch.linspace(-1, 1, Y, device=device)
    x_lin = torch.linspace(-1, 1, X, device=device)
    zz, yy, xx = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1)  # (Z, Y, X, 3)

    projections = []
    for angle in angles_t:
        theta = torch.deg2rad(angle)
        cos, sin = torch.cos(theta), torch.sin(theta)

        # Rotation matrix around y-axis
        R = torch.tensor([[cos, 0, sin],
                          [0,   1, 0],
                          [-sin,0, cos]], device=device)

        # Rotate coordinates
        grid_rot = grid @ R.T

        # grid_sample expects (N, D, H, W), add batch & channel
        grid_rot = grid_rot.unsqueeze(0)  # (1, Z, Y, X, 3)

        vol_in = volume_t.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
        rotated = F.grid_sample(vol_in, grid_rot, align_corners=True)
        rotated = rotated.squeeze()  # (Z, Y, X)

        # Projection along Z-axis
        proj = rotated.sum(dim=0)  # (Y, X)
        projections.append(proj)

    projections = torch.stack(projections, dim=0)  # (N, Y, X)

    # R-factor
    r_factor = torch.sum(torch.abs(images_t - projections)) / torch.sum(torch.abs(images_t))
    return r_factor.item()


def calc_r_factor_memory_efficient(volume, images, angles, device="cuda"):
    """
    Memory-efficient GPU-accelerated R-factor calculation using PyTorch.
    Processes one angle and one y-slice at a time to minimize memory usage.

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (Z, Y, X).
    images : np.ndarray
        Ground truth projection images of shape (N, Y, X).
    angles : np.ndarray
        Rotation angles around the y-axis in degrees, shape (N,).
    device : str
        "cuda" or "cpu".
    shift_half_pixel : bool
        Whether to shift coordinates by half a pixel.

    Returns
    -------
    float
        The R-factor value.
    """
    # Convert inputs to torch tensors
    volume_t = torch.tensor(volume, dtype=torch.float32, device=device)
    images_t = torch.tensor(images, dtype=torch.float32, device=device)
    angles_t = torch.tensor(angles, dtype=torch.float32, device=device)

    N, Y, X = images.shape
    Z, _, _ = volume.shape

    # Pre-compute coordinate grids for XZ plane (reused for each y-slice)
    # Normalize coordinates for grid_sample: range [-1, 1]
    z_coords = torch.linspace(-1, 1, Z, device=device)
    x_coords = torch.linspace(-1, 1, X, device=device)

    # Create meshgrid for XZ plane
    zz, xx = torch.meshgrid(z_coords, x_coords, indexing="ij")  # (Z, X)

    # Initialize accumulator for total absolute difference and sum of ground truth
    total_diff = 0.0
    total_gt_sum = torch.sum(torch.abs(images_t)).item()

    # Process one angle at a time
    for angle_idx, angle in enumerate(angles_t):
        theta = torch.deg2rad(angle)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        # Rotation matrix around y-axis for XZ coordinates
        # [x', z'] = [cos*x + sin*z, -sin*x + cos*z]
        xx_rot = cos_theta * xx + sin_theta * zz
        zz_rot = -sin_theta * xx + cos_theta * zz

        # Initialize projection for this angle
        projection = torch.zeros(Y, X, device=device)

        # Process one y-slice at a time
        for y_idx in range(Y):
            # Create sampling grid for this y-slice: (1, 1, Z, X, 3)
            # grid_sample expects (batch, depth, height, width, 3) for 3D
            # where the last dimension is [x, y, z] coordinates
            y_coord = torch.full_like(xx, 2.0 * y_idx / (Y - 1) - 1.0 if Y > 1 else 0.0)

            # Stack coordinates: [x, y, z] format for grid_sample
            grid_slice = torch.stack([xx_rot, y_coord, zz_rot], dim=-1)  # (Z, X, 3)
            grid_slice = grid_slice.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, X, 3)

            # Sample from volume (add batch and channel dimensions)
            vol_batch = volume_t.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)

            # Sample the rotated slice
            rotated_slice = F.grid_sample(
                vol_batch,
                grid_slice,
                align_corners=True,
                mode='bilinear',
                padding_mode='zeros'
            )
            rotated_slice = rotated_slice.squeeze()  # (Z, X)

            # Project along Z-axis (sum over Z dimension)
            projection[y_idx, :] = rotated_slice.sum(dim=0)

        # Accumulate difference for this angle
        diff = torch.sum(torch.abs(images_t[angle_idx] - projection))
        total_diff += diff.item()

    # Calculate R-factor
    r_factor = total_diff / total_gt_sum
    return r_factor


def angular_errors_deg(true_rots, est_rots, degrees=True):
    N = true_rots.shape[0]
    angles = np.zeros(N)
    for i in range(N):
        M = true_rots[i].T @ est_rots[i]
        # numeric safeguard
        val = (np.trace(M) - 1.0) / 2.0
        val = np.clip(val, -1.0, 1.0)
        angles[i] = np.arccos(val)
    if degrees:
        angles = np.degrees(angles)
    return angles

def save_output(points, weights, bandwidth, reconstruction, logging_prefix, outdir):
    print("Saving output in", outdir, "shape:", reconstruction.shape)
    reconstruction = reconstruction.copy()
    os.makedirs(outdir, exist_ok=True)
    # Save reconstruction as npz
    np.savez(os.path.join(outdir, f"{logging_prefix}_volume_reconstruction.npz"),
             reconstruction=reconstruction)


if __name__ == "__main__":
    print("Creating test data...")
    np.random.seed(42)
    torch.manual_seed(42)

    z, y, x = 80, 64, 64  # example volume dimensions
    N = 20  # number of angles/images
    Y, X = 64, 64  # image dimensions

    # Create arrays
    volume = np.random.rand(z, y, x)  # random volume
    angles = np.arange(N)  # range from 0 to N-1
    images = np.random.rand(N, Y, X)  # random images

    r1 = calc_r_factor(volume, images, angles, device="cuda")
    r2 = calc_r_factor_memory_efficient(volume, images, angles)
    print(r1)
    print(r2)
