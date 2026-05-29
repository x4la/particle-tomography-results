from pathlib import Path
import os
import time
import h5py
import numpy as np
import json
from typing import Dict, Any

import torch
import particle_tomography as pt

try:
    from scripts.reconstruct.fbp_astra import reconstruct_fbp
    from scripts.reconstruct.sirt_astra import reconstruct_sirt
except ModuleNotFoundError as exc:
    if exc.name != "astra":
        raise

    def _missing_astra(*args, **kwargs):
        raise RuntimeError("ASTRA Toolbox is required for SIRT/FBP. Install the conda environment or run without those methods.")

    reconstruct_fbp = _missing_astra
    reconstruct_sirt = _missing_astra

from scripts.datasets import benchmark_dataset_configs, dataset_names
from scripts.utils import fsc, calc_r_factor_memory_efficient, save_output


ROOT = Path(__file__).resolve().parent.parent
PARTICLE_TOMOGRAPHY_DEVICE = os.environ.get("PARTICLE_TOMOGRAPHY_DEVICE", "auto")
R_FACTOR_DEVICE = os.environ.get("R_FACTOR_DEVICE", PARTICLE_TOMOGRAPHY_DEVICE)
DATASET_CONFIGS = benchmark_dataset_configs(include_external=True)
DEFAULT_METHODS = ["particle_tomography", "SIRT", "FBP"]

METHOD_CONFIGS = {
    "particle_tomography": {
        "function": None,  # Custom handling
        "needs_angles": False,
        "needs_rotations": True,  # Uses rotation matrices
        "needs_shifts": True,  # When available
        "iterations": None,  # Dataset dependent
        "dataset_params": {
            "vesicle": {"num_points": 5000, "total_iterations": 2000, "kernel_size": 3},
            "protein": {"num_points": 20000, "total_iterations": 2500, "kernel_size": 3},
            "thinfilm": {
                "num_points": 50000,
                "particle_init_mode": "thinfilm",
                "total_iterations": 4500,
                "lr": 5e-3,
                "num_rejuvenates": 2,
                "kernel_size": 9,
                "geom_start_fraction": 0.8
            },
            "platinum": {
                "num_points": 500000,
                "total_iterations": 250,
                "lr": 5e-3,
                "num_rejuvenates": 0,
                "kernel_size": 13,
                "start_bandwidth": 5,
                "geom_start_fraction": 0.8,
            }
        }
    },
    "RESIRE": {
        "function": None,
        "needs_angles": False,
        "needs_rotations": False,
        "needs_shifts": False,
        "iterations": 1,  # Loaded from precomputed external output
    },
    "SIRT": {
        "function": reconstruct_sirt,
        "needs_angles": True,  # Uses angles_deg[:,1]
        "needs_rotations": False,
        "needs_shifts": False,
        "iterations": 200,  # Default, can be overridden
    },
    "FBP": {
        "function": reconstruct_fbp,
        "needs_angles": True,  # Uses angles_deg[:,1]
        "needs_rotations": False,
        "needs_shifts": False,
        "iterations": 1,  # Single pass
    }
}




def _resolve_torch_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

RESIRE_METADATA = {
    "vesicle": {
        "iterations": 1000,
        "training_time": 2.2,
        "r_factor_custom_forward": 0.089
    },
    "protein": {
        "iterations": 600,
        "training_time": 19.6,
        "r_factor_custom_forward": 0.014
    },
    "thinfilm": {
        "iterations": 200,
        "training_time": 123,
        "r_factor_custom_forward": 0.389
    },
    "platinum": {
        "iterations": 50,
        "training_time": 1329,
        "r_factor_custom_forward": 0.085
    }
}


class DatasetManager:
    """Manages loading and providing data in the format each method needs"""

    def __init__(self):
        self._cached_data = {}

    def get_data(self, dataset_name: str) -> Dict[str, Any]:
        """Load and cache dataset, return all available data"""
        if dataset_name in self._cached_data:
            return self._cached_data[dataset_name]

        config = DATASET_CONFIGS[dataset_name]
        input_paths = {k: v for k, v in config.items() if k.endswith('_file')}

        # Load data based on dataset characteristics
        if config["has_shifts"] and config["has_true_volume"]:
            images, rotations, shifts, true_volume, angles_deg = config["loader"](
                input_paths, return_angles=True
            )
            data = {
                "images": images,
                "rotations": rotations,
                "shifts": shifts,
                "true_volume": true_volume,
                "angles_deg": angles_deg
            }
        elif config["has_true_volume"]:
            images, rotations, true_volume, angles_deg = config["loader"](
                input_paths, return_angles=True
            )
            data = {
                "images": images,
                "rotations": rotations,
                "true_volume": true_volume,
                "angles_deg": angles_deg
            }
        else:
            images, rotations, angles_deg = config["loader"](
                input_paths, return_angles=True
            )
            data = {
                "images": images,
                "rotations": rotations,
                "angles_deg": angles_deg
            }

        self._cached_data[dataset_name] = data
        return data


class MethodRunner:
    """Handles running different reconstruction methods with their specific requirements"""

    def __init__(self, data_manager: DatasetManager):
        self.data_manager = data_manager

    def run_method(self, method_name: str, dataset_name: str):
        """Run a specific method on a dataset and return result + metadata"""
        config = METHOD_CONFIGS[method_name]
        data = self.data_manager.get_data(dataset_name)
        dataset_config = DATASET_CONFIGS[dataset_name]

        # Prepare arguments for standard methods
        args = [data["images"]]
        kwargs = {}

        if config.get("needs_angles"):
            args.append(data["angles_deg"][:, 1])
        if config.get("needs_rotations"):
            args.append(data["rotations"])
        if config.get("needs_shifts") and "shifts" in data:
            args.append(data["shifts"])
        if dataset_config.get("has_shifts"):
            kwargs["shift_half_pixel"] = True

        # Special handling for particle_tomography
        if method_name == "particle_tomography":
            params = config["dataset_params"][dataset_name].copy()
            params["device"] = PARTICLE_TOMOGRAPHY_DEVICE

            # Build args specifically for particle_tomography
            pt_args = [data["images"], data["rotations"]]
            if "shifts" in data:
                pt_args.append(data["shifts"])

            t1 = time.time()
            model = pt.reconstruct(*pt_args, **params)
            result = model.get_volume()
            t2 = time.time()

            # Save sparse model
            model_outdir = ROOT / "out" / "particle_tomography" / dataset_name
            model_outdir.mkdir(parents=True, exist_ok=True)
            model.save_model(model_outdir / "model.pt")

            fsc_at_nyquist = None
            if dataset_config.get("has_true_volume"):
                true_vol = data["true_volume"]
                _, fsc_vals = model.get_fsc(true_vol)
                fsc_at_nyquist = fsc_vals[-1].item()

            r_factor_sparse = model.get_r_factor()
            r_factor_voxel = calc_r_factor_memory_efficient(result, data["images"], data["angles_deg"][:, 1], device=_resolve_torch_device(R_FACTOR_DEVICE))

            metadata = {
                "iterations": params.get("total_iterations"),
                "method_name": method_name,
                "dataset_name": dataset_name
            }

            return result, metadata, t2-t1, fsc_at_nyquist, r_factor_voxel, r_factor_sparse

        # Special handling for RESIRE (precomputed loader)
        elif method_name == "RESIRE":
            return self._run_resire_loader(dataset_name, data, dataset_config)

        # Standard reconstruction method
        t1 = time.time()
        result = config["function"](*args, **kwargs)
        t2 = time.time()

        fsc_at_nyquist = None
        if dataset_config.get("has_true_volume"):
            true_vol = data["true_volume"]
            _, fsc_vals = fsc(true_vol, result)
            fsc_at_nyquist = fsc_vals[-1].item()

        r_factor_voxel = calc_r_factor_memory_efficient(result, data["images"], data["angles_deg"][:, 1], device=_resolve_torch_device(R_FACTOR_DEVICE))
        r_factor_sparse = None

        metadata = {
            "iterations": config.get("iterations"),
            "method_name": method_name,
            "dataset_name": dataset_name
        }

        return result, metadata, t2-t1, fsc_at_nyquist, r_factor_voxel, r_factor_sparse

    def _run_resire_loader(self, dataset_name: str, data: Dict[str, Any], dataset_config: Dict[str, Any]):
        """Load precomputed RESIRE reconstruction and calculate metrics"""
        import scipy.io

        # Try to load RESIRE volume from either .npz or .mat
        path = ROOT / "out" / "RESIRE" / dataset_name / "RESIRE_volume_reconstruction.mat"
        if dataset_name == "platinum":
            with h5py.File(path, "r") as f:
                volume = np.array(f["rec"])
        else:
            vol_data = scipy.io.loadmat(path)
            keys = [k for k in vol_data if not k.startswith("__")]
            volume = vol_data[keys[0]].transpose(1, 0, 2).copy()


        if volume is None:
            raise FileNotFoundError(f"Could not find RESIRE reconstruction for dataset {dataset_name}")

        # Get metadata for this dataset
        resire_meta = RESIRE_METADATA[dataset_name]
        training_time = resire_meta["training_time"]

        # Calculate FSC at Nyquist if true volume available
        fsc_at_nyquist = None
        if dataset_config["has_true_volume"]:
            true_vol = data["true_volume"]
            freqs, fsc_vals = fsc(true_vol, volume)
            fsc_at_nyquist = fsc_vals[-1].item()

        # Calculate r-factor
        r_factor_voxel = calc_r_factor_memory_efficient(volume, data["images"], data["angles_deg"][:, 1], device=_resolve_torch_device(R_FACTOR_DEVICE))
        r_factor_custom_forward = resire_meta["r_factor_custom_forward"]

        metadata = {
            "iterations": resire_meta["iterations"],
            "method_name": "RESIRE",
            "dataset_name": dataset_name
        }

        return volume, metadata, training_time, fsc_at_nyquist, r_factor_voxel, r_factor_custom_forward

    def _run_particle_tomography(self, dataset_name: str, data: Dict[str, Any]):
        """Special handling for particle_tomography method"""
        config = METHOD_CONFIGS["particle_tomography"]
        params = config["dataset_params"][dataset_name].copy()
        dataset_config = DATASET_CONFIGS[dataset_name]

        # Build arguments
        args = [data["images"], data["rotations"]]
        if "shifts" in data:
            args.append(data["shifts"])

        # Add device parameter
        params["device"] = PARTICLE_TOMOGRAPHY_DEVICE

        # Extract iterations for metadata
        iterations = params.get("total_iterations", None)

        # Run reconstruction
        t1 = time.time()
        model = pt.reconstruct(*args, **params)
        result = model.get_volume()
        t2 = time.time()

        # save sparse model
        model_outdir = ROOT / "out" / "particle_tomography" / dataset_name
        model_outdir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_outdir / "model.pt")

        fsc_at_nyquist = None
        if dataset_config["has_true_volume"]:
            true_vol = data["true_volume"]
            freqs, fsc_vals = model.get_fsc(true_vol)
            fsc_at_nyquist = fsc_vals[-1].item()
        r_factor_sparse = model.get_r_factor()
        r_factor_voxel = calc_r_factor_memory_efficient(result, data["images"], data["angles_deg"][:, 1], device=_resolve_torch_device(R_FACTOR_DEVICE))


        metadata = {
            "iterations": iterations,
            "method_name": "particle_tomography",
            "dataset_name": dataset_name,
        }

        return result, metadata, t2-t1, fsc_at_nyquist, r_factor_voxel, r_factor_sparse


def benchmark_method(method_name: str, dataset_name: str, data_manager: DatasetManager,
                     method_runner: MethodRunner) -> Dict[str, Any]:
    """Run reconstruction and capture all metrics"""
    print(f"Benchmarking {method_name} on {dataset_name}...")
    # Run the method
    reconstructed, metadata, t, fsc, r_factor_voxel, r_factor_sparse = method_runner.run_method(method_name, dataset_name)
    success = None
    # Save output if successful
    if reconstructed is not None:
        success = True
        if method_name != "RESIRE": # dont save Resire reconstructions, since they already exist
            outdir = ROOT / "out" / method_name / dataset_name
            save_output(None, None, None, reconstructed,
                             logging_prefix=method_name, outdir=outdir)

    if fsc is not None:
        fsc *= 100 # display in %
    r_factor_voxel *= 100 # display in %
    if r_factor_sparse is not None:
        r_factor_sparse *= 100

    return {
        'dataset': dataset_name,
        'method': method_name,
        'runtime_sec': t,
        'iterations': metadata.get("iterations"),
        'fsc_nyquist': fsc, # display in %
        'r_factor_voxel': r_factor_voxel, # display in %
        'r_factor_sparse': r_factor_sparse,
        'success': success,
        'dataset_size_mb': None  # Can be calculated from data if needed
    }


def run_benchmarks(dataset_names_filter=None, method_names_filter=None) -> list:
    """Run selected method/dataset combinations and collect results."""
    data_manager = DatasetManager()
    method_runner = MethodRunner(data_manager)

    results = []
    datasets = list(dataset_names_filter or dataset_names(include_external=False))
    methods = list(method_names_filter or DEFAULT_METHODS)

    unknown_datasets = sorted(set(datasets) - set(DATASET_CONFIGS))
    unknown_methods = sorted(set(methods) - set(METHOD_CONFIGS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")

    for dataset in datasets:
        for method in methods:
            if dataset == "platinum" and method == "SIRT":
                # out of memory, so dont try to run it
                results.append({
                    'dataset': dataset,
                    'method': method,
                    'runtime_sec': None,
                    'iterations': None,
                    'fsc_nyquist': None,
                    'r_factor_voxel': None,
                    'r_factor_sparse': None,
                    'success': False,
                    'failure_reason': 'out_of_memory',
                    'dataset_size_mb': None
                })
                continue
            result = benchmark_method(method, dataset, data_manager, method_runner)
            results.append(result)

            # Save intermediate results
            with open(ROOT / "benchmark_results.json", 'w') as f:
                json.dump(results, f, indent=2)

    # when done, read benchmark_results.json and generate latex table.
    print("Done with benchmarks. Generating latex table from benchmark_results.json.")
    latex_table = json_to_latex_table()
    # Save table to file
    with open(ROOT / "latex_table_rows.txt", 'w') as f:
        f.write(latex_table)
    return results


def run_all_benchmarks() -> list:
    """Run the included reproducible benchmark subset."""
    return run_benchmarks()


def json_to_latex_table(results_file: str = None) -> str:
    """Convert benchmark results to LaTeX table format"""
    if results_file is None:
        results_file = ROOT / "benchmark_results.json"

    with open(results_file) as f:
        data = json.load(f)

    # Filter successful results
    data_set_name = {"vesicle": r"Simulated Vesicle\\ (41 \texttimes 64 \texttimes 64)",
                     "protein": r"Simulated Protein\\ (71 \texttimes 132 \texttimes 132)",
                     "thinfilm": r"Thin Film\\ (51\texttimes406\texttimes425)",
                     "platinum": r"Platinum Particles\\ (140\texttimes947\texttimes1033)"}

    # Hardcoded volume sizes (z, y, x)
    volume_sizes = {
        "vesicle": (64, 64, 64),
        "protein": (132, 132, 132),
        "thinfilm": (425, 406, 425),
        "platinum": (1033, 947, 1033),
    }

    # Group by dataset
    by_dataset = {}
    for row in data:
        dataset = row['dataset']
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(row)

    # Generate LaTeX table rows
    latex_rows = []
    for dataset, methods in by_dataset.items():
        dataset_display = data_set_name[dataset]
        voxel_volume_bytes = volume_sizes[dataset][0] * volume_sizes[dataset][1] * volume_sizes[dataset][2] * 4 # z * y * x * size_of(float32)
        # (num_points * num_params_per_point + 1) * size_of(float32) where num_params_per_point = 4 and +1 comes from the global bandwidth
        sparse_volume_bytes = (METHOD_CONFIGS["particle_tomography"]["dataset_params"][dataset]["num_points"] * 4 + 1) * 4
        compression_factor = voxel_volume_bytes / sparse_volume_bytes
        volume_display = f"{compression_factor:.1f}"

        for i, method in enumerate(methods):
            if i == 0:
                dataset_cell = f"\\multirow{{{len(methods)}}}{{*}}{{\\parbox{{2cm}}{{{dataset_display}}}}}"
                size_cell = f"\\multirow{{{len(methods)}}}{{*}}{{{volume_display}}}"
            else:
                dataset_cell = ""
                size_cell = ""

            # Handle None values
            runtime = f"{method['runtime_sec']:.1f}" if method['runtime_sec'] is not None else "-"
            iterations = str(method['iterations']) if method['iterations'] is not None else "N/A"
            fsc = f"{method['fsc_nyquist']:.1f}" if method['fsc_nyquist'] is not None else "-"
            r_factor_voxel = f"{method['r_factor_voxel']:.1f}" if method['r_factor_voxel'] is not None else "-"
            r_factor_sparse = f"{method['r_factor_sparse']:.1f}" if method['r_factor_sparse'] is not None else "-"
            if method['method'] == "particle_tomography":
                row = f"{dataset_cell} & Particle Tomography & {runtime} & {iterations} & {fsc} & {r_factor_voxel} ({r_factor_sparse}) & {size_cell} \\\\"
            elif method['method'] == "RESIRE":
                row = f"{dataset_cell} & {method['method']} & {runtime} & {iterations} & {fsc} & {r_factor_voxel} ({r_factor_sparse}) & {size_cell} \\\\"
            elif method['method'] == "SIRT":
                if dataset == "thinfilm":
                    row = f"{dataset_cell} & {method['method']} & {runtime} & {iterations} & {fsc} & {r_factor_voxel} \\phantom{{(00.0)}} & {size_cell} \\\\"
                if dataset == "platinum":
                    row = f"{dataset_cell} & {method['method']} & \\multicolumn{{4}}{{c}}{{\\textit{{out of memory}}}} & \\\\"
                else:
                    row = f"{dataset_cell} & {method['method']} & {runtime} & {iterations} & {fsc} & {r_factor_voxel} \\phantom{{(0.0)}} & {size_cell} \\\\"
            else:
                if dataset == "thinfilm":
                    row = f"{dataset_cell} & {method['method']} & {runtime} & {iterations} & {fsc} & {r_factor_voxel} \\phantom{{(00.0)}} & {size_cell} \\\\"
                else:
                    row = f"{dataset_cell} & {method['method']} & {runtime} & {iterations} & {fsc} & {r_factor_voxel} \\phantom{{(0.0)}} & {size_cell} \\\\"
            latex_rows.append(row)

        if dataset != list(by_dataset.keys())[-1]:  # Add midrule between datasets
            latex_rows.append("\\midrule")

    return '\n'.join(latex_rows)



if __name__ == "__main__":
    # Run the included reproducible benchmark subset and save reconstructions in /out.
    run_all_benchmarks()

    # read benchmark_results.json and generate latex table.
    latex_table = json_to_latex_table()
    print("LaTeX table rows:")
    print(latex_table)

    # Save table to file
    with open(ROOT / "latex_table_rows.txt", 'w') as f:
        f.write(latex_table)