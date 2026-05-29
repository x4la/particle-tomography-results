from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from data.loader import load_platinum_data, load_protein_data, load_thinfilm_data, load_vesicle_data


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    included: bool
    files: dict[str, Path]
    loader: Callable
    has_shifts: bool
    has_true_volume: bool
    source: str
    note: str = ""

    @property
    def resire_output(self) -> Path:
        return ROOT / "out" / "RESIRE" / self.name / "RESIRE_volume_reconstruction.mat"


DATASETS = {
    "protein": DatasetSpec(
        name="protein",
        included=True,
        files={
            "projection_file": ROOT / "data" / "protein" / "projections_6b3r_wedge40_step-2.0_pixel-132_blur-0.6299118258855213.npz",
            "rotations_file": ROOT / "data" / "protein" / "projections_6b3r_wedge40_step-2.0_euler_angles.txt",
            "true_volume_file": ROOT / "data" / "protein" / "6b3r_pixel-132_blur-0.6299118258855213.npz",
        },
        loader=load_protein_data,
        has_shifts=False,
        has_true_volume=True,
        source="Bundled simulated protein dataset.",
    ),
    "vesicle": DatasetSpec(
        name="vesicle",
        included=False,
        files={
            "projection_file": ROOT / "data" / "vesicle" / "projections_vesicle.mat",
            "rotations_file": ROOT / "data" / "vesicle" / "projections_vesicle_euler_angles.mat",
            "true_volume_file": ROOT / "data" / "vesicle" / "vesicle.mrc",
        },
        loader=load_vesicle_data,
        has_shifts=True,
        has_true_volume=True,
        source="RESIRE tomography dataset: https://zenodo.org/records/7819857",
        note="Not redistributed here because it is third-party data.",
    ),
    "thinfilm": DatasetSpec(
        name="thinfilm",
        included=False,
        files={
            "projection_file": ROOT / "data" / "thinfilm" / "projections_thinfilm.mat",
            "rotations_file": ROOT / "data" / "thinfilm" / "projections_thinfilm_euler_angles.mat",
        },
        loader=load_thinfilm_data,
        has_shifts=False,
        has_true_volume=False,
        source="RESIRE tomography dataset: https://zenodo.org/records/7819857",
        note="Not redistributed here because it is third-party data and large.",
    ),
    "platinum": DatasetSpec(
        name="platinum",
        included=False,
        files={
            "projection_file": ROOT / "data" / "platinum_particle" / "tiltser_180.tif",
            "rotations_file": ROOT / "data" / "platinum_particle" / "euler_angles.txt",
        },
        loader=load_platinum_data,
        has_shifts=False,
        has_true_volume=False,
        source="Nanomaterial tomography dataset: https://springernature.figshare.com/collections/Nanomaterial_datasets_to_advance_tomography_in_scanning_transmission_electron_microscopy/2185342",
        note="Not redistributed here because it is third-party data and large.",
    ),
}


def dataset_names(include_external=False):
    return [name for name, spec in DATASETS.items() if include_external or spec.included]


def benchmark_dataset_configs(include_external=False):
    configs = {}
    for name in dataset_names(include_external=include_external):
        spec = DATASETS[name]
        configs[name] = {
            **spec.files,
            "loader": spec.loader,
            "has_shifts": spec.has_shifts,
            "has_true_volume": spec.has_true_volume,
        }
    return configs


def missing_files(name, include_resire=False):
    spec = DATASETS[name]
    paths = list(spec.files.values())
    if include_resire:
        paths.append(spec.resire_output)
    return [path for path in paths if not path.exists()]


def describe_dataset(name, include_resire=False):
    spec = DATASETS[name]
    missing = missing_files(name, include_resire=include_resire)
    status = "ok" if not missing else "missing"
    scope = "included" if spec.included else "external"
    return {
        "name": name,
        "scope": scope,
        "status": status,
        "source": spec.source,
        "note": spec.note,
        "files": spec.files,
        "resire_output": spec.resire_output if include_resire else None,
        "missing": missing,
    }
