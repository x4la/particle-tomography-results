import argparse
import importlib.util
import sys

from scripts import benchmark, plot
from scripts.datasets import DATASETS, dataset_names, missing_files


PLOTTERS = {
    "protein": plot.plot_protein,
    "vesicle": plot.plot_vesicle,
    "thinfilm": plot.plot_thinfilm,
    "platinum": plot.plot_platinum,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproducible result workflows.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to run. Defaults to the bundled reproducible subset. Can be passed more than once.",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Run all registered datasets, including third-party datasets that must be placed under data/ first.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(benchmark.METHOD_CONFIGS),
        default=None,
        help="Methods to benchmark. Defaults to particle_tomography, SIRT, and FBP.",
    )
    parser.add_argument(
        "--with-resire",
        action="store_true",
        help="Include precomputed RESIRE outputs from out/RESIRE/<dataset>/ in metrics and plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run benchmarks only and skip plot generation.",
    )
    parser.add_argument(
        "--show-3d-volumes",
        action="store_true",
        help="Open interactive 3D volume views during plotting.",
    )
    return parser.parse_args()


def require_available_inputs(names, include_resire=False):
    missing = []
    for name in names:
        missing.extend(missing_files(name, include_resire=include_resire))
    if missing:
        print("Missing required input files:")
        for path in missing:
            print(f"  {path}")
        print("\nRun `python -m scripts.check_data --include-external --include-resire` for details.")
        return False
    return True


def require_astra_if_needed(methods):
    if not ({"SIRT", "FBP"} & set(methods)):
        return True
    if importlib.util.find_spec("astra") is not None:
        return True
    print("ASTRA Toolbox is required for SIRT/FBP. Install environment.yml or choose different --methods.")
    return False


def main():
    args = parse_args()
    names = args.dataset or dataset_names(include_external=args.include_external)
    methods = list(args.methods or benchmark.DEFAULT_METHODS)
    if args.with_resire and "RESIRE" not in methods:
        methods.append("RESIRE")

    if not require_available_inputs(names, include_resire=args.with_resire):
        return 1
    if not require_astra_if_needed(methods):
        return 1

    benchmark.run_benchmarks(dataset_names_filter=names, method_names_filter=methods)

    if not args.skip_plots:
        for name in names:
            PLOTTERS[name](show_3d_volumes=args.show_3d_volumes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
