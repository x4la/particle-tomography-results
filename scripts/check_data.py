import argparse
import importlib.util
import sys

from scripts.datasets import DATASETS, ROOT, describe_dataset, dataset_names


def module_available(name):
    return importlib.util.find_spec(name) is not None


def print_environment_status(require_astra=False):
    checks = {
        "particle_tomography": module_available("particle_tomography"),
        "rasterizer": module_available("rasterizer"),
        "torch": module_available("torch"),
        "astra": module_available("astra"),
    }
    for name, ok in checks.items():
        print(f"{name:22} {'ok' if ok else 'missing'}")

    required = ["particle_tomography", "rasterizer", "torch"]
    if require_astra:
        required.append("astra")
    return [name for name in required if not checks[name]]


def display_path(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def print_dataset_status(names, include_resire=False):
    missing_any = []
    for name in names:
        info = describe_dataset(name, include_resire=include_resire)
        print(f"\n{name} ({info['scope']}): {info['status']}")
        print(f"  source: {info['source']}")
        if info["note"]:
            print(f"  note: {info['note']}")
        for label, path in info["files"].items():
            marker = "ok" if path.exists() else "missing"
            print(f"  {label:18} {marker:8} {display_path(path)}")
        if info["resire_output"] is not None:
            path = info["resire_output"]
            marker = "ok" if path.exists() else "missing"
            print(f"  {'resire_output':18} {marker:8} {display_path(path)}")
        missing_any.extend(info["missing"])
    return missing_any


def parse_args():
    parser = argparse.ArgumentParser(description="Check available data and optional method dependencies.")
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also check third-party datasets that are not bundled in this repository.",
    )
    parser.add_argument(
        "--include-resire",
        action="store_true",
        help="Also check for precomputed RESIRE output volumes under out/RESIRE/<dataset>/.",
    )
    parser.add_argument(
        "--require-astra",
        action="store_true",
        help="Treat ASTRA Toolbox as required. Use this before running SIRT/FBP benchmarks.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help="Check only the selected dataset. Can be passed more than once.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    names = args.dataset or dataset_names(include_external=args.include_external)

    print("Environment")
    missing_modules = print_environment_status(require_astra=args.require_astra)

    print("\nData")
    missing_paths = print_dataset_status(names, include_resire=args.include_resire)

    if missing_modules or missing_paths:
        print("\nMissing requirements detected.")
        return 1
    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
