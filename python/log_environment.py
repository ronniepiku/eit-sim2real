"""Environment and hardware logging for reproducibility.

Captures system information, package versions, and hardware details
to support the reproducibility claims in the methodology chapter.

Usage:
    uv run python/log_environment.py
    uv run python/log_environment.py --output results/environment.json
"""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path


def get_environment_info() -> dict:
    """Collect comprehensive environment information."""
    info: dict = {
        "timestamp": datetime.now(UTC).isoformat(),
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "packages": {},
        "hardware": {},
    }

    # Package versions
    packages_to_check = [
        "torch",
        "numpy",
        "scipy",
        "sklearn",
        "pandas",
        "joblib",
        "yaml",
    ]
    for pkg_name in packages_to_check:
        try:
            if pkg_name == "sklearn":
                import sklearn

                info["packages"]["scikit-learn"] = sklearn.__version__
            elif pkg_name == "yaml":
                import yaml

                info["packages"]["pyyaml"] = yaml.__version__
            else:
                mod = __import__(pkg_name)
                info["packages"][pkg_name] = mod.__version__
        except (ImportError, AttributeError):
            info["packages"][pkg_name] = "not installed"

    # PyTorch-specific info
    try:
        import torch

        info["hardware"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["hardware"]["cuda_version"] = torch.version.cuda
            info["hardware"]["gpu_name"] = torch.cuda.get_device_name(0)
            info["hardware"]["gpu_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1024**2
            )
            info["hardware"]["gpu_count"] = torch.cuda.device_count()
        info["hardware"]["torch_backends"] = {
            "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None,
            "cudnn_enabled": torch.backends.cudnn.enabled,
        }
    except ImportError:
        pass

    # CPU info
    try:
        import os

        info["hardware"]["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # Random seeds used
    info["reproducibility"] = {
        "numpy_seed": 42,
        "torch_seed": 42,
        "matlab_seed": 42,
        "train_test_split_seed": 42,
    }

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Log environment for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/environment.json"),
        help="Output path for environment JSON.",
    )
    args = parser.parse_args()

    info = get_environment_info()

    # Print to console
    print("=" * 60)
    print("ENVIRONMENT REPORT")
    print("=" * 60)
    print(f"OS: {info['system']['os']} {info['system']['os_version']}")
    print(f"Python: {info['system']['python_version']}")
    print(f"Architecture: {info['system']['architecture']}")
    print(f"Processor: {info['system']['processor']}")
    print()
    print("Packages:")
    for pkg, ver in info["packages"].items():
        print(f"  {pkg}: {ver}")
    print()
    print("Hardware:")
    for key, val in info["hardware"].items():
        print(f"  {key}: {val}")
    print("=" * 60)

    # Save to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
