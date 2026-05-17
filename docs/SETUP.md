# Setup & Installation Guide

## EIDORS Installation

1. Download EIDORS from: http://eidors3d.sourceforge.net/download.shtml
2. Extract the archive to `matlab/eidors/`
3. The directory structure should be:
   ```
   matlab/eidors/
   ├── startup.m
   ├── eidors/
   ├── models/
   └── ...
   ```

## MATLAB Requirements

- MATLAB R2019b or later
- No additional toolboxes required (EIDORS is self-contained)
- For 3D cylindrical models: Netgen mesh generator
  - Download from: https://ngsolve.org/
  - Add to system PATH

## Python Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Install uv (if not already installed)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install all dependencies
uv sync

# Activate environment (optional, uv run handles this)
# Or use: uv run python <script.py>
```

## Verification

### MATLAB
```matlab
cd matlab
setup_eidors()
% Should print: "EIDORS initialised successfully"

% Quick test: create a mesh
[fmdl, vh] = create_mesh();
fprintf('Mesh has %d elements, %d measurements\n', size(fmdl.elems,1), length(vh.meas));
```

### Python
```bash
uv run python -c "from python.data.load_dataset import load_mat_dataset; print('OK')"
```

## Troubleshooting

### EIDORS startup fails
- Ensure you're running MATLAB from the `matlab/` directory
- Check EIDORS is fully extracted (not nested in extra directory)

### Netgen not found (3D models)
- 3D cylindrical models require Netgen
- If unavailable, use `geometry: '2d_circle'` in main.m config

### Memory issues during dataset generation
- Reduce `config.samples_per_class` in `main.m`
- For 16-electrode adjacent pattern: each sample is ~208 measurements
- 25,000 samples × 208 features × 8 bytes ≈ 42 MB (manageable)
