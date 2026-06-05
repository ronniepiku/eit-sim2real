# Contributing

## Development Setup

```bash
# Clone and install
git clone https://github.com/ronniepiku/eit-sim2real.git
cd eit-sim2real
uv sync

# Install PyTorch (CPU for development):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install package in editable mode with dev tools:
uv pip install -e .
uv pip install pytest pytest-cov ruff mypy types-PyYAML pre-commit

# Set up pre-commit hooks:
pre-commit install
```

## Workflow

1. Create a feature branch from `main`
2. Make changes in `src/eit_sim2real/`
3. Add or update tests in `tests/`
4. Run the local pipeline to verify:

```bash
# Windows PowerShell:
.\scripts\run_pipeline.ps1

# Or manually:
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/eit_sim2real/ --ignore-missing-imports
pytest tests/ --cov=eit_sim2real -q
```

5. Commit (pre-commit hooks run lint + format automatically)
6. Open a pull request against `main`

## Code Style

- **Formatter**: ruff (line-length 88, Python 3.11 target)
- **Linter**: ruff with rules: E, F, W, I, N, UP, B, A, SIM, PT
- **Type checker**: mypy (strict mode)
- Uppercase variable names (`X`, `X_train`) follow ML convention
- Use `logging.getLogger(__name__)` — never `print()` in library code
- Configure logging only in CLI entry points

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=eit_sim2real --cov-report=term-missing

# Run specific test file
pytest tests/test_cnn1d.py -v

# Run tests matching a pattern
pytest tests/ -k "noise"
```

### Test Organisation

| File | Covers |
|------|--------|
| `test_baselines.py` | SVM, RF, MLP creation and training |
| `test_cnn1d.py` | CNN architecture, shapes, gradient flow |
| `test_config.py` | YAML config loading |
| `test_data.py` | Dataset splitting, normalisation, CV |
| `test_noise.py` | Noise model config, application, reproducibility |
| `test_cli.py` | CLI command invocation and error handling |
| `test_integration.py` | End-to-end train → evaluate pipeline |

### Writing Tests

- Use `pytest` fixtures for shared setup
- Tests must not require the full dataset (use synthetic data)
- Keep tests fast (< 1s each where possible)
- Name test files `test_<module>.py` and test functions `test_<behaviour>`

## Project Layout

```
src/eit_sim2real/       # Package source (imported as eit_sim2real)
tests/                  # All tests (pythonpath configured to src/)
scripts/                # Developer scripts
.github/workflows/      # CI configuration
```

### Import Convention

All internal imports use the full package path:

```python
from eit_sim2real.data import load_mat_dataset, prepare_splits
from eit_sim2real.models import EITConv1D, get_baseline
from eit_sim2real.utils import get_device, set_seeds
from eit_sim2real.constants import CLASS_NAMES
```

## CI/CD

GitHub Actions runs on every push/PR to `main`:

1. **Lint** — `ruff check` + `ruff format --check`
2. **Typecheck** — `mypy` with `--ignore-missing-imports`
3. **Test** — `pytest` on Python 3.11 + 3.12 matrix

All three jobs must pass before merge.

## Releasing

1. Update version in `src/eit_sim2real/__init__.py`
2. Update `CHANGELOG.md` (if maintained)
3. Tag: `git tag v0.x.y && git push --tags`

## Scientific Code

When modifying experiment code (`src/eit_sim2real/experiments/`):

- **Never change random seeds** without documenting the impact
- **Preserve exact numerical behaviour** — same inputs must produce same outputs
- **Test reproducibility** by running experiments twice and comparing results
- Add `# noqa` comments sparingly and only with justification
