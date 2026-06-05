# scripts/run_pipeline.ps1
# Local CI-equivalent pipeline for Windows development
# Usage: .\scripts\run_pipeline.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== EIT Sim-to-Real Local Pipeline ===" -ForegroundColor Cyan

# 1. Lint
Write-Host "`n[1/4] Linting..." -ForegroundColor Yellow
ruff check src/ tests/
if ($LASTEXITCODE -ne 0) { Write-Host "LINT FAILED" -ForegroundColor Red; exit 1 }
ruff format --check src/ tests/
if ($LASTEXITCODE -ne 0) { Write-Host "FORMAT CHECK FAILED" -ForegroundColor Red; exit 1 }
Write-Host "  OK" -ForegroundColor Green

# 2. Type check
Write-Host "`n[2/4] Type checking..." -ForegroundColor Yellow
mypy src/eit_sim2real/ --ignore-missing-imports
if ($LASTEXITCODE -ne 0) { Write-Host "TYPECHECK FAILED" -ForegroundColor Red; exit 1 }
Write-Host "  OK" -ForegroundColor Green

# 3. Tests
Write-Host "`n[3/4] Running tests..." -ForegroundColor Yellow
pytest tests/ --cov=eit_sim2real --cov-report=term-missing -q
if ($LASTEXITCODE -ne 0) { Write-Host "TESTS FAILED" -ForegroundColor Red; exit 1 }
Write-Host "  OK" -ForegroundColor Green

# 4. CLI smoke test
Write-Host "`n[4/4] CLI smoke test..." -ForegroundColor Yellow
eit --version
if ($LASTEXITCODE -ne 0) { Write-Host "CLI FAILED" -ForegroundColor Red; exit 1 }
Write-Host "  OK" -ForegroundColor Green

Write-Host "`n=== All checks passed ===" -ForegroundColor Green
