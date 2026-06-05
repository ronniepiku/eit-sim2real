"""Tests for the Click CLI entry points."""

from __future__ import annotations

from click.testing import CliRunner

from eit_sim2real.cli import cli


class TestCLIRoot:
    """Tests for the root CLI group."""

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "EIT Sim-to-Real" in result.output

    def test_commands_listed(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "train" in result.output
        assert "evaluate" in result.output
        assert "experiments" in result.output
        assert "validate-dataset" in result.output
        assert "log-environment" in result.output


class TestTrainCLI:
    """Tests for the train command group."""

    def test_train_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "cnn" in result.output
        assert "baselines" in result.output

    def test_train_cnn_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "cnn", "--help"])
        assert result.exit_code == 0
        assert "--noise" in result.output
        assert "--epochs" in result.output
        assert "--output-dir" in result.output

    def test_train_baselines_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "baselines", "--help"])
        assert result.exit_code == 0
        assert "--noise" in result.output
        assert "--output-dir" in result.output


class TestEvaluateCLI:
    """Tests for the evaluate command."""

    def test_evaluate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model-path" in result.output

    def test_evaluate_missing_model_path(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()


class TestExperimentsCLI:
    """Tests for the experiments command group."""

    def test_experiments_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["experiments", "--help"])
        assert result.exit_code == 0
        assert "run-all" in result.output
        assert "ablation" in result.output
        assert "additional" in result.output
        assert "hyperopt" in result.output
        assert "architecture-sweep" in result.output

    def test_ablation_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["experiments", "ablation", "--help"])
        assert result.exit_code == 0


class TestValidateCLI:
    """Tests for the validate-dataset command."""

    def test_validate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate-dataset", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output


class TestEnvironmentCLI:
    """Tests for the log-environment command."""

    def test_environment_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["log-environment", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
