"""Contract tests for CLI flags, exit codes, and output structures."""

from click.testing import CliRunner
from src.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Hiver Support Agent" in result.output
    assert "ingest" in result.output
    assert "audit-leakage" in result.output
    assert "handle" in result.output
    assert "evaluate" in result.output


def test_cli_handle_single_ticket():
    runner = CliRunner()
    result = runner.invoke(cli, [
        "handle",
        "--text", "My iPhone battery dies very quickly.",
        "--json"
    ])
    assert result.exit_code == 0
    assert "INT-BATTERY" in result.output
    assert "auto-handle" in result.output or "escalate" in result.output


def test_cli_evaluate_help_and_benchmark_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--benchmark" in result.output
    assert "--golden-set" in result.output
    assert "--human-manifest" in result.output
