import json

from regime_fusion import provenance
from regime_fusion.config import load_config


def test_record_contains_what_reproduction_needs(tmp_path):
    config = load_config()
    record = provenance.collect(config, stock="AAPL", test_year=2019, seed=123)
    for key in ("created_utc", "git", "python", "packages", "config_digest", "config"):
        assert key in record
    assert record["config_digest"] == config.digest()
    assert record["seed"] == 123
    assert "numpy" in record["packages"]

    path = tmp_path / "run" / "provenance.json"
    provenance.write(path, record)
    assert json.loads(path.read_text())["stock"] == "AAPL"


def test_git_is_asked_without_taking_its_index_lock(monkeypatch, tmp_path):
    import subprocess

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs.get("env", {})))
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n" if "rev-parse" in args else "", stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    assert provenance.git_state(tmp_path) == {"commit": "abc123", "dirty": False}
    assert [args[1] for args, _ in calls] == ["rev-parse", "status"]
    assert all(env.get("GIT_OPTIONAL_LOCKS") == "0" for _, env in calls)   # a stopped job leaves no lock
