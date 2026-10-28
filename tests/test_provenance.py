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
