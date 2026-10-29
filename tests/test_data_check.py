import json

from regime_fusion.data import check, cli, store
from regime_fusion.data.store import DATA_DIR_ENV


def write(root, name, content):
    path = store.layer("reports", name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def test_issues_are_gathered_per_stock_and_source(tmp_path):
    write(tmp_path, "price_checks.json", {
        "AAPL": {"missing_sessions": 0, "flags": [], "big_moves": [["2020-03-16", -0.13]]},
        "KODK": {"missing_sessions": 2, "flags": ["zero volume: 1"], "big_moves": [["2020-07-29", 3.18]]}})
    write(tmp_path, "sec_checks.json", {"AAPL": {"flags": []}, "KODK": {"flags": ["few filings in 2016"]}})
    write(tmp_path, "fa_coverage.json", {"AAPL": {"missing": []}, "KODK": {"missing": ["ebitda"]}})
    write(tmp_path, "ta_coverage.json", {"AAPL": {"missing": []}})
    write(tmp_path, "sa_coverage.json", {"AAPL": {"sessions_with_news": 0.97}, "KODK": {"sessions_with_news": 0.3},
                                         "_coverage": {"gdelt_days_used": 10, "gdelt_days_pending": 5,
                                                       "sessions_covered": 0.002}})
    reports = check.load_reports(tmp_path)
    assert not any(check.stock_issues("AAPL", reports).values())
    kodk = check.stock_issues("KODK", reports)
    assert kodk["prices"] == ["zero volume: 1", "2 sessions without a bar", "moves over 25%: 2020-07-29 +318%"]
    assert kodk["sec"] == ["few filings in 2016"] and kodk["fa"] == ["thin: ebitda"]
    assert kodk["ta"] == ["no coverage report"] and kodk["sa"] == ["news on 30% of sessions"]
    reports["sa"]["AAPL"]["sessions_with_news"] = float("nan")
    assert check.stock_issues("AAPL", reports)["sa"] == ["no session with news coverage yet"]
    text = check.markdown([{"stock": "KODK", "role": "main", "issues": kodk},
                           {"stock": "AAPL", "role": "main", "issues": {"prices": []}}], ["dc 2019 stage 1"],
                          reports["sa"]["_coverage"])
    assert "2 stocks checked; 1 with something to look at" in text and "| KODK | main |" in text
    assert "10 days extracted, 5 still to come" in text and "dc 2019 stage 1" in text


def test_check_command_writes_a_readable_report(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    write(tmp_path, "price_checks.json", {"AAPL": {"missing_sessions": 0, "flags": [], "big_moves": []}})
    assert cli.main(["check"]) == 0
    text = store.layer("reports", "data_check.md").read_text()
    assert text.startswith("# Data check") and "| AAPL | main | sec: no SEC checks" in text
    saved = json.loads(store.layer("reports", "data_check.json").read_text())
    assert len(saved["stocks"]) == 50 and saved["missing_regime_files"]
