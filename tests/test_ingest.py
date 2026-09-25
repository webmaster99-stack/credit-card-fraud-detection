from pathlib import Path

from fraud.data.ingest import missing_files


def test_missing_files(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x")
    assert missing_files(tmp_path, ["a.csv", "b.csv"]) == ["b.csv"]
