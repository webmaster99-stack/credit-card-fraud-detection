from pathlib import Path

import app as demo_app  # demo/app.py; `demo` is on pytest's pythonpath
import pandas as pd
import pytest

from fraud.serving import load_model
from fraud.serving.model import TEMPLATE_FILE
from fraud.serving.schema import TEMPLATE_ROWS


@pytest.fixture(scope="module")
def model(bundle_dir: Path):
    return load_model(bundle_dir)


def _form(**overrides: object) -> dict:
    first = TEMPLATE_ROWS[0]
    form = {
        "amount": first["amt"],
        "category": first["category"],
        "when": pd.Timestamp(first["trans_ts"]).to_pydatetime(),
        "dob": pd.Timestamp(first["dob"]).to_pydatetime(),
        "gender": first["gender"],
        "state": first["state"],
        "city_pop": first["city_pop"],
        "lat": first["lat"],
        "long": first["long"],
        "merch_lat": first["merch_lat"],
        "merch_long": first["merch_long"],
    }
    return form | overrides


def test_build_app_wires_three_tabs(model) -> None:
    app = demo_app.build_app(model)
    tabs = [b.label for b in app.blocks.values() if type(b).__name__ == "Tab"]
    assert tabs == ["Single transaction", "Batch CSV", "About"]


def test_score_single_returns_verdict_chart_and_reasons(model) -> None:
    verdict, chart, reasons = demo_app.score_single(model, **_form())

    assert "Fraud probability" in verdict
    assert "test-1" in verdict and "features-test" in verdict  # which model answered
    assert len(chart) == 5 and set(chart["effect"]) <= set(demo_app.CHART_COLORS)
    assert reasons.count("\n- ") == 5 and "the fraud score" in reasons


def test_score_single_explains_bad_input_in_plain_words(model) -> None:
    verdict, chart, reasons = demo_app.score_single(
        model, **_form(amount=-3, category="yachts", when=None)
    )
    assert verdict.startswith("### Could not score this transaction")
    assert "Column 'amt'" in verdict and "'yachts'" in verdict
    assert chart.empty and reasons == ""


def test_score_single_lowercase_state_is_accepted(model) -> None:
    verdict, _, _ = demo_app.score_single(model, **_form(state=" il "))
    assert "Fraud probability" in verdict


def test_score_batch_scores_the_template_and_writes_a_download(model, bundle_dir: Path) -> None:
    summary, table, download = demo_app.score_batch(model, str(bundle_dir / TEMPLATE_FILE))

    assert "Scored 2 transactions" in summary
    assert list(table.columns) == [
        "row",
        "amt",
        "category",
        "fraud_probability",
        "flagged",
        "top_reason",
    ]
    assert table["fraud_probability"].is_monotonic_decreasing
    saved = pd.read_csv(download)
    assert len(saved) == 2
    assert {"fraud_probability", "flagged", "top_reason"} <= set(saved.columns)
    # every flagged row gets a reason; unflagged rows do not
    reason_present = saved["top_reason"].notna()
    assert reason_present.tolist() == saved["flagged"].tolist()


def test_score_batch_reports_problems_and_missing_file(model, tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"amt": [1.0]}).to_csv(bad, index=False)
    summary, table, download = demo_app.score_batch(model, str(bad))
    assert summary.startswith("### Could not score this file")
    assert "Missing required column" in summary
    assert table is None and download is None

    assert demo_app.score_batch(model, None)[0].startswith("### No file uploaded")


def test_score_batch_enforces_the_row_cap(model, bundle_dir: Path, monkeypatch) -> None:
    monkeypatch.setitem(demo_app.SETTINGS, "max_batch_rows", 1)
    summary, table, _ = demo_app.score_batch(model, str(bundle_dir / TEMPLATE_FILE))
    assert "Too many rows" in summary and table is None


def test_example_buttons_load_real_rows_of_the_requested_class(model) -> None:
    examples = model.read_table("examples.csv")
    for is_fraud, word in ((1, "fraud"), (0, "legitimate")):
        values = demo_app.example_fields(examples, is_fraud)
        assert len(values) == 14  # 13 form fields + the note
        assert f"Ground truth: {word}" in values[-1]
        assert values[5] in set(model.read_table("cities.csv")["label"])  # home city is pickable


def test_city_pickers_fill_their_fields(model) -> None:
    cities = model.read_table("cities.csv")
    label = cities["label"].iloc[0]
    state, pop, lat, long = demo_app.home_fields(cities, label)
    assert state == cities["state"].iloc[0] and isinstance(pop, int)
    assert (lat, long) == demo_app.merchant_fields(cities, label)


def test_about_tab_states_where_test_metrics_come_from(model) -> None:
    text = demo_app.about_markdown(model)
    assert "fraud-classifier" in text and "version test-1" in text
    assert "Test-set results belong to a different model" in text
    assert "misses the 0.50" in text
