import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud.serving import InputValidationError, explain, load_model, predict
from fraud.serving.explain import make_reason, reasons_frame, source_feature
from fraud.serving.model import EXAMPLES_FILE, TEMPLATE_FILE
from fraud.serving.schema import (
    REQUIRED_COLUMNS,
    read_transactions_csv,
    template_frame,
    validate_transactions,
)


def test_smoke_load_bundle_and_score_template(bundle_dir: Path) -> None:
    """The CI smoke test: load a bundle, score the template CSV it ships."""
    model = load_model(bundle_dir)
    template = pd.read_csv(bundle_dir / TEMPLATE_FILE)
    scored = predict(model, template)

    assert list(template.columns) == REQUIRED_COLUMNS
    assert len(scored) == len(template)
    assert scored["fraud_probability"].between(0, 1).all()
    assert scored["flagged"].tolist() == (scored["fraud_probability"] >= model.threshold).tolist()
    assert model.model_version == "test-1"
    assert model.pipeline_version == "features-test"


def test_predict_matches_pipeline_and_ignores_batching(bundle_dir: Path) -> None:
    model = load_model(bundle_dir)
    df = pd.read_csv(bundle_dir / EXAMPLES_FILE).drop(columns="is_fraud")
    batch = predict(model, df)["fraud_probability"].to_numpy()
    single = np.concatenate(
        [predict(model, df.iloc[[i]])["fraud_probability"].to_numpy() for i in range(len(df))]
    )
    # The v1 pipeline is stateless: a row scores identically alone or in a batch.
    np.testing.assert_allclose(batch, single)
    direct = model.pipeline.predict_proba(validate_transactions(df))[:, 1]
    np.testing.assert_allclose(batch, direct)


def test_explain_returns_top_k_reasons_per_row(bundle_dir: Path) -> None:
    model = load_model(bundle_dir)
    template = template_frame()
    reasons = explain(model, template, top_k=3)

    assert len(reasons) == len(template)
    for row in reasons:
        assert len(row) == 3
        magnitudes = [abs(r.contribution) for r in row]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert {r.direction for r in row} <= {"raises", "lowers"}
        # one entry per source feature: one-hot columns are folded together
        assert len({r.feature for r in row}) == 3
    assert all(r.value and r.label for row in reasons for r in row)


def test_explain_contributions_sum_to_the_model_score(bundle_dir: Path) -> None:
    model = load_model(bundle_dir)
    template = template_frame()
    everything = explain(model, template, top_k=100)
    from fraud.features.pipeline import transform_with_context

    X = transform_with_context(model.feature_pipeline, validate_transactions(template))
    raw_score = model.classifier.estimator.predict(X, output_margin=True)
    base = float(np.atleast_1d(model.explainer.expected_value)[0])
    for i, row in enumerate(everything):
        assert base + sum(r.contribution for r in row) == pytest.approx(raw_score[i], abs=1e-3)


def test_source_feature_folds_one_hot_columns() -> None:
    assert source_feature("category_shopping_net") == "category"
    assert source_feature("state_TX") == "state"
    assert source_feature("gender_F") == "gender"
    assert source_feature("log_amt") == "log_amt"
    assert source_feature("is_night") == "is_night"


def test_make_reason_shows_unscaled_values() -> None:
    assert make_reason("log_amt", 0.4, np.log1p(412.5)).text == (
        "Amount = $412.50 raises the fraud score"
    )
    assert make_reason("hour", -0.2, 2.0).value == "02:00"
    assert make_reason("distance_km", 0.1, 812.4).value == "812 km"
    frame = reasons_frame([make_reason("category", -0.3, "travel")])
    assert frame.to_dict("records") == [
        {
            "reason": "Merchant category: travel",
            "contribution": -0.3,
            "effect": "lowers fraud score",
        }
    ]


def test_validation_accepts_the_template_and_ignores_extra_columns() -> None:
    df = template_frame().assign(is_fraud=0, card_id="abc")
    out = validate_transactions(df)
    assert pd.api.types.is_datetime64_any_dtype(out["trans_ts"])
    assert pd.api.types.is_datetime64_any_dtype(out["dob"])
    assert out.index.equals(df.index)


def test_validation_reports_readable_problems() -> None:
    df = template_frame()
    df.loc[0, "amt"] = -5
    df.loc[1, "category"] = "yachts"
    df.loc[1, "trans_ts"] = "not a date"
    df = df.drop(columns="gender")
    with pytest.raises(InputValidationError) as err:
        validate_transactions(df)
    text = str(err.value)
    assert "Missing required column 'gender'" in text
    assert "Column 'amt' must be greater than 0 (rows 1): -5.0." in text
    assert "Column 'category' has unexpected values (rows 2): 'yachts'. Allowed: " in text
    assert "Column 'trans_ts' has values that are not valid (rows 2): 'not a date'." in text
    assert text.count("trans_ts") == 1  # one message per problem, not one per failed check
    assert "TypeError" not in text and "isin(" not in text


def test_validation_rejects_birth_after_transaction() -> None:
    df = template_frame()
    df.loc[0, "dob"] = "2021-01-01"
    with pytest.raises(InputValidationError, match="Date of birth must be earlier"):
        validate_transactions(df)


def test_validation_row_cap_and_empty_input() -> None:
    with pytest.raises(InputValidationError, match="Too many rows"):
        validate_transactions(template_frame(), max_rows=1)
    with pytest.raises(InputValidationError, match="no rows"):
        validate_transactions(template_frame().iloc[:0])


def test_read_csv_stops_at_the_row_cap() -> None:
    csv = template_frame().to_csv(index=False)
    assert len(read_transactions_csv(io.StringIO(csv), max_rows=5)) == 2
    capped = read_transactions_csv(io.StringIO(csv), max_rows=1)
    assert len(capped) == 2  # one past the cap, so validate_transactions can refuse it
    with pytest.raises(InputValidationError, match="Too many rows"):
        validate_transactions(capped, max_rows=1)


def test_load_model_rejects_incomplete_metadata(bundle_dir: Path, tmp_path: Path) -> None:
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(bundle_dir, broken)
    (broken / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_model(broken)
