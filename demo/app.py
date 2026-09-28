"""Gradio demo: score one transaction or a CSV, explain each decision, show the model version.

Everything model-related goes through `fraud.serving`, the same code the Phase 5 API will use.
The bundle is downloaded from the Hugging Face Hub at startup (or read from a local directory), so
the Space needs no DVC or MLflow credentials.

Run locally:  uv run python demo/app.py           (bundle from `serving.hf_model_repo`)
              FRAUD_MODEL_SOURCE=data/bundle uv run python demo/app.py   (local export)
"""

import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

# On the Space the `fraud` package is vendored next to this file (see demo/build_space.py).
_VENDORED_SRC = Path(__file__).resolve().parent / "src"
if _VENDORED_SRC.is_dir():
    sys.path.insert(0, str(_VENDORED_SRC))

from fraud.data.schemas import CATEGORIES  # noqa: E402
from fraud.params import load_params  # noqa: E402
from fraud.serving import (  # noqa: E402
    InputValidationError,
    ServingModel,
    explain,
    load_model,
    predict,
)
from fraud.serving.explain import reasons_frame  # noqa: E402
from fraud.serving.model import TEMPLATE_FILE  # noqa: E402
from fraud.serving.schema import TEMPLATE_ROWS, read_transactions_csv  # noqa: E402

try:  # ZeroGPU Spaces refuse to start without a @spaces.GPU function. The model runs on CPU, so
    import spaces  # this probe only satisfies that startup check; it is never called.

    @spaces.GPU
    def _zerogpu_probe() -> None:
        return None

except ImportError:
    pass

SETTINGS = load_params()["serving"]
CHART_COLORS = {"raises fraud score": "#d95f02", "lowers fraud score": "#1f77b4"}
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DOB_FORMAT = "%Y-%m-%d"
_example_rng = random.Random(load_params()["seed"])


def percent(p: float) -> str:
    return f"{p:.2%}" if p < 0.1 else f"{p:.1%}"


def version_line(model: ServingModel) -> str:
    return (
        f"Scored by `{model.metadata['model_name']}` v{model.model_version} "
        f"(`{model.metadata['step']}`), pipeline `{model.pipeline_version}`."
    )


def problems_markdown(title: str, err: InputValidationError) -> str:
    return f"### {title}\n" + "\n".join(f"- {p}" for p in err.problems)


def verdict_markdown(model: ServingModel, probability: float, flagged: bool) -> str:
    headline = "Flagged as likely fraud" if flagged else "Not flagged"
    return (
        f"### {headline}\n"
        f"**Fraud probability: {percent(probability)}** (flag threshold: {percent(model.threshold)})\n\n"
        f"{version_line(model)}"
    )


def score_single(
    model: ServingModel,
    amount: float | None,
    category: str | None,
    when: datetime | None,
    dob: datetime | None,
    gender: str | None,
    state: str | None,
    city_pop: float | None,
    lat: float | None,
    long: float | None,
    merch_lat: float | None,
    merch_long: float | None,
) -> tuple[str, pd.DataFrame | None, str]:
    """Verdict text, reasons chart data and reasons text for one transaction."""
    row = {
        "trans_ts": when,
        "amt": amount,
        "category": category,
        "gender": gender,
        "state": (state or "").strip().upper() or None,
        "city_pop": city_pop,
        "dob": dob,
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
    }
    try:
        frame = pd.DataFrame([row])
        scored = predict(model, frame)
        reasons = explain(model, frame)[0]
    except InputValidationError as err:
        return problems_markdown("Could not score this transaction", err), None, ""
    verdict = verdict_markdown(
        model, float(scored["fraud_probability"].iloc[0]), bool(scored["flagged"].iloc[0])
    )
    text = "**Why this score** (largest effect first)\n\n" + "\n".join(
        f"- {r.text}" for r in reasons
    )
    return verdict, reasons_frame(reasons), text


def score_batch(
    model: ServingModel, path: str | None
) -> tuple[str, pd.DataFrame | None, str | None]:
    """Summary text, results table (highest scores first) and a results CSV path for an upload."""
    if not path:
        return "### No file uploaded\nChoose a CSV in the documented schema.", None, None
    max_rows = SETTINGS["max_batch_rows"]
    try:
        frame = read_transactions_csv(path, max_rows)
        scored = predict(model, frame, max_rows=max_rows)
        flagged = scored[scored["flagged"]].sort_values("fraud_probability", ascending=False)
        to_explain = flagged.head(SETTINGS["batch_explain_rows"]).index
        reasons = explain(model, frame.loc[to_explain], top_k=1) if len(to_explain) else []
    except InputValidationError as err:
        return problems_markdown("Could not score this file", err), None, None

    top_reason = pd.Series("", index=frame.index)
    for idx, row_reasons in zip(to_explain, reasons, strict=True):
        top_reason[idx] = row_reasons[0].text
    results = frame.assign(
        fraud_probability=scored["fraud_probability"].round(6),
        flagged=scored["flagged"],
        top_reason=top_reason,
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", prefix="fraud_scores_", delete=False, encoding="utf-8", newline=""
    ) as out:
        results.to_csv(out, index=False)

    shown = (
        results.assign(row=range(1, len(results) + 1))
        .sort_values("fraud_probability", ascending=False)
        .head(SETTINGS["batch_display_rows"])[
            ["row", "amt", "category", "fraud_probability", "flagged", "top_reason"]
        ]
    )
    n, n_flagged = len(results), len(flagged)
    summary = (
        f"### Scored {n:,} transactions\n"
        f"**{n_flagged:,} flagged** ({n_flagged / n:.1%}) at threshold {percent(model.threshold)}; "
        f"{n - n_flagged:,} not flagged.\n\n"
        f"The table shows the {len(shown):,} highest scores. The top reason is computed for flagged "
        f"rows only (the {min(n_flagged, SETTINGS['batch_explain_rows']):,} highest). "
        f"The download has every row.\n\n{version_line(model)}"
    )
    return summary, shown, out.name


def home_fields(cities: pd.DataFrame, label: str | None) -> tuple[Any, ...]:
    """State, population and coordinates for a picked home city (unchanged if not found)."""
    match = cities[cities["label"] == label]
    if match.empty:
        return (gr.skip(),) * 4
    c = match.iloc[0]
    return c["state"], int(c["city_pop"]), float(c["lat"]), float(c["long"])


def merchant_fields(cities: pd.DataFrame, label: str | None) -> tuple[Any, ...]:
    match = cities[cities["label"] == label]
    if match.empty:
        return (gr.skip(),) * 2
    return float(match.iloc[0]["lat"]), float(match.iloc[0]["long"])


def example_fields(examples: pd.DataFrame, is_fraud: int) -> tuple[Any, ...]:
    """Form values for a random real test-set transaction of the requested class."""
    pool = examples[examples["is_fraud"] == is_fraud]
    r = pool.iloc[_example_rng.randrange(len(pool))]
    truth = "fraud" if is_fraud else "legitimate"
    note = (
        f"Real transaction from the held-out test set, sampled at random. "
        f"**Ground truth: {truth}.** The model has not seen this row."
    )
    return (
        float(r["amt"]),
        r["category"],
        pd.Timestamp(r["trans_ts"]).strftime(TS_FORMAT),
        pd.Timestamp(r["dob"]).strftime(DOB_FORMAT),
        r["gender"],
        f"{r['city']}, {r['state']}",
        r["state"],
        int(r["city_pop"]),
        float(r["lat"]),
        float(r["long"]),
        None,
        float(r["merch_lat"]),
        float(r["merch_long"]),
        note,
    )


def about_markdown(model: ServingModel) -> str:
    m, v = model.metadata, model.metadata["validation"]
    champ, ci = m["reference_champion_test"], m["reference_champion_test"]["confidence_intervals"]
    repo = SETTINGS["hf_model_repo"]
    return f"""## About this model

| | |
| --- | --- |
| Registered model | `{m["model_name"]}` version {model.model_version} (alias `{m["alias"]}`) |
| Algorithm | `{m["step"]}`, {m["calibration"]}-calibrated |
| Feature pipeline | `{model.pipeline_version}`, `{m["feature_set"]}` (stateless: uses only the fields on the form) |
| Dataset | `{m["dataset_name"]}` `{m["dataset_version"]}` (simulated card transactions, Jan 2019 - Dec 2020) |
| Split | {m["split_spec"]} |
| Decision threshold | {model.threshold:.6f} calibrated probability |
| Git commit | `{m["git_commit"]}` |

**How the threshold was chosen.** On the validation split, the highest-recall threshold that keeps
precision at or above {m["min_precision"]:.2f} (at most one false alarm per caught fraud). At that
threshold this model reached precision {v["precision"]:.3f} and recall {v["recall"]:.3f} on validation.

**Test-set results belong to a different model.** The project's single test evaluation was run on
the Phase 3 champion (`{champ["step"]}` with card-history features), which this stateless demo
cannot use: recall {champ["recall"]:.3f} (95% CI {ci["recall"][0]:.3f}-{ci["recall"][1]:.3f}), precision
{champ["precision"]:.3f} ({ci["precision"][0]:.3f}-{ci["precision"][1]:.3f}), which **misses the 0.50
precision target**. This demo model has not been scored on the test split.

**How the reasons work.** SHAP contributions to the model's raw score, summed per input feature and
shown beside the value you entered. They explain the score's direction and ranking, not its exact
probability.

## Limitations

- The data is **simulated**; real fraud is harder and these numbers will not carry over.
- The demo model sees one transaction at a time. It cannot use a card's recent spending pattern,
  which is what lifts recall from about {v["recall"]:.2f} to about {champ["recall"]:.2f} in the champion.
- Gender is a model input. The champion showed recall and false-alarm-rate gaps between groups
  (see its model card); this model has not had its own fairness check.
- A portfolio demonstration, not a system for real decisions.

## Links

- [Source code and docs]({SETTINGS["github_repo"]})
- [Model card and bundle](https://huggingface.co/{repo})
"""


def build_app(model: ServingModel) -> gr.Blocks:
    cities = model.read_table("cities.csv")
    examples = model.read_table("examples.csv")
    labels = cities["label"].tolist()
    first = TEMPLATE_ROWS[0]

    with gr.Blocks(title="Credit card fraud detector") as app:
        gr.Markdown(
            "# Credit card fraud detector\n"
            "Score a simulated card transaction, see why, and see which model version answered."
        )
        with gr.Tab("Single transaction"):
            with gr.Row():
                with gr.Column():
                    amount = gr.Number(label="Amount (USD)", value=first["amt"], minimum=0.01)
                    category = gr.Dropdown(
                        CATEGORIES, label="Merchant category", value=first["category"]
                    )
                    when = gr.DateTime(
                        label="Transaction date and time",
                        include_time=True,
                        type="datetime",
                        value=first["trans_ts"],
                    )
                    dob = gr.DateTime(
                        label="Customer date of birth",
                        include_time=False,
                        type="datetime",
                        value=first["dob"],
                    )
                    gender = gr.Radio(["F", "M"], label="Customer gender", value=first["gender"])
                    home_city = gr.Dropdown(
                        labels, label="Customer city (fills the details below)", filterable=True
                    )
                    merch_city = gr.Dropdown(
                        labels, label="Merchant city (fills the coordinates below)", filterable=True
                    )
                    with gr.Accordion(
                        "Location details (filled from the city, editable)", open=False
                    ):
                        state = gr.Textbox(label="Customer state (2 letters)", value=first["state"])
                        city_pop = gr.Number(
                            label="Customer city population", value=first["city_pop"]
                        )
                        lat = gr.Number(label="Customer latitude", value=first["lat"])
                        long = gr.Number(label="Customer longitude", value=first["long"])
                        merch_lat = gr.Number(label="Merchant latitude", value=first["merch_lat"])
                        merch_long = gr.Number(
                            label="Merchant longitude", value=first["merch_long"]
                        )
                    score_button = gr.Button("Score transaction", variant="primary")
                    with gr.Row():
                        fraud_example = gr.Button("Try a real fraud example")
                        legit_example = gr.Button("Try a real legitimate example")
                    example_note = gr.Markdown()
                with gr.Column():
                    verdict = gr.Markdown()
                    chart = gr.BarPlot(
                        None,
                        x="reason",
                        y="contribution",
                        color="effect",
                        color_map=CHART_COLORS,
                        x_title="Reason",
                        y_title="Effect on fraud score",
                        x_label_angle=-30,
                        sort="-y",
                        height=340,
                    )
                    reasons_text = gr.Markdown()

            form = [
                amount,
                category,
                when,
                dob,
                gender,
                state,
                city_pop,
                lat,
                long,
                merch_lat,
                merch_long,
            ]
            outputs = [verdict, chart, reasons_text]

            def score(*values: Any) -> tuple[str, pd.DataFrame | None, str]:
                return score_single(model, *values)

            home_city.input(
                lambda label: home_fields(cities, label),
                home_city,
                [state, city_pop, lat, long],
                api_name=False,
            )
            merch_city.input(
                lambda label: merchant_fields(cities, label),
                merch_city,
                [merch_lat, merch_long],
                api_name=False,
            )
            score_button.click(score, form, outputs, api_name="score_single")
            example_targets = [
                amount, category, when, dob, gender, home_city, state, city_pop, lat, long,
                merch_city, merch_lat, merch_long, example_note,
            ]  # fmt: skip
            for button, is_fraud in ((fraud_example, 1), (legit_example, 0)):
                button.click(
                    lambda is_fraud=is_fraud: example_fields(examples, is_fraud),
                    None,
                    example_targets,
                    api_name=f"example_{'fraud' if is_fraud else 'legitimate'}",
                ).then(score, form, outputs, api_name=False)

        with gr.Tab("Batch CSV"):
            max_rows = SETTINGS["max_batch_rows"]
            gr.Markdown(
                f"Upload a CSV with these columns: `trans_ts, amt, category, gender, state, city_pop, "
                f"dob, lat, long, merch_lat, merch_long` (up to {max_rows:,} rows; extra columns are "
                f"kept and ignored). Start from the template."
            )
            template_path = str(Path(model.bundle_dir or ".") / TEMPLATE_FILE)
            gr.DownloadButton("Download template CSV", value=template_path)
            upload = gr.File(label="Transactions CSV", file_types=[".csv"], type="filepath")
            batch_button = gr.Button("Score file", variant="primary")
            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(label="Results (highest scores first)", wrap=True)
            batch_download = gr.File(label="Download all results (CSV)")
            batch_button.click(
                lambda path: score_batch(model, path),
                upload,
                [batch_summary, batch_table, batch_download],
                api_name="score_batch",
            )

        with gr.Tab("About"):
            gr.Markdown(about_markdown(model))
    return app


if __name__ == "__main__":
    build_app(load_model(os.environ.get("FRAUD_MODEL_SOURCE"))).launch()
