"""What the rounds did, read through the CLI rather than the run store.

    uv run streamlit run dashboard.py

**It shells out to `auto-labeller report --json`.** Reaching into `RunStore`
directly would be shorter and would couple this to a schema that is nobody's
published interface. The JSON is the seam an eval package would inherit, so
it is the seam this is written against — and using it is also the only way
to find out whether it carries enough, which it turned out to (`params` and
`classes` are what the comparability grouping below needs).

**The one thing this refuses to do is draw a line between two runs that are
not answering the same question.** A project's `[model.params]` can change
what a metric is *of* — this one's `arm` picks between a head over 38
(species, disease) pairs and a head over 21 diseases — and the run store has
no notion of that. `report`'s own table shows the two arms as 0.8454 then
0.8984 with a delta column, which reads as five points of progress and is
arithmetic. So runs are split into series by what they were asked to do, and
a series is never joined to another.
"""

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from plant_disease.comparability import differing_params, question_of

PROJECT = Path("projects/plants")
MANIFEST = PROJECT / "datasets" / "plants" / "v001" / "manifest.json"

# Slots 1-3 of the reference categorical palette, in order and unmodified.
# Validated light, all-pairs: worst CVD ΔE 9.2, worst normal-vision ΔE 24.0.
# The aqua sits at 2.74:1 on this surface, which is why every chart here also
# ships its numbers as a table — that relief is required, not optional.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"

# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------


def _cli(*args: str) -> dict:
    exe = shutil.which("auto-labeller")
    if exe is None:
        st.error("`auto-labeller` is not on PATH. Install with `uv sync --extra cli`.")
        st.stop()
    done = subprocess.run([exe, *args], capture_output=True, text=True)
    if done.returncode != 0:
        st.error(f"`auto-labeller {' '.join(args)}` failed:\n\n```\n{done.stderr}\n```")
        st.stop()
    return json.loads(done.stdout)


@st.cache_data(show_spinner=False)
def history(metric: str) -> dict:
    return _cli("report", "-p", str(PROJECT), "--metric", metric, "--json")


@st.cache_data(show_spinner=False)
def detail(run_id: str) -> dict:
    return _cli("report", "-p", str(PROJECT), "--run", run_id, "--json")


@st.cache_data(show_spinner=False)
def manifest() -> dict | None:
    if not MANIFEST.exists():
        return None
    return json.loads(MANIFEST.read_text())


# ----------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------


def _axis(title: str | None, fmt: str | None = None):
    """Recessive grid and axes, in muted ink rather than series colour.

    ``format`` and ``title`` are only passed when set: Altair validates
    against its schema, where these are ``str`` and not ``str | None``, so
    handing it an explicit None is a schema error rather than a default.
    """
    options: dict = {
        "labelColor": MUTED,
        "titleColor": MUTED,
        "gridColor": GRID,
        "domainColor": GRID,
        "tickColor": GRID,
    }
    if title is not None:
        options["title"] = title
    if fmt is not None:
        options["format"] = fmt
    return alt.Axis(**options)


def history_chart(frame: pd.DataFrame, metric: str) -> alt.Chart:
    """One line per question, never across them.

    Points as well as a line because a series here is often two or three
    rounds long, and a line between two points implies a trend that two
    points cannot show.
    """
    base = alt.Chart(frame).encode(
        x=alt.X("order:O", axis=_axis("Round within its series")),
        y=alt.Y(f"{metric}:Q", axis=_axis(metric, ".3f"), scale=alt.Scale(zero=False)),
        color=alt.Color(
            "question:N",
            scale=alt.Scale(range=SERIES),
            legend=alt.Legend(title="Asked to predict", labelColor=INK, titleColor=MUTED),
        ),
        tooltip=[
            alt.Tooltip("run:N", title="Run"),
            alt.Tooltip("question:N", title="Asked"),
            alt.Tooltip(f"{metric}:Q", title=metric, format=".4f"),
            alt.Tooltip("version:N", title="Dataset"),
            alt.Tooltip("lineage:N", title="Lineage"),
        ],
    )
    line = base.mark_line(strokeWidth=2)
    dots = base.mark_point(size=90, filled=True, stroke="#fcfcfb", strokeWidth=2)
    return (line + dots).properties(height=320)


def curve_chart(frame: pd.DataFrame, column: str) -> alt.Chart:
    """One measure, one axis.

    Loss and accuracy live on different scales and get separate charts —
    a second y-axis would let any pair of lines be made to cross wherever
    the reader's eye happened to land.
    """
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True))
        .encode(
            x=alt.X("epoch:O", axis=_axis("Epoch")),
            y=alt.Y(f"{column}:Q", axis=_axis(column, ".3f"), scale=alt.Scale(zero=False)),
            color=alt.value(SERIES[0]),
            tooltip=[
                alt.Tooltip("epoch:O", title="Epoch"),
                alt.Tooltip(f"{column}:Q", title=column, format=".4f"),
            ],
        )
        .properties(height=260)
    )


def bar_chart(frame: pd.DataFrame, value: str, label: str, title: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4, color=SERIES[0])
        .encode(
            x=alt.X(f"{value}:Q", axis=_axis(title)),
            y=alt.Y(f"{label}:N", sort="-x", axis=_axis(None)),
            tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=",")],
        )
    )


# ----------------------------------------------------------------------
# the page
# ----------------------------------------------------------------------

st.set_page_config(page_title="Plant disease — rounds", layout="wide")
st.title("Plant disease rounds")
st.caption(
    "Read through `auto-labeller report --json`. Runs are grouped by what they "
    "were asked to predict; series are never joined."
)

runs_tab, curve_tab, data_tab = st.tabs(["Rounds", "One round", "The dataset"])


with runs_tab:
    metric = st.selectbox("Metric", ["val_accuracy", "accuracy", "loss"], index=0)
    payload = history(metric)
    runs = payload["runs"]

    rows = []
    order: dict[str, int] = {}
    for run in runs:
        question = question_of(run)
        order[question] = order.get(question, 0) + 1
        rows.append(
            {
                "run": run["short"],
                "question": question,
                metric: run["value"],
                "delta": run["delta"],
                "version": f"v{run['dataset_version']}" if run["dataset_version"] else "—",
                "lineage": run["lineage"],
                "classes": len(run["classes"]),
                "order": order[question],
                "id": run["id"],
            }
        )
    frame = pd.DataFrame(rows)

    if frame.empty:
        st.info("No runs recorded yet. Run `auto-labeller train`.")
    else:
        questions = sorted(frame["question"].unique())
        if len(questions) > 1:
            differed = ", ".join(f"`{p}`" for p in sorted(differing_params(runs))) or "—"
            st.warning(
                f"**{len(questions)} incomparable groups of run.** These were asked "
                f"different questions, so their numbers do not belong on one axis and "
                f"no line joins them:\n\n"
                + "\n".join(f"- `{q}`" for q in questions)
                + f"\n\nThey differ on: {differed}. The run store keeps this in "
                "`params` and never interprets it — `report`'s own table draws a "
                "delta column straight across the lot."
            )
        st.altair_chart(history_chart(frame, metric), width="stretch")

        # Required relief for the palette's contrast warning, and the
        # non-colour path to the same information.
        st.dataframe(
            frame.drop(columns=["order", "id"]),
            width="stretch",
            hide_index=True,
        )


with curve_tab:
    payload = history("val_accuracy")
    labels = {r["short"]: r["id"] for r in payload["runs"]}
    if not labels:
        st.info("No runs recorded yet.")
    else:
        chosen = st.selectbox("Round", list(labels))
        one = detail(labels[chosen])
        run, curve = one["run"], one["curve"]

        left, right = st.columns([2, 1])
        with left:
            st.subheader(run["short"])
            st.caption(
                f"{run['model']} v{run['model_version']} · dataset "
                f"{run['dataset']} v{run['dataset_version']} · asked: "
                f"`{question_of(run)}`"
            )
        with right:
            st.metric("val_accuracy", f"{run['metrics'].get('val_accuracy', float('nan')):.4f}")

        if not curve:
            st.info(
                "No per-epoch curve for this round. Rounds trained before the store "
                "began recording one have only their final numbers — a model that "
                "never calls `on_epoch` records nothing, which is the honest answer."
            )
        else:
            cframe = pd.DataFrame(curve)
            plottable = [c for c in cframe.columns if c != "epoch" and cframe[c].nunique() > 1]
            for column in plottable:
                st.markdown(f"**{column}**")
                st.altair_chart(curve_chart(cframe, column), width="stretch")
            st.dataframe(cframe, width="stretch", hide_index=True)

        with st.expander("Everything the run recorded"):
            st.json(run)


with data_tab:
    m = manifest()
    if m is None:
        st.info(f"No manifest at {MANIFEST}. Run `auto-labeller train` first.")
    else:
        samples = m["samples"]
        train = [s for s in samples if not s["val"]]
        val = [s for s in samples if s["val"]]

        sides: dict[str, set] = {}
        for s in samples:
            if s["group_id"] is not None:
                sides.setdefault(s["group_id"], set()).add(s["val"])
        straddling = [g for g, v in sides.items() if len(v) > 1]

        a, b, c, d = st.columns(4)
        a.metric("Samples", f"{len(samples):,}")
        b.metric("Train / val", f"{len(train):,} / {len(val):,}")
        c.metric("Leaf groups", f"{len(sides):,}")
        d.metric(
            "Groups straddling the split",
            f"{len(straddling):,}",
            delta="intact" if not straddling else "LEAKED",
            delta_color="normal" if not straddling else "inverse",
        )
        st.caption(
            f"Val ratio asked {m['val_ratio']}, achieved {m['val_ratio_achieved']:.4f}. "
            "A group is every photograph of one leaf; 40,490 of these images are "
            "repeat shots of a leaf already photographed, so a random split would "
            "score the model on what it has memorised."
        )

        counts: dict[str, int] = {}
        for s in samples:
            for name in (s.get("value") or {}).get("values", []):
                counts[name] = counts.get(name, 0) + 1
        if counts:
            st.subheader("Class balance")
            cframe = pd.DataFrame(
                sorted(counts.items(), key=lambda kv: -kv[1]), columns=["class", "samples"]
            )
            st.altair_chart(
                bar_chart(cframe, "samples", "class", "Samples").properties(
                    height=24 * len(cframe)
                ),
                width="stretch",
            )
            st.dataframe(cframe, width="stretch", hide_index=True)

        st.subheader("How many shots of one leaf")
        per_group = Counter(s["group_id"] for s in samples if s["group_id"] is not None)
        histogram = Counter(per_group.values())
        ungrouped = sum(1 for s in samples if s["group_id"] is None)
        sizes = pd.DataFrame(sorted(histogram.items()), columns=["shots of one leaf", "leaves"])
        st.altair_chart(
            bar_chart(sizes, "leaves", "shots of one leaf", "Leaves").properties(
                height=max(24 * len(sizes), 120)
            ),
            width="stretch",
        )
        st.caption(
            f"{ungrouped:,} image(s) carry no leaf mapping and are each their own "
            "group — the honest answer where the corpus does not say, rather than "
            "a guess that would put two unrelated leaves together."
        )
        st.dataframe(sizes, width="stretch", hide_index=True)
