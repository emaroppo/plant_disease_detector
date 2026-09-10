"""The dashboard actually runs, against the real project.

Streamlit's own harness executes the script the way a browser session does,
so an exception in any tab surfaces here rather than as a red box nobody
sees until they open it.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("auto-labeller") is None,
    reason="needs the CLI on PATH — `uv sync --extra cli`",
)


@pytest.fixture(scope="module")
def app():
    # Absolute: relative paths resolve against this file, not the cwd, and
    # the script reads the project relative to the cwd it is run from.
    script = Path(__file__).resolve().parent.parent / "dashboard.py"
    at = AppTest.from_file(str(script), default_timeout=120)
    at.run()
    return at


def test_it_runs_without_raising(app):
    assert not app.exception, [e.message for e in app.exception]


def test_it_names_the_project(app):
    assert any("Plant disease" in t.value for t in app.title)


def test_it_warns_that_the_two_arms_are_not_one_series(app):
    """The whole point. If this stops firing, the chart started lying."""
    warnings = " ".join(w.value for w in app.warning)
    assert "incomparable" in warnings
    assert "arm" in warnings


def test_it_draws_the_charts(app):
    """History, class balance, and the leaf-group histogram."""
    assert len(app.get("vega_lite_chart")) >= 3


def test_every_chart_ships_its_numbers_as_a_table(app):
    """Required relief: one series colour is below 3:1 on this surface.

    The palette validator WARNs that the aqua slot sits at 2.74:1 against
    this surface, and that warning obliges visible labels or a table view.
    A table per chart is the relief, so there is never fewer of one.
    """
    assert len(app.dataframe) >= len(app.get("vega_lite_chart"))


def test_the_split_is_reported_as_intact(app):
    """The claim the whole demo rests on, asserted rather than described."""
    straddling = [m for m in app.metric if "straddling" in m.label]
    assert straddling and straddling[0].value == "0"
