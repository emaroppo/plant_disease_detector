"""When two runs are answering the same question, and when they are not.

A run records its dataset version and its model version, so a metric that
moved can be attributed to the data or to the model. It does not record what
the model was *asked to do* — that arrives as ``[model.params]``, an opaque
dict the store keeps and never interprets.

For most projects that is harmless, because the params are epochs and a
learning rate. For this one it is not: ``arm`` decides between a head over
38 (species, disease) pairs and a head over 21 diseases, and the two are
scored on different alternatives. ``report`` puts them in one table with a
delta column, and 0.8454 followed by 0.8984 reads as five points of
progress. It is arithmetic.

So this splits runs into groups that may be compared, and the rule is
deliberately **conservative**: a parameter is assumed to change the question
unless it is one of a named few that plainly cannot. Assuming the other way
is what produces a chart that reads as improvement.

None of this belongs here in the long run. It is a consumer reconstructing
something the producer knows and does not say, and the honest fix is
upstream — a run recording what its params meant, or a project declaring
which of them are part of the question.
"""

from typing import Any, Iterable

#: Parameters that change how long or how fast a round trained, not what it
#: was asked to predict.
#:
#: ``image_size`` is deliberately absent. A resize is preprocessing, and two
#: models trained on differently prepared tensors are not scored on the same
#: thing — the same argument ``docs/proposals.md`` §1 makes for giving a
#: pipeline a version of its own.
TRAINING_ONLY = frozenset({"epochs", "batch_size", "num_workers", "device", "lr", "weight_decay"})


def asked_of(params: dict[str, Any] | None) -> dict[str, Any]:
    """The part of a run's params that decides what it was asked to predict."""
    return {k: v for k, v in (params or {}).items() if k not in TRAINING_ONLY}


def question_of(run: dict[str, Any]) -> str:
    """A label for what this run was asked to do.

    Two runs belong on one axis only if this matches. List-valued params are
    summarised by length rather than spelled out: a fourteen-name species
    list is part of the question, but printing it makes an unreadable legend
    and two lists of different length differ anyway.
    """
    asked = asked_of(run.get("params"))
    if not asked:
        return run.get("model") or "model"
    parts = []
    for key, value in sorted(asked.items()):
        shown = f"{len(value)} item(s)" if isinstance(value, (list, tuple)) else value
        parts.append(f"{key}={shown}")
    return ", ".join(parts)


def group_runs(runs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Runs by question, each list in the order it was given.

    A dict rather than a filter, because the caller needs to know there was
    more than one group — that is the thing worth saying out loud.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(question_of(run), []).append(run)
    return groups


def differing_params(runs: Iterable[dict[str, Any]]) -> set[str]:
    """Which question-bearing params are not the same across these runs.

    What to name when telling someone why their runs were split up. Saying
    "these are incomparable" without saying on what is a dead end.
    """
    seen: dict[str, set[str]] = {}
    for run in runs:
        for key, value in asked_of(run.get("params")).items():
            seen.setdefault(key, set()).add(repr(value))
    return {key for key, values in seen.items() if len(values) > 1}


__all__ = [
    "TRAINING_ONLY",
    "asked_of",
    "differing_params",
    "group_runs",
    "question_of",
]
