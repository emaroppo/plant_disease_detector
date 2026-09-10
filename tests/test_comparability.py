"""The rule that decides which runs may share an axis."""

from plant_disease.comparability import (
    asked_of,
    differing_params,
    group_runs,
    question_of,
)


def _run(**params):
    return {"model": "plant-disease", "params": params}


def test_training_knobs_do_not_change_the_question():
    """Two rounds of different length are still asking the same thing."""
    a = _run(arm="disease", epochs=10, batch_size=64, lr=0.001)
    b = _run(arm="disease", epochs=40, batch_size=256, lr=0.01)

    assert question_of(a) == question_of(b)
    assert len(group_runs([a, b])) == 1


def test_the_arm_does_change_it():
    """38 composite pairs against 21 diseases is not one series."""
    flat = _run(arm="flat", epochs=10)
    disease = _run(arm="disease", epochs=10)

    assert question_of(flat) != question_of(disease)
    assert len(group_runs([flat, disease])) == 2
    assert differing_params([flat, disease]) == {"arm"}


def test_image_size_is_part_of_the_question():
    """A resize is preprocessing, and two pipelines are not one measurement."""
    assert question_of(_run(arm="flat", image_size=64)) != question_of(
        _run(arm="flat", image_size=256)
    )


def test_an_unknown_parameter_is_assumed_to_matter():
    """Conservative by design: guessing the other way invents progress."""
    assert question_of(_run(arm="flat", something_new=1)) != question_of(
        _run(arm="flat", something_new=2)
    )


def test_a_run_with_no_params_falls_back_to_its_model():
    assert question_of({"model": "multilabel", "params": {}}) == "multilabel"
    assert question_of({"model": "multilabel"}) == "multilabel"


def test_a_list_parameter_is_summarised_rather_than_spelled_out():
    """A fourteen-name species list is part of the question, not a legend."""
    label = question_of(_run(arm="flat", species=["a", "b", "c"]))

    assert "3 item(s)" in label
    assert "'a'" not in label


def test_lists_of_different_length_are_different_questions():
    assert question_of(_run(species=["a", "b"])) != question_of(_run(species=["a"]))


def test_asked_of_drops_only_the_training_knobs():
    assert asked_of({"arm": "flat", "epochs": 3, "lr": 0.1}) == {"arm": "flat"}


def test_nothing_differs_across_identical_runs():
    a = _run(arm="flat", epochs=1)
    b = _run(arm="flat", epochs=99)
    assert differing_params([a, b]) == set()


def test_grouping_keeps_the_order_it_was_given():
    runs = [_run(arm="flat"), _run(arm="disease"), _run(arm="flat")]
    groups = group_runs(runs)
    assert [len(v) for v in groups.values()] == [2, 1]
