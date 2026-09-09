"""The model against strata's contract, in both arms, plus what is ours."""

import pytest
from PIL import Image
from strata.labels import Choices, ClassificationSchema
from strata.modelling.conformance import ModelContract
from strata.modelling.model import Example

from plant_disease.model import PlantDiseaseClassifier

SPECIES = ("Tomato", "Potato")
PAIRS = [
    ("Tomato", "Early_blight"),
    ("Tomato", "healthy"),
    ("Potato", "Late_blight"),
    ("Potato", "healthy"),
]


def _image(path, colour):
    Image.new("RGB", (48, 48), colour).save(path, "JPEG")
    return path


def _examples(tmp_path, with_species: bool):
    out = []
    for i, (species, disease) in enumerate(PAIRS):
        path = _image(tmp_path / f"{i}.jpg", (i * 60 % 256, 128, 200 - i * 40))
        values = [species, disease] if with_species else [disease]
        out.append(Example(path=path, target=Choices(values=values)))
    return out


def _model(**kwargs):
    defaults = dict(epochs=1, image_size=32, batch_size=2, device="cpu")
    return PlantDiseaseClassifier(**{**defaults, **kwargs})


class TestDiseaseArm(ModelContract):
    """The control: no species anywhere, so a plain single-class label set."""

    @pytest.fixture
    def model(self):
        return _model(arm="disease")

    @pytest.fixture
    def examples(self, tmp_path):
        return _examples(tmp_path, with_species=False)


class TestFlatArm(ModelContract):
    """The composite: every target asserts a species and a disease."""

    @pytest.fixture
    def model(self):
        return _model(arm="flat", species=SPECIES)

    @pytest.fixture
    def examples(self, tmp_path):
        return _examples(tmp_path, with_species=True)


# -- the arms differ in what they say ----------------------------------


def test_the_flat_arm_predicts_both_halves(tmp_path):
    examples = _examples(tmp_path, with_species=True)
    classes = list(SPECIES) + sorted({d for _, d in PAIRS})
    model = _model(arm="flat", species=SPECIES)
    model.finetune(examples, classes)

    for prediction in model.predict([e.path for e in examples]):
        assert len(prediction.values) == 2
        assert prediction.values[0] in SPECIES
        assert prediction.values[1] not in SPECIES
        # Positional, and one decision stands behind both names
        assert len(prediction.confidences) == 2


def test_the_disease_arm_never_mentions_a_species(tmp_path):
    examples = _examples(tmp_path, with_species=True)
    classes = list(SPECIES) + sorted({d for _, d in PAIRS})
    model = _model(arm="disease", species=SPECIES)
    model.finetune(examples, classes)

    for prediction in model.predict([e.path for e in examples]):
        assert len(prediction.values) == 1
        assert prediction.values[0] not in SPECIES


def test_the_flat_arm_learns_one_head_position_per_pair(tmp_path):
    examples = _examples(tmp_path, with_species=True)
    classes = list(SPECIES) + sorted({d for _, d in PAIRS})
    model = _model(arm="flat", species=SPECIES)
    model.finetune(examples, classes)

    assert len(model.vocab) == len(PAIRS)


def test_the_disease_arm_collapses_the_pairs(tmp_path):
    """Two species sharing 'healthy' is one class here and two in the flat arm."""
    examples = _examples(tmp_path, with_species=True)
    classes = list(SPECIES) + sorted({d for _, d in PAIRS})
    model = _model(arm="disease", species=SPECIES)
    model.finetune(examples, classes)

    assert len(model.vocab) == len({d for _, d in PAIRS})


# -- refusals and edges -------------------------------------------------


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="Unknown arm"):
        PlantDiseaseClassifier(arm="sideways")


def test_a_single_choice_label_set_is_refused_when_species_are_expected():
    """Both halves cannot be present if the set permits only one class."""
    model = _model(arm="flat", species=SPECIES)
    with pytest.raises(ValueError, match="single-choice"):
        model.requires_schema(ClassificationSchema(classes=["Tomato"], multiple=False))


def test_a_single_choice_label_set_is_fine_without_species():
    model = _model(arm="disease")
    model.requires_schema(ClassificationSchema(classes=["healthy"], multiple=False))


def test_training_on_nothing_it_can_represent_is_refused(tmp_path):
    """Species-only targets leave the disease arm with no class to learn."""
    path = _image(tmp_path / "only.jpg", (10, 20, 30))
    examples = [Example(path=path, target=Choices(values=["Tomato"]))]
    model = _model(arm="disease", species=SPECIES)

    with pytest.raises(ValueError, match="carry a class this arm can learn"):
        model.finetune(examples, ["Tomato"])


def test_the_checkpoint_carries_the_arm_and_the_species(tmp_path):
    """A default-constructed model must come back as the one that was saved."""
    examples = _examples(tmp_path, with_species=True)
    classes = list(SPECIES) + sorted({d for _, d in PAIRS})
    model = _model(arm="flat", species=SPECIES)
    model.finetune(examples, classes)
    checkpoint = tmp_path / "checkpoint.pt"
    model.save(checkpoint)

    restored = PlantDiseaseClassifier(device="cpu")
    assert restored.arm == "disease" and restored.species == ()
    restored.load(checkpoint)

    assert restored.arm == "flat"
    assert restored.species == SPECIES
    assert restored.image_size == 32


# -- warm starts --------------------------------------------------------


def _classes():
    return list(SPECIES) + sorted({d for _, d in PAIRS})


def test_a_warm_start_keeps_the_loaded_network(tmp_path):
    """Rebuilding here would train from scratch and call it a warm round."""
    examples = _examples(tmp_path, with_species=True)
    model = _model(arm="flat", species=SPECIES)
    model.finetune(examples, _classes())
    checkpoint = tmp_path / "checkpoint.pt"
    model.save(checkpoint)

    warm = _model(arm="flat", species=SPECIES)
    warm.load(checkpoint)
    loaded = warm.net
    warm.finetune(examples, _classes())

    assert warm.net is loaded, "finetune rebuilt the net, discarding the warm start"


def test_warm_starting_across_a_change_of_arm_is_refused(tmp_path):
    """The heads are different sizes, so the weights cannot carry over."""
    examples = _examples(tmp_path, with_species=True)
    flat = _model(arm="flat", species=SPECIES)
    flat.finetune(examples, _classes())
    checkpoint = tmp_path / "checkpoint.pt"
    flat.save(checkpoint)

    other = _model(arm="disease", species=SPECIES)
    other.load(checkpoint)
    with pytest.raises(ValueError, match="warm-started"):
        other.finetune(examples, _classes())


def test_a_cold_model_of_the_same_arm_still_trains(tmp_path):
    """The guard must not fire when nothing was loaded."""
    examples = _examples(tmp_path, with_species=True)
    model = _model(arm="disease", species=SPECIES)
    assert isinstance(model.finetune(examples, _classes()), dict)
