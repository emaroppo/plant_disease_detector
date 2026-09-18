"""The preparer against strata's contract, plus what is specific to this corpus."""

import json

import pytest
from strata.prepare import run
from strata.prepare.conformance import PreparerContract

from plant_disease.preparer import LEAF_MAP_NAME, PlantVillagePreparer

CLASS = "Tomato___Early_blight"
FILENAME = "0a1b2c3d-4e5f-6789-abcd-ef0123456789___RS_Early.B 7557.JPG"

#: A real JPEG, so the type admitting it is not the only thing being tested:
#: a 1x1 baseline image, which is the smallest thing a decoder will accept.
JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffc2000b080001000101011100ffc40014000100000000000000000000000000000"
    "009ffda0008010100000110ffd9"
)


def _corpus(root, class_name=CLASS, filename=FILENAME, leaf_map=None):
    """A corpus root holding one class folder, and optionally a leaf map."""
    folder = root / class_name
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / filename
    image.write_bytes(JPEG)
    if leaf_map is not None:
        (root / LEAF_MAP_NAME).write_text(json.dumps(leaf_map), encoding="utf-8")
    return image


class TestPlantVillagePreparer(PreparerContract):
    @pytest.fixture
    def preparer(self):
        return PlantVillagePreparer()

    @pytest.fixture
    def source(self, tmp_path):
        return _corpus(tmp_path / "corpus")


# -- what the folder says ----------------------------------------------


def test_it_reads_species_and_disease_off_the_folder(tmp_path):
    source = _corpus(tmp_path / "corpus")
    index = run(PlantVillagePreparer(), [source], tmp_path / "out")

    (entry,) = index.samples.values()
    assert entry.value.values == ["Tomato", "Early_blight"]
    assert entry.metadata["species"] == "Tomato"
    assert entry.metadata["disease"] == "Early_blight"


def test_a_disease_may_contain_spaces(tmp_path):
    source = _corpus(tmp_path / "corpus", class_name="Corn_(maize)___Cercospora_leaf_spot Gray")
    index = run(PlantVillagePreparer(), [source], tmp_path / "out")

    (entry,) = index.samples.values()
    assert entry.value.values == ["Corn_(maize)", "Cercospora_leaf_spot Gray"]


def test_a_folder_that_does_not_name_a_class_carries_no_candidate(tmp_path):
    preparer = PlantVillagePreparer()
    source = _corpus(tmp_path / "corpus", class_name="loose_images")
    index = run(preparer, [source], tmp_path / "out")

    (entry,) = index.samples.values()
    assert entry.value is None
    assert preparer.report()["images_in_an_unnamed_class_folder"] == 1


def test_the_class_folder_is_preserved_in_the_output(tmp_path):
    """Filenames repeat across classes, so a flat output would collide."""
    source = _corpus(tmp_path / "corpus")
    index = run(PlantVillagePreparer(), [source], tmp_path / "out")

    assert list(index.samples) == [f"{CLASS}/{FILENAME}"]


def test_the_bytes_are_carried_unchanged(tmp_path):
    source = _corpus(tmp_path / "corpus")
    out = tmp_path / "out"
    run(PlantVillagePreparer(), [source], out)

    assert (out / CLASS / FILENAME).read_bytes() == JPEG


@pytest.mark.parametrize("suffix", [".JPG", ".jpg", ".jpeg", ".png"])
def test_every_extension_the_corpus_actually_contains_is_admitted(tmp_path, suffix):
    source = _corpus(tmp_path / "corpus", filename=f"a___RS_Early.B 1{suffix}")
    assert PlantVillagePreparer().allows(source)


# -- the leaf grouping -------------------------------------------------


def test_it_groups_repeat_shots_of_one_leaf(tmp_path):
    leaf_map = {"rs_early.b 7557": [f"{CLASS}:::115.0"]}
    source = _corpus(tmp_path / "corpus", leaf_map=leaf_map)
    index = run(PlantVillagePreparer(), [source], tmp_path / "out")

    (entry,) = index.samples.values()
    assert entry.metadata.get("leaf") == f"{CLASS}:::115.0"


def test_a_key_claimed_by_another_class_is_left_ungrouped(tmp_path):
    """2,152 keys are reused across classes in the real mapping.

    Taking the first entry would put an apple leaf and a soybean leaf in
    one group, and grouping is what decides the train/val split.
    """
    preparer = PlantVillagePreparer()
    leaf_map = {"rs_early.b 7557": ["Soybean___healthy:::398.0", "Apple___healthy:::101.0"]}
    source = _corpus(tmp_path / "corpus", leaf_map=leaf_map)
    index = run(preparer, [source], tmp_path / "out")

    (entry,) = index.samples.values()
    assert entry.metadata.get("leaf") is None
    assert preparer.report()["images_with_no_leaf_group"] == 1


def test_a_key_the_mapping_does_not_cover_is_its_own_group(tmp_path):
    source = _corpus(tmp_path / "corpus", leaf_map={"something else": [f"{CLASS}:::1.0"]})
    index = run(PlantVillagePreparer(), [source], tmp_path / "out")

    (entry,) = index.samples.values()
    assert entry.metadata.get("leaf") is None


def test_two_shots_of_one_leaf_share_a_group(tmp_path):
    root = tmp_path / "corpus"
    leaf_map = {"rs_early.b 7557": [f"{CLASS}:::115.0"], "rs_early.b 7554": [f"{CLASS}:::115.0"]}
    first = _corpus(root, filename="aaa___RS_Early.B 7557.JPG", leaf_map=leaf_map)
    second = _corpus(root, filename="bbb___RS_Early.B 7554.JPG")

    index = run(PlantVillagePreparer(), [first, second], tmp_path / "out")
    groups = {entry.metadata.get("leaf") for entry in index.samples.values()}

    assert groups == {f"{CLASS}:::115.0"}


def test_no_mapping_at_all_leaves_everything_ungrouped(tmp_path):
    source = _corpus(tmp_path / "corpus")
    index = run(PlantVillagePreparer(), [source], tmp_path / "out")

    (entry,) = index.samples.values()
    assert entry.metadata.get("leaf") is None
