"""PlantVillage into a catalog: the labels it arrived with, and its leaves.

PlantVillage arrives as a directory of JPEGs, which is already the shape an
``image`` catalog holds — so unlike mail or video, nothing here needs
converting. What this conversion exists for is the two things a directory of
files cannot say on its own:

**The labels.** A folder named ``Tomato___Early_blight`` carries the answer
for every file inside it. Those are candidates, not answers — nothing here
reaches the catalog on its own, and landing them is a separate, deliberate
step under a source of its own.

**The leaves.** 40,490 of the 54,305 images are repeat shots of a leaf
already photographed, in groups averaging five and reaching thirty-three.
Split at random and near-duplicates of one leaf land on both sides, so a
validation score measures memorisation. The dataset's own authors hit this
and published the mapping they used; ``group_id`` is how a catalog is told
about it, and grouping is indivisible across a split.

The mapping is joined on **class and key together**, never the key alone.
2,152 keys are reused across classes — ``rs_hl 6251`` is both
``Soybean___healthy:::398.0`` and ``Apple___healthy:::101.0`` — and taking
the first entry would put an apple leaf and a soybean leaf in one group.
Where the key exists but under no matching class, the image is left
ungrouped rather than guessed at.

Reference: Mohanty, Hughes & Salathé (2016), *Using Deep Learning for
Image-Based Plant Disease Detection*, which reports 41,112 such images —
this joins 40,490 of them, the difference being the cross-class collisions
above, which it declines to resolve.
"""

import json
import shutil
from pathlib import Path
from typing import ClassVar, Iterable

from strata.catalog.types.preparers import Prepared, Preparer
from strata.labels import Choices

#: What the leaf mapping is called, looked for above the corpus. Named by
#: convention rather than configured because a preparer is constructed with
#: no arguments — see ``_leaf_map_for``.
LEAF_MAP_NAME = "leaf-map.json"

#: How far above an image to look for the mapping. Enough to reach the
#: corpus root from ``<root>/<class>/<file>`` with room for one more level,
#: and bounded so a missing file is not a walk to ``/``.
_SEARCH_DEPTH = 4

#: What separates species from disease in a class folder, and class from
#: leaf number in the mapping.
_CLASS_SEP = "___"
_LEAF_SEP = ":::"


class PlantVillagePreparer(Preparer):
    """PlantVillage's directory layout, read as labels and leaf groups."""

    name: ClassVar[str] = "plantvillage"
    produces: ClassVar[str] = "image"
    #: The corpus is 52,803 ``.JPG``, 1,500 ``.jpg``, one ``.png`` and one
    #: ``.jpeg``. Extensions are matched lowercased, so the case split costs
    #: nothing — but declaring only ``jpg`` would drop the last two with a
    #: warning rather than an error, and a corpus that arrives quietly
    #: smaller than its source is the failure worth avoiding.
    sources: ClassVar[frozenset[str]] = frozenset({"jpg", "jpeg", "png"})

    def __init__(self) -> None:
        # Keyed by the directory the mapping was found in, so a corpus is
        # read once rather than once per image.
        self._maps: dict[Path, dict[str, list[str]]] = {}
        self._unlabelled = 0
        self._ungrouped = 0

    # -- the conversion -------------------------------------------------

    def prepare(self, source: Path, out_dir: Path) -> Iterable[Prepared]:
        """One image in, the same image out, with what its folder knew.

        The bytes are copied rather than rewritten. A JPEG is already what
        an ``image`` catalog stores, so re-encoding would invent bytes for
        no reason — and re-encoding is not reproducible across library
        versions, which is the one thing a preparer may not be.
        """
        source = Path(source)
        folder = source.parent.name

        target_dir = Path(out_dir) / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        # Copied, not linked: the source is kept as the corpus arrived and
        # a hard link would make an edit to one an edit to both.
        shutil.copyfile(source, target)

        return [
            Prepared(
                path=target,
                metadata=self._metadata_for(folder, source),
                group_id=self._group_for(source, folder),
                value=self._value_for(folder),
            )
        ]

    def report(self) -> dict[str, int]:
        """What could not be read off the layout, counted rather than logged."""
        counts = {}
        if self._unlabelled:
            counts["images_in_an_unnamed_class_folder"] = self._unlabelled
        if self._ungrouped:
            counts["images_with_no_leaf_group"] = self._ungrouped
        return counts

    # -- reading the layout ---------------------------------------------

    def _split(self, folder: str) -> tuple[str, str] | None:
        """``Tomato___Early_blight`` into its two halves.

        Split once from the left: a disease may contain the separator's
        constituent characters and several contain spaces
        (``Cercospora_leaf_spot Gray_leaf_spot``), but the species never
        contains the separator itself.
        """
        if _CLASS_SEP not in folder:
            return None
        species, disease = folder.split(_CLASS_SEP, 1)
        if not species or not disease:
            return None
        return species, disease

    def _value_for(self, folder: str) -> Choices | None:
        """The candidate annotation, or nothing where the folder does not say.

        Both halves in one value, because they are one label set: the
        species is a fact about the sample that a model may be told, and
        the disease is what it is asked for. Which is which is the label
        set's business, not this one's.
        """
        parts = self._split(folder)
        if parts is None:
            self._unlabelled += 1
            return None
        return Choices(values=list(parts))

    def _metadata_for(self, folder: str, source: Path) -> dict:
        """What the layout knew, recorded on the sample at ingest."""
        metadata: dict = {"source_folder": folder}
        parts = self._split(folder)
        if parts is not None:
            metadata["species"], metadata["disease"] = parts
        return metadata

    # -- the leaf mapping -----------------------------------------------

    def _group_for(self, source: Path, folder: str) -> str | None:
        """Which leaf this is a photograph of, where that is known.

        Null means the sample is its own group, which is the honest answer
        for an image the mapping does not cover — and for one whose key is
        claimed by a different class.
        """
        mapping = self._leaf_map_for(source)
        if not mapping:
            self._ungrouped += 1
            return None

        candidates = [
            entry
            for entry in mapping.get(self._leaf_key(source), [])
            if entry.split(_LEAF_SEP, 1)[0] == folder
        ]
        if len(candidates) != 1:
            self._ungrouped += 1
            return None
        return candidates[0]

    def _leaf_key(self, source: Path) -> str:
        """``…___RS_Early.B 7557.JPG`` into ``rs_early.b 7557``.

        The mapping is keyed on the half of the filename after the UUID,
        lowercased. Filenames that do not carry one fall back to the whole
        stem, which simply will not match — a miss, not a wrong group.
        """
        stem = source.stem
        tail = stem.split(_CLASS_SEP, 1)[1] if _CLASS_SEP in stem else stem
        return tail.lower()

    def _leaf_map_for(self, source: Path) -> dict[str, list[str]]:
        """The mapping governing this image, found by walking up from it.

        Found rather than configured, and that is a limitation rather than
        a design: a preparer is constructed by name with no arguments, so
        there is nowhere for a project to say where its side files are. A
        model gets ``[model.params]``; a preparer gets nothing.
        """
        directory = source.parent
        for _ in range(_SEARCH_DEPTH):
            directory = directory.parent
            if directory in self._maps:
                return self._maps[directory]
            candidate = directory / LEAF_MAP_NAME
            if candidate.exists():
                self._maps[directory] = json.loads(candidate.read_text(encoding="utf-8"))
                return self._maps[directory]
            if directory == directory.parent:
                break
        return {}


__all__ = ["LEAF_MAP_NAME", "PlantVillagePreparer"]
