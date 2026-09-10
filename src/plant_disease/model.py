"""The disease classifier, as a registered model plugin.

The architecture is the one this project already had — three convolutions,
max pooling, two fully connected layers and heavy dropout — carried across
so the port is recognisably the same model rather than a new one that
happens to score better.

What changed is where it sits. It is registered through ``strata.models``
rather than referenced as a file, because the modelling service refuses any
reference containing ``:``: a host serving a round cannot resolve a path on
the caller's machine, so a file reference would confine every round to one
laptop.

**Two arms, and neither is the model this project wants.**

``flat``
    One head over every (species, disease) pair observed in training — the
    38-way composite the original PlantVillage paper used. Predicts both
    halves at once.

``disease``
    One head over the diseases alone, with no species input. The control:
    it is the same trunk as ``flat`` with strictly less information.

The arm this project actually wants — the species *supplied* at inference,
because whoever owns the plant knows what it is — cannot be expressed here.
:meth:`predict` receives paths and nothing else, and :class:`Example` carries
one target and no features, so there is no channel through which a known
attribute reaches the model. That is the gap this port exists to find, and
it is why ``flat`` and ``disease`` are the two arms and not three.

**Which classes are species has to be restated here.** The label set holds
all 35 names in one list with nothing marking the 14 that are species, so
the split arrives through ``[model.params] species``. Two copies of one fact
that can drift apart — the label set is where it belongs, and it has no way
to say it.
"""

from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from strata.labels import ChoicesPrediction, ClassificationSchema
from strata.modelling.model import BatchReport, EpochReport, Example, Model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# A handful of PlantVillage files are truncated; decode what is there rather
# than failing a whole round on one bad JPEG.
ImageFile.LOAD_TRUNCATED_IMAGES = True

#: Separates the halves of a composite label. Not the corpus's own '___',
#: so a composite is never mistaken for a folder name.
_PAIR_SEP = "\x1f"


class _Net(nn.Module):
    """The original trunk, with the head sized by whatever it is learning."""

    def __init__(self, outputs: int, image_size: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Three poolings, so the side is divided by eight. Computed rather
        # than hardcoded at 32: the original assumed a 256px input and
        # silently produced a shape error at any other size.
        side = max(image_size // 8, 1)
        self.fc1 = nn.Linear(64 * side * side, 64)
        self.fc2 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, outputs)
        self._flat = 64 * side * side

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self._flat)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


class _Images(Dataset):
    def __init__(self, paths, targets, transform):
        self.paths = list(paths)
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.paths[idx]).convert("RGB"))
        if self.targets is None:
            return image
        return image, self.targets[idx]


class PlantDiseaseClassifier(Model):
    """Disease classification over leaf images, in one of two arms."""

    task: ClassVar[str] = "classification"
    version: ClassVar[str] = "1"

    def __init__(
        self,
        arm: str = "disease",
        species: tuple[str, ...] = (),
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.75,
        image_size: int = 256,
        num_workers: int = 0,
        device: str | None = None,
    ):
        if arm not in ("flat", "disease"):
            raise ValueError(f"Unknown arm {arm!r}; expected 'flat' or 'disease'")
        self.arm = arm
        #: What the project asked for. Kept apart from :attr:`arm` because
        #: :meth:`load` adopts the checkpoint's arm — it has to, since a
        #: checkpoint is restored into a default-constructed model that
        #: cannot know what it was. :meth:`finetune` compares the two.
        self._requested_arm = arm
        self.species = tuple(species)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.image_size = image_size
        # Decoding and resizing 54,000 JPEGs is the actual cost of an epoch
        # here; the network is small enough that a single loader thread
        # starves it. Zero by default so the test suite stays in one process.
        self.num_workers = num_workers
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.classes: list[str] = []
        #: What the head's neurons mean, by position.
        self.vocab: list[str] = []
        self.net: _Net | None = None

    # -- what this model can be asked to learn --------------------------

    def requires_schema(self, schema) -> None:
        """Refuse a label set that cannot carry a species alongside a disease.

        Only when species are configured. A single-choice set permits one
        class per sample, so the species and the disease cannot both be
        present — and this model would train on whichever one arrived,
        reporting a number for it either way.
        """
        if not self.species:
            return
        if isinstance(schema, ClassificationSchema) and not schema.multiple:
            raise ValueError(
                "This label set is single-choice, but this model was given a "
                "species list, which means it expects each sample to assert "
                "both its species and its disease. Either drop [model.params] "
                "species, or declare the label set multi-choice."
            )

    # -- encoding -------------------------------------------------------

    def _split(self, values: list[str]) -> tuple[str | None, str | None]:
        """A target's species and disease halves, either of which may be absent."""
        known = set(self.species)
        found_species = next((v for v in values if v in known), None)
        target = next((v for v in values if v not in known), None)
        return found_species, target

    def _encode(self, values: list[str]) -> str | None:
        """The one thing this arm is learning to say about a sample."""
        found_species, target = self._split(values)
        if self.arm == "disease":
            return target
        if found_species is None:
            return target
        return f"{found_species}{_PAIR_SEP}{target}" if target else None

    def _decode(self, token: str) -> list[str]:
        """A head position back into the class names it asserts."""
        return token.split(_PAIR_SEP) if _PAIR_SEP in token else [token]

    # -- training -------------------------------------------------------

    def finetune(
        self,
        train: list[Example],
        classes: list[str],
        val: list[Example] | None = None,
        on_epoch: EpochReport | None = None,
    ) -> dict[str, float]:
        # Refused before the round rather than during it. A warm start
        # across a change of arm restores a head of the wrong size for the
        # job asked for, and every number the round reported afterwards
        # would be about the other arm.
        if self.arm != self._requested_arm:
            raise ValueError(
                f"This round asked for arm {self._requested_arm!r}, but warm-started "
                f"from a checkpoint trained as {self.arm!r}. The two learn different "
                f"heads — one class per (species, disease) pair against one per "
                f"disease — so the weights cannot carry over. Train fresh instead."
            )
        self.classes = list(classes)

        # Without a species list there is nothing separating the two halves
        # of a target, and _split would take whichever came first. On this
        # corpus that is the species, so the round trains a species
        # classifier, reports a confident number for it, and calls it a
        # disease model. Caught here rather than in requires_schema because
        # the label set permitting several classes is not the problem — a
        # sample actually carrying several, with no way to tell them apart,
        # is.
        if not self.species:
            ambiguous = sum(1 for e in train if len(e.target.values) > 1)
            if ambiguous:
                raise ValueError(
                    f"{ambiguous:,} of {len(train):,} training sample(s) assert more "
                    f"than one class, and no species were configured — so there is "
                    f"nothing to say which half is the target. Set "
                    f"[model.params] species."
                )

        encoded = [(e.path, self._encode(e.target.values)) for e in train]
        usable = [(p, t) for p, t in encoded if t is not None]
        # A sample whose target this arm cannot represent is dropped rather
        # than encoded as position zero, which is a real class.
        dropped = len(encoded) - len(usable)
        if not usable:
            raise ValueError(
                f"None of the {len(train)} training sample(s) carry a class this "
                f"arm can learn. Arm {self.arm!r} with "
                f"{len(self.species)} species name(s) configured."
            )

        if not self.vocab:
            self.vocab = sorted({t for _, t in usable})
        index = {token: i for i, token in enumerate(self.vocab)}
        # Keep a loaded network when it still fits what is being learned;
        # that is the whole of a warm start. Rebuilding here unconditionally
        # meant a warm round trained from scratch and said nothing about it,
        # which reads in the run store as a warm start that did not help.
        if self.net is None or self.net.fc3.out_features != len(self.vocab):
            self.net = _Net(len(self.vocab), self.image_size, self.dropout)
        self.net = self.net.to(self.device)

        loader = DataLoader(
            _Images([p for p, _ in usable], [index[t] for _, t in usable], self._transform()),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        optimiser = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        metrics: dict[str, float] = {}
        for epoch in range(self.epochs):
            self.net.train()
            total_loss = correct = seen = 0.0
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                optimiser.zero_grad()
                outputs = self.net(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimiser.step()

                total_loss += loss.item() * targets.size(0)
                correct += (outputs.argmax(1) == targets).sum().item()
                seen += targets.size(0)

            metrics = {
                "loss": total_loss / max(seen, 1),
                "accuracy": correct / max(seen, 1),
                "train_samples": float(len(usable)),
                "classes": float(len(self.vocab)),
            }
            if dropped:
                metrics["train_targets_dropped"] = float(dropped)
            if on_epoch is not None:
                on_epoch(epoch + 1, self.epochs, metrics)

        if val:
            metrics.update(self._validate(val))
        return metrics

    def _validate(self, val: list[Example]) -> dict[str, float]:
        encoded = [(e.path, self._encode(e.target.values)) for e in val]
        usable = [(p, t) for p, t in encoded if t is not None and t in set(self.vocab)]
        if not usable:
            return {}
        index = {token: i for i, token in enumerate(self.vocab)}
        loader = DataLoader(
            _Images([p for p, _ in usable], [index[t] for _, t in usable], self._transform()),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.net.eval()
        correct = seen = 0.0
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                correct += (self.net(images).argmax(1) == targets).sum().item()
                seen += targets.size(0)
        return {"val_accuracy": correct / max(seen, 1), "val_samples": float(len(usable))}

    # -- prediction -----------------------------------------------------

    def predict(
        self, paths: list[Path], on_batch: BatchReport | None = None
    ) -> list[ChoicesPrediction]:
        if not paths:
            return []
        if self.net is None:
            raise ValueError("This model has not been trained or loaded")

        loader = DataLoader(
            _Images(paths, None, self._transform()),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.net.eval()
        out: list[ChoicesPrediction] = []
        with torch.no_grad():
            for images in loader:
                probabilities = torch.softmax(self.net(images.to(self.device)), dim=1)
                best = probabilities.argmax(1)
                for row, position in zip(probabilities, best):
                    names = self._decode(self.vocab[int(position)])
                    confidence = float(row[int(position)])
                    # Positional: a composite asserts two names on one
                    # decision, so both carry that decision's confidence.
                    out.append(
                        ChoicesPrediction(values=names, confidences=[confidence] * len(names))
                    )
                if on_batch is not None:
                    on_batch(len(out), len(paths))
        return out

    # -- checkpoints ----------------------------------------------------

    def save(self, path: Path) -> None:
        """Everything needed to rebuild this, including what it was told.

        The species list and the arm travel with the weights: a checkpoint
        loaded without its project.toml beside it would otherwise decode
        head positions against a different vocabulary.
        """
        torch.save(
            {
                "state_dict": self.net.state_dict() if self.net else None,
                "classes": self.classes,
                "vocab": self.vocab,
                "arm": self.arm,
                "species": list(self.species),
                "image_size": self.image_size,
                "dropout": self.dropout,
                "version": type(self).version,
            },
            path,
        )

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.classes = payload["classes"]
        self.vocab = payload["vocab"]
        self.arm = payload["arm"]
        self.species = tuple(payload["species"])
        self.image_size = payload["image_size"]
        self.dropout = payload["dropout"]
        self.net = _Net(len(self.vocab), self.image_size, self.dropout).to(self.device)
        if payload["state_dict"] is not None:
            self.net.load_state_dict(payload["state_dict"])
        self.net.eval()

    # -- helpers --------------------------------------------------------

    def _transform(self):
        return transforms.Compose(
            [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()]
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(arm={self.arm!r}, "
            f"species={len(self.species)}, classes={len(self.vocab)})"
        )


__all__ = ["PlantDiseaseClassifier"]
