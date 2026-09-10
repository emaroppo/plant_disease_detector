# Plant disease detection, on a labelling catalog

A leaf photograph and a plant's species in; a diagnosis out.

The classifier is not the point. This exists to answer a question about
**strata**, a semi-automatic labelling pipeline and sample catalog built
separately: *can a project that was not written alongside it use it?* So
this repository is deliberately an outsider. It brings its own corpus, its
own data converter and its own model, and registers all of them through
strata's public plugin surfaces.

Most of it worked untouched. Where it did not, that was the answer: the
port could not tell a model something it already knew about a sample, so
strata gained a way to say it.

It turned up 23 findings, six of which became fixes in strata — four of
them bugs that only appear at scale, and one a whole missing capability.
They are written up in [`docs/pick-up-here.md`](docs/pick-up-here.md) and
are the more interesting output.

## Results

54,284 images, 43,427 train / 10,857 held out, ten epochs, one run each.

| Arm | Head | Species | val accuracy |
|---|---|---|---|
| A | 38 species–disease pairs | folded into the label | 0.8454 |
| B | 21 diseases | absent | 0.8984 |
| C | 21 diseases | **supplied at inference** | **0.9202** |

**B → C is the comparison that means something: +2.18 points.** Same head,
same hyperparameters, both trained cold, and the same 10,857 held-out
samples — the second dataset version inherited the first's split exactly,
which is checked rather than assumed. The only difference is whether the
model was told the species.

A belongs on its own axis: it chooses among 38 and the others among 21, so
the gap to it is arithmetic rather than learning.

**Why arm C exists.** Whoever owns a plant knows what it is; what they need
is a diagnosis. Making the model rediscover the species from the image is
solving a problem the user does not have. Mohanty et al. (2016) measured
the same idea on out-of-distribution photographs and found 31.4% → 47.93%.
Here, in-distribution, the model can largely tell an apple leaf from a
tomato one by itself, so being told is worth about two points instead of
sixteen. Both numbers describe the same feature.

**Caveats, stated rather than buried.** One run per arm, no seeds, no
repeats — +2.18 is an observation, not an interval. PlantVillage is a
laboratory dataset with known background bias, and ~99% on a held-out split
is the published result for a much larger model; the numbers here are a
three-layer CNN trained from scratch for ten epochs. None of this transfers
to a photograph taken in a garden.

## What is actually being demonstrated

**Two plugins, registered from outside.** A preparer that reads
PlantVillage's directory layout, and a model — both declared in this
repository's own `pyproject.toml` and discovered through entry points.

**Near-duplicates cannot straddle the split.** 40,490 of the images are
repeat shots of a leaf already photographed, in groups of up to 33. Split at
random and a model is scored on leaves it has memorised. The preparer reads
the dataset authors' own leaf mapping — joined on class *and* key, because
2,152 keys are reused across classes and taking the first match merges an
apple leaf with a soybean one — and the catalog guarantees a group never
straddles train and validation. Verified: **zero groups straddling**.

**A feature is a role, not a fact about the data.** Species lives in its own
label set. That set is a species-identification project's *target* and this
project's *feature*, at the same time, over one catalog; only the
declaration differs:

```toml
[[data.features]]
name = "species"
source = "label_set"
ref = "plant-species"
```

## Running it

```bash
uv sync --extra cli --extra model
uv run python scripts/download_corpus.py --project projects/plants
uv run auto-labeller prepare -p projects/plants
uv run auto-labeller ingest  -p projects/plants --config config.toml
uv run python scripts/land_candidates.py --project projects/plants --config config.toml --apply
uv run python scripts/seed_species_label_set.py --project projects/plants --config config.toml --apply
uv run auto-labeller train   -p projects/plants --config config.toml
```

No database, no object store and no Label Studio: the committed
`config.toml` is a SQLite index and a directory of blobs. Switch arms with
`[model.params] arm` — `flat`, `disease` or `conditioned`.

```bash
uv run streamlit run dashboard.py
```

The dashboard reads `auto-labeller report --json` rather than the run store,
and refuses to draw a line between two runs that were asked different
questions.

## The corpus

[PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage), the
colour variant: 54,305 images, 38 classes, 14 crop species.
**CC-BY-SA-3.0.** Downloaded rather than vendored — share-alike is a real
constraint, and the licence is stated inconsistently across mirrors.

> Mohanty, S.P., Hughes, D.P., Salathé, M. (2016). *Using Deep Learning for
> Image-Based Plant Disease Detection.* Frontiers in Plant Science 7:1419.

## Layout

```
src/plant_disease/
    preparer.py        the PlantVillage layout, as labels and leaf groups
    model.py           the classifier, in three arms
    comparability.py   which runs may share an axis
projects/plants/       the labelling job: label set, features, model params
scripts/               fetch the corpus, land its labels, seed the species set
dashboard.py           rounds, curves and the dataset
docs/pick-up-here.md   what the port found
```

## History

`legacy` holds this project as it was before the port — a Streamlit app, a
training script and checkpoints written beside the code. It is kept because
the comparison is the point: the same model, the same data, and everything
around it replaced by infrastructure that can say what it trained on.
