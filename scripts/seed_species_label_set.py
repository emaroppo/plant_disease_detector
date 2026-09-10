"""Give species a label set of its own, so it can be a feature.

    uv run python scripts/seed_species_label_set.py --project projects/plants --apply

Species is currently half of a target: the `plants` label set holds 14
species and 21 diseases together, and a sample asserts one of each. That
was the only way to carry it before features existed.

It belongs in a set of its own. A label set is *a question asked about a
corpus*, and "which plant is this" is a different question from "what is
wrong with it" — one this project happens to already know the answer to,
and a species-identification project would be trying to learn. Split, the
same annotation is that project's target and this one's feature, and only
the declaration differs.

Landed under ``source="import"``: these came off a folder name, and nobody
has looked at them.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from strata.catalog import tables as t
from strata.labels import Choices, ClassificationSchema
from strata.labeller.cli import _catalog_for
from strata.labeller.config import Settings
from strata.labeller.project import Project

SPECIES_SET = "plant-species"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    project = Project.load(args.project)
    settings = Settings.load(args.config)
    catalog, _root = _catalog_for(settings, args.config, name=project.catalog.name)

    with catalog.engine.connect() as conn:
        rows = conn.execute(
            select(t.sample.c.id, t.sample.c.metadata).where(
                t.sample.c.deleted_at.is_(None)
            )
        ).all()

    found = {sid: (meta or {}).get("species") for sid, meta in rows}
    covered = {sid: name for sid, name in found.items() if name}
    classes = sorted(set(covered.values()))

    print(f"{len(rows):,} live sample(s)")
    print(f"  {len(covered):,} carry a species, {len(classes)} distinct")
    if len(covered) < len(rows):
        # The completeness rule, answered rather than assumed. A feature
        # that does not cover the project's collections leaves samples that
        # can never be scored, and therefore never surface for review.
        print(f"  {len(rows) - len(covered):,} do NOT — they will not be scorable")
    for name, n in Counter(covered.values()).most_common(3):
        print(f"    {name:<28} {n:,}")

    if not args.apply:
        print("\nReport only. Pass --apply to write.")
        return 0
    if not classes:
        print("Nothing to land.", file=sys.stderr)
        return 1

    try:
        label_set_id, _ = catalog.label_set(SPECIES_SET)
        print(f"\nLabel set {SPECIES_SET!r} already exists")
    except Exception:
        label_set_id = catalog.create_label_set(
            SPECIES_SET, ClassificationSchema(classes=classes, multiple=False)
        )
        print(f"\nCreated label set {SPECIES_SET!r} — single-choice, {len(classes)} classes")

    items = [(sid, Choices(values=[name])) for sid, name in covered.items()]
    annotated, skipped = catalog.annotate_many(label_set_id, items, source="import")
    print(f"Landed {annotated:,} annotation(s) as source='import'")
    if skipped:
        print(f"  {skipped:,} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
