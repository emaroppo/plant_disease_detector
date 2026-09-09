"""Land the labels this corpus arrived with, as candidates rather than answers.

    uv run python scripts/land_candidates.py --project projects/plants --apply

PlantVillage arrives labelled: the folder a file sits in is the answer for
every file inside it. The preparer carries that into ``prepared.json``, and
nothing moves it into the catalog on its own — landing annotations is a
separate, deliberate step, and what lands here goes in under
``source="import"`` because nobody has looked at these.

**This should not exist here.** Strata has no core ``import-annotations``;
the only landing command that ships is ``strata-seed-email``, whose selection
logic is specific to spans and to mail. This is the second corpus to arrive
labelled, which is the condition that TODO item was waiting on. What is
genuinely plugin-side is *which* candidates to land — the stratified seed a
review loop wants. What is not is everything else in this file: reading the
index, matching by checksum, and writing under a non-human source.

It also reaches past the public API to build the checksum map, because
``by_checksum`` answers one sample at a time and there are 54,284 of them.
A core command would not have to.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from strata.catalog import tables as t
from strata.catalog.blobs import checksum_of
from strata.catalog.prepared import PreparedIndex
from strata.labeller.cli import _catalog_for
from strata.labeller.config import Settings
from strata.labeller.project import Project


def checksum_map(catalog) -> dict[str, int]:
    """Every live sample's checksum to its id, in one query.

    The public route is ``by_checksum``, one round trip each. At this size
    that is a query per image and minutes of it.
    """
    stmt = select(t.sample.c.checksum, t.sample.c.id).where(t.sample.c.deleted_at.is_(None))
    with catalog.engine.connect() as conn:
        return {row.checksum: row.id for row in conn.execute(stmt)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--apply", action="store_true", help="Write; otherwise report only"
    )
    args = parser.parse_args(argv)

    project = Project.load(args.project)
    index = PreparedIndex.load(project.data_dir)
    if index is None:
        print(
            f"No prepared corpus under {project.data_dir}; run "
            f"'auto-labeller prepare' first.",
            file=sys.stderr,
        )
        return 1

    carrying = {name: e for name, e in index.samples.items() if e.value is not None}
    if not carrying:
        print("The prepared corpus carries no candidate labels; nothing to land.")
        return 0

    histogram: Counter = Counter()
    for entry in carrying.values():
        histogram.update(entry.value.values)
    print(f"{len(carrying):,} sample(s) carry a candidate label.")
    print(f"  across {len(histogram)} class(es), rarest first:")
    for name, count in sorted(histogram.items(), key=lambda kv: kv[1])[:5]:
        print(f"    {name:<40} {count:,}")

    if not args.apply:
        print("\nReport only. Pass --apply to write.")
        return 0

    settings = Settings.load(args.config)
    catalog, _root = _catalog_for(settings, args.config, name=project.catalog.name)
    label_set_id, _schema = catalog.label_set(project.label_set_name)

    print("\nBuilding the checksum map...", flush=True)
    by_checksum = checksum_map(catalog)
    print(f"  {len(by_checksum):,} sample(s) in the catalog")

    print("Matching the prepared corpus against it...", flush=True)
    items, missing = [], 0
    for name, entry in carrying.items():
        path = project.data_dir / name
        if not path.exists():
            missing += 1
            continue
        sample_id = by_checksum.get(checksum_of(path))
        if sample_id is None:
            missing += 1
            continue
        items.append((sample_id, entry.value))

    # Not "human": nobody has looked at these. That distinction is the only
    # thing separating a reviewed label from a folder name, and an export
    # overwrites it the moment somebody submits the task.
    annotated, skipped = catalog.annotate_many(label_set_id, items, source="import")
    print(f"\nLanded {annotated:,} annotation(s) as source='import'")
    if skipped:
        print(f"  {skipped:,} skipped")
    if missing:
        print(f"  {missing:,} prepared file(s) were not in the catalog — run ingest first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
