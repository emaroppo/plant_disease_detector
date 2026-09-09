"""Fetch PlantVillage into the project's source directory.

    uv run python scripts/download_corpus.py --project projects/plants

The corpus is not vendored. It is 54,305 images under **CC-BY-SA-3.0**, and
share-alike is a real constraint on redistribution — fetching it sidesteps
the question entirely, and keeps a 900MB corpus out of a repository that is
otherwise a few hundred kilobytes.

**Where it comes from.** HuggingFace ``mohanty/PlantVillage``, uploaded by
the first author of the paper the dataset was published with. Zenodo record
1204914 is the other canonical source and states its licence inconsistently
— CC-BY-4.0 in the rights field, CC-BY-SA-3.0-US in the description — where
the HuggingFace copy states one thing plainly.

**The leaf mapping matters as much as the images.** 40,490 of them are
repeat shots of a leaf already photographed, and a split that separates two
photographs of one leaf scores a model on what it has memorised. The mapping
that says which is which ships alongside the corpus, and the preparer looks
for it by name one level above the images.

Cite: Mohanty, Hughes & Salathé (2016), *Using Deep Learning for Image-Based
Plant Disease Detection*, Frontiers in Plant Science 7:1419.
"""

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main"
ARCHIVE = "data.zip"
LEAF_MAP = "leaf_grouping/leaf-map.json"

#: Which of the three variants to keep. Grayscale and segmented are the same
#: leaves processed differently, so taking more than one would need grouping
#: across variants for no gain this demo needs.
VARIANT = "color"

#: Roughly, for the progress line. The archive holds all three variants.
ARCHIVE_MB = 2185


def fetch(url: str, target: Path) -> None:
    """Download to a temporary name, then rename.

    A partial file left at the real path is indistinguishable from a
    complete one on the next run, and the next run would skip it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    print(f"  {url.rsplit('/', 1)[-1]} -> {target}")
    try:
        with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.URLError as e:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"Could not fetch {url}: {e}") from None
    partial.rename(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("projects/plants"),
        help="Project whose data/ directory the corpus lands in",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help=f"Keep the {ARCHIVE_MB}MB archive after extracting",
    )
    args = parser.parse_args(argv)

    data = args.project / "data"
    source = data / "source"
    download = data / "download"
    leaf_map = data / "leaf-map.json"

    if source.exists() and any(source.iterdir()):
        print(f"{source} is not empty — nothing to do.")
        print("Delete it to re-fetch.")
        return 0

    print(f"Fetching PlantVillage ({ARCHIVE_MB}MB, CC-BY-SA-3.0)")
    if not leaf_map.exists():
        fetch(f"{REPO}/{LEAF_MAP}", leaf_map)
    else:
        print(f"  leaf-map.json already at {leaf_map}")

    archive = download / ARCHIVE
    if not archive.exists():
        fetch(f"{REPO}/{ARCHIVE}", archive)
    else:
        print(f"  {ARCHIVE} already at {archive}")

    print(f"\nExtracting the '{VARIANT}' variant...")
    prefix = f"raw/{VARIANT}/"
    staging = data / "extract"
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.startswith(prefix)]
        if not members:
            raise SystemExit(
                f"No '{prefix}' entries in {archive}. The archive layout has "
                f"changed; expected raw/{{color,grayscale,segmented}}/."
            )
        z.extractall(staging, members=members)

    extracted = staging / "raw" / VARIANT
    source.parent.mkdir(parents=True, exist_ok=True)
    extracted.rename(source)
    shutil.rmtree(staging, ignore_errors=True)

    classes = sorted(p.name for p in source.iterdir() if p.is_dir())
    images = sum(1 for _ in source.rglob("*") if _.is_file())
    print(f"  {images:,} image(s) in {len(classes)} class folder(s) under {source}")

    if not args.keep_archive:
        archive.unlink()
        print(f"  removed {archive} (pass --keep-archive to keep it)")

    print("\nNext: auto-labeller prepare -p", args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
